#include "sift_engine.h"
#include <opencv2/opencv.hpp>
#include <libusb-1.0/libusb.h>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <algorithm>

using namespace std;
using namespace cv;

// RAII helper for LibUSB device lifecycle & kernel driver management
class ScopedUsbDevice {
public:
    libusb_context *ctx = nullptr;
    libusb_device_handle *dev = nullptr;
    bool kernel_detached = false;
    bool interface_claimed = false;

    // Non-copyable, non-movable to prevent double-free
    ScopedUsbDevice(const ScopedUsbDevice&) = delete;
    ScopedUsbDevice& operator=(const ScopedUsbDevice&) = delete;
    ScopedUsbDevice(ScopedUsbDevice&&) = delete;
    ScopedUsbDevice& operator=(ScopedUsbDevice&&) = delete;

    ScopedUsbDevice(uint16_t vid, uint16_t pid) {
        if (libusb_init(&ctx) < 0) return;
        dev = libusb_open_device_with_vid_pid(ctx, vid, pid);
        if (!dev) return;

        if (libusb_kernel_driver_active(dev, 0) == 1) {
            if (libusb_detach_kernel_driver(dev, 0) == 0) {
                kernel_detached = true;
            }
        }
        libusb_set_configuration(dev, 1);
        if (libusb_claim_interface(dev, 0) == 0) {
            interface_claimed = true;
        }
    }

    ~ScopedUsbDevice() {
        if (dev) {
            if (interface_claimed) {
                libusb_release_interface(dev, 0);
            }
            if (kernel_detached) {
                libusb_attach_kernel_driver(dev, 0);
            }
            libusb_close(dev);
        }
        if (ctx) {
            libusb_exit(ctx);
        }
    }

    bool is_valid() const {
        return dev != nullptr && interface_claimed;
    }
};

// ===================== IMAGE PROCESSING PIPELINE =====================

// Advanced fingerprint preprocessing: denoise + 2x upscale + CLAHE
// Pre-denoising prevents bicubic upscaling and CLAHE from amplifying sensor
// pixel noise, while 2x upscaling (80x80 -> 160x160) gives SIFT an extra octave.
static Mat process_image(const Mat& img) {
    // 1. Light 3x3 Gaussian denoise (sigma=0.8) preserves diagonal ridges under 20-40 deg rotation
    Mat denoised;
    GaussianBlur(img, denoised, Size(3, 3), 0.8);

    // 2. Upscale 2x with bicubic interpolation (80x80 -> 160x160)
    Mat upscaled;
    resize(denoised, upscaled, Size(img.cols * 2, img.rows * 2), 0, 0, INTER_CUBIC);

    // 3. CLAHE with 4x4 tile grid (40x40px tiles) for rotation-stable ridge contrast
    Ptr<CLAHE> clahe = createCLAHE(2.5, Size(4, 4));
    Mat enhanced;
    clahe->apply(upscaled, enhanced);

    return enhanced;
}

// Rotate an 80x80 sensor frame around its center to align rotated finger touches
static Mat rotate_sensor_frame(const Mat& img, double angle_deg) {
    if (std::abs(angle_deg) < 0.1) return img;
    Point2f center(img.cols / 2.0f, img.rows / 2.0f);
    Mat rot_mat = getRotationMatrix2D(center, angle_deg, 1.0);
    Mat rotated;
    warpAffine(img, rotated, rot_mat, img.size(), INTER_LINEAR, BORDER_REFLECT);
    return rotated;
}

// Create SIFT detector tuned for upscaled 160x160 fingerprint sensor images
static Ptr<SIFT> create_tuned_sift() {
    return SIFT::create(
        150,    // nfeatures: cap to top 150 strongest keypoints to prevent ratio dilution
        4,      // nOctaveLayers: scale levels suited for 160x160 ridge patterns
        0.035,  // contrastThreshold: sensitive to diagonal/rotated ridges while rejecting noise
        12,     // edgeThreshold: reject sensor boundary artifacts while keeping ridge endings
        1.6     // sigma: standard SIFT base blur for stable gradient orientations
    );
}

// Compute frame quality score for selecting the best capture
// Higher score = better frame (more keypoints with good spatial coverage)
static double compute_quality(const Mat& img, const vector<KeyPoint>& keypoints) {
    if (keypoints.size() < 5) return 0.0;

    // Compute spatial coverage: how much of the image the keypoints span
    float min_x = FLT_MAX, min_y = FLT_MAX, max_x = 0, max_y = 0;
    for (const auto& kp : keypoints) {
        min_x = min(min_x, kp.pt.x);
        min_y = min(min_y, kp.pt.y);
        max_x = max(max_x, kp.pt.x);
        max_y = max(max_y, kp.pt.y);
    }
    double kp_area = (double)(max_x - min_x) * (max_y - min_y);
    double img_area = (double)img.cols * img.rows;
    double coverage = (img_area > 0) ? kp_area / img_area : 0.0;

    // Score = keypoint richness × spatial coverage (partial touch = low coverage)
    return (double)keypoints.size() * max(coverage, 0.1);
}

// ===================== USB CAPTURE =====================

// Validate raw 16-bit sensor frame to reject empty, saturated, or corrupted USB buffers
static bool is_valid_raw_frame(const Mat& raw, double& out_stddev) {
    if (raw.empty()) return false;
    double min_val = 0.0, max_val = 0.0;
    minMaxLoc(raw, &min_val, &max_val);
    if (max_val <= min_val) return false;

    Scalar mean, stddev;
    meanStdDev(raw, mean, stddev);
    out_stddev = stddev.val[0];
    // Reject all-zero or rail-saturated corrupted transfers
    if (mean.val[0] < 1.0 || mean.val[0] > 65000.0) return false;
    return true;
}

// Helper to request and read a single 80x80 16-bit frame over USB
static bool read_sensor_frame(libusb_device_handle *dev, unsigned char *buffer, int buf_len) {
    unsigned char req1[] = {0x40, 0x3f};
    unsigned char req2[] = {0x00, 0x09};
    int transferred = 0;

    if (libusb_bulk_transfer(dev, 0x01, req1, sizeof(req1), &transferred, 1000) != 0) return false;
    if (libusb_bulk_transfer(dev, 0x01, req2, sizeof(req2), &transferred, 1000) != 0) return false;

    memset(buffer, 0, buf_len);
    int res = libusb_bulk_transfer(dev, 0x82, buffer, buf_len, &transferred, 1000);
    return (res == 0 && transferred == 12800);
}

// Estimate baseline noise from idle sensor (fast 2-sample probe).
// Ignores samples with stddev >= 220 so that a finger already resting on the
// sensor at startup never inflates touch_threshold.
static double estimate_baseline(libusb_device_handle *dev,
                                unsigned char *buffer, int buf_len) {
    double baseline_stddev = 0.0;
    int baseline_samples = 0;

    for (int i = 0; i < 2; ++i) {
        if (read_sensor_frame(dev, buffer, buf_len)) {
            Mat raw(80, 80, CV_16UC1, buffer);
            double stddev = 0.0;
            if (is_valid_raw_frame(raw, stddev) && stddev < 220.0) {
                baseline_stddev += stddev;
                baseline_samples++;
            }
        }
        this_thread::sleep_for(chrono::milliseconds(10));
    }

    return (baseline_samples > 0) ? baseline_stddev / baseline_samples : 100.0;
}

// Capture images with finger-on/off detection, in-loop quality gate, and live stage progress
static bool capture_usb_images(vector<Mat>& out_images, int required_touches, int timeout_sec,
                               sift_enroll_progress_cb progress_cb, void* user_data) {
    ScopedUsbDevice usb(0x04f3, 0x0c4f);
    if (!usb.is_valid()) return false;

    int transferred = 0;
    unsigned char wake_cmd[] = {0x40, 0x31};
    libusb_bulk_transfer(usb.dev, 0x01, wake_cmd, sizeof(wake_cmd), &transferred, 1000);
    this_thread::sleep_for(chrono::milliseconds(50));

    unsigned char buffer[12800];
    double baseline = estimate_baseline(usb.dev, buffer, sizeof(buffer));
    double touch_on_threshold = max(240.0, baseline * 2.1);
    double touch_off_threshold = min(touch_on_threshold * 0.65, baseline * 1.45);

    Ptr<SIFT> sift = create_tuned_sift();
    int touches = 0;
    bool waiting_for_off = false;
    bool touch_settling = false;
    auto start_time = chrono::steady_clock::now();

    while (touches < required_touches &&
           chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - start_time).count() < timeout_sec) {

        if (read_sensor_frame(usb.dev, buffer, sizeof(buffer))) {
            Mat raw(80, 80, CV_16UC1, buffer);
            double stddev = 0.0;
            if (is_valid_raw_frame(raw, stddev)) {
                if (waiting_for_off) {
                    if (stddev < touch_off_threshold) {
                        waiting_for_off = false;
                        touch_settling = false;
                        this_thread::sleep_for(chrono::milliseconds(60));
                    }
                } else if (stddev > touch_on_threshold) {
                    if (!touch_settling) {
                        touch_settling = true;
                        this_thread::sleep_for(chrono::milliseconds(25));
                        continue;
                    }

                    Mat norm_img;
                    normalize(raw, norm_img, 0, 255, NORM_MINMAX, CV_8UC1);

                    Mat clean_img = process_image(norm_img);
                    vector<KeyPoint> kp;
                    Mat des;
                    sift->detectAndCompute(clean_img, noArray(), kp, des);

                    if (!des.empty() && kp.size() >= 8 && compute_quality(clean_img, kp) >= 2.2) {
                        out_images.push_back(norm_img.clone());
                        touches++;
                        if (progress_cb) {
                            progress_cb(touches, user_data);
                        }
                        waiting_for_off = true;
                        touch_settling = false;
                    }
                } else {
                    touch_settling = false;
                }
            }
        }
        this_thread::sleep_for(chrono::milliseconds(25));
    }

    return touches >= 5;
}

// Check if a single 8-bit sensor frame matches the enrolled super-template.
// Evaluates 0 deg first (~3ms) for instant unlock, then tests rotated angles
// (-20, +20, -38, +38, -55, +55 deg) so a finger rotated by 20-60 deg aligns cleanly.
static bool match_single_frame(const Mat& norm_img, const Mat& saved_des, Ptr<SIFT>& sift, BFMatcher& matcher) {
    static const double angles[] = {0.0, -20.0, 20.0, -38.0, 38.0, -55.0, 55.0};

    for (double angle : angles) {
        Mat rotated = rotate_sensor_frame(norm_img, angle);
        Mat clean_img = process_image(rotated);
        vector<KeyPoint> keypoints;
        Mat descriptors;
        sift->detectAndCompute(clean_img, noArray(), keypoints, descriptors);

        if (descriptors.empty() || keypoints.size() < 6) continue;
        if (compute_quality(clean_img, keypoints) < 1.8) continue;

        vector<vector<DMatch>> knn_matches;
        matcher.knnMatch(descriptors, saved_des, knn_matches, 2);

        int good_matches = 0;
        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i].size() >= 2) {
                if (knn_matches[i][0].distance < 0.77f * knn_matches[i][1].distance) {
                    good_matches++;
                }
            }
        }

        double match_ratio = (double)good_matches / (double)keypoints.size();
        if ((good_matches >= 5 && match_ratio >= 0.06) || good_matches >= 7) {
            return true;
        }
    }
    return false;
}

// ===================== SIFT ENGINE API =====================

extern "C" {

int sift_engine_init(void) { return 1; }

int sift_engine_enroll(unsigned char** out_data, sift_enroll_progress_cb progress_cb, void* user_data) {
    if (!out_data) return 0;
    *out_data = nullptr;

    vector<Mat> images;
    if (!capture_usb_images(images, 8, 60, progress_cb, user_data)) return 0;

    Ptr<SIFT> sift = create_tuned_sift();
    Mat super_des;
    BFMatcher dedupe_matcher(NORM_L2);

    for (const auto& img : images) {
        Mat clean_img = process_image(img);
        vector<KeyPoint> kp;
        Mat des;
        sift->detectAndCompute(clean_img, noArray(), kp, des);

        if (des.empty() || kp.size() < 8) continue;

        double quality = compute_quality(clean_img, kp);
        if (quality < 2.2) continue;

        // Deduplicate descriptors against existing super-template (threshold 200.0f
        // prevents cross-touch self-cancellation in Lowe's ratio test)
        if (super_des.empty()) {
            super_des = des.clone();
        } else {
            Mat new_unique_des;
            for (int i = 0; i < des.rows; ++i) {
                Mat single_des = des.row(i);
                vector<DMatch> matches;
                dedupe_matcher.match(single_des, super_des, matches);
                if (matches.empty() || matches[0].distance > 200.0f) {
                    new_unique_des.push_back(single_des);
                }
            }
            if (!new_unique_des.empty()) {
                vconcat(super_des, new_unique_des, super_des);
            }
        }
    }

    if (super_des.empty() || super_des.rows < 2) return 0;

    int data_bytes = super_des.total() * super_des.elemSize();
    int header_size = 3 * sizeof(int);
    int total_size = header_size + data_bytes;

    *out_data = (unsigned char*)malloc(total_size);
    if (!*out_data) return 0;

    int* header = (int*)(*out_data);
    header[0] = super_des.rows;
    header[1] = super_des.cols;
    header[2] = super_des.type();

    memcpy(*out_data + header_size, super_des.data, data_bytes);
    return total_size;
}

int sift_engine_verify(const unsigned char* saved_data, int data_size) {
    if (!saved_data || data_size < 12) return 0;

    const int* header = (const int*)saved_data;
    int rows = header[0], cols = header[1], type = header[2];

    // Sanity bounds: reject corrupted/malicious data before arithmetic
    if (rows < 2 || rows > 10000 || cols <= 0 || cols > 256 || type < 0 || type > 30) return 0;

    size_t elem_size = CV_ELEM_SIZE(type);
    size_t expected_data_size = (size_t)rows * (size_t)cols * elem_size;
    if (expected_data_size > (size_t)(data_size - 12)) return 0;
    if (12 + (int)expected_data_size != data_size) return 0;

    Mat saved_des(rows, cols, type, (void*)(saved_data + 12));
    Mat saved_des_cloned = saved_des.clone();

    ScopedUsbDevice usb(0x04f3, 0x0c4f);
    if (!usb.is_valid()) return 0;

    int transferred = 0;
    unsigned char wake_cmd[] = {0x40, 0x31};
    libusb_bulk_transfer(usb.dev, 0x01, wake_cmd, sizeof(wake_cmd), &transferred, 1000);
    this_thread::sleep_for(chrono::milliseconds(50));

    unsigned char buffer[12800];
    double baseline = estimate_baseline(usb.dev, buffer, sizeof(buffer));
    double touch_threshold = max(240.0, baseline * 2.1);

    Ptr<SIFT> sift = create_tuned_sift();
    BFMatcher matcher(NORM_L2);

    auto start_time = chrono::steady_clock::now();
    bool touch_detected = false;
    int frames_checked = 0;

    // Continuous multi-angle streaming verification:
    // Matches on the first settled frame (~0.1-0.2s), tests +/-20, +/-38, +/-55 deg rotations,
    // and allows finger repositioning during the touch window without premature exit.
    while (chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - start_time).count() < 10) {
        if (read_sensor_frame(usb.dev, buffer, sizeof(buffer))) {
            Mat raw(80, 80, CV_16UC1, buffer);
            double stddev = 0.0;

            if (is_valid_raw_frame(raw, stddev) && stddev > touch_threshold) {
                if (!touch_detected) {
                    touch_detected = true;
                    this_thread::sleep_for(chrono::milliseconds(25));
                    continue;
                }

                Mat norm_img;
                normalize(raw, norm_img, 0, 255, NORM_MINMAX, CV_8UC1);
                frames_checked++;

                if (match_single_frame(norm_img, saved_des_cloned, sift, matcher)) {
                    return 1; // Instant unlock!
                }

                if (frames_checked >= 20) break;
                this_thread::sleep_for(chrono::milliseconds(25));
                continue;
            } else {
                touch_detected = false;
                if (frames_checked >= 12) break;
            }
        }
        this_thread::sleep_for(chrono::milliseconds(25));
    }

    return 0;
}

}