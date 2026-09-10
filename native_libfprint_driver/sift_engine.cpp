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

// Advanced fingerprint preprocessing: upscale + CLAHE + unsharp mask
// Upscaling 80x80 → 160x160 gives SIFT one extra octave to extract features,
// yielding 2-3x more keypoints on tiny sensor images.
Mat process_image(const Mat& img) {
    // 1. Upscale 2x with bicubic interpolation
    Mat upscaled;
    resize(img, upscaled, Size(img.cols * 2, img.rows * 2), 0, 0, INTER_CUBIC);

    // 2. CLAHE with elevated clip limit for fingerprint ridge contrast
    Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8, 8));
    Mat enhanced;
    clahe->apply(upscaled, enhanced);

    // 3. Unsharp mask — sharpen ridge edges for better SIFT keypoint detection
    Mat blurred;
    GaussianBlur(enhanced, blurred, Size(3, 3), 1.0);
    Mat sharpened;
    addWeighted(enhanced, 1.5, blurred, -0.5, 0, sharpened);

    return sharpened;
}

// Create SIFT detector tuned for 80x80 fingerprint sensor images
Ptr<SIFT> create_tuned_sift() {
    return SIFT::create(
        0,      // nfeatures: unlimited — extract everything possible
        5,      // nOctaveLayers: more scale levels to find features in tiny images
        0.03,   // contrastThreshold: lower = more sensitive to subtle ridges
        15,     // edgeThreshold: higher to reject artifacts from sensor boundary
        1.2     // sigma: slightly lower initial blur for sharper features
    );
}

// Compute frame quality score for selecting the best capture
// Higher score = better frame (more keypoints with good spatial coverage)
double compute_quality(const Mat& img, const vector<KeyPoint>& keypoints) {
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

// Estimate baseline noise from idle sensor
static double estimate_baseline(libusb_device_handle *dev,
                                unsigned char *req1, int req1_len,
                                unsigned char *req2, int req2_len,
                                unsigned char *buffer, int buf_len) {
    double baseline_stddev = 0.0;
    int baseline_samples = 0;
    int transferred = 0;

    for (int i = 0; i < 5; ++i) {
        libusb_bulk_transfer(dev, 0x01, req1, req1_len, &transferred, 500);
        libusb_bulk_transfer(dev, 0x01, req2, req2_len, &transferred, 500);
        int res = libusb_bulk_transfer(dev, 0x82, buffer, buf_len, &transferred, 500);
        if (res == 0 && transferred == 12800) {
            Mat raw(80, 80, CV_16UC1, buffer);
            Scalar mean, stddev;
            meanStdDev(raw, mean, stddev);
            baseline_stddev += stddev.val[0];
            baseline_samples++;
        }
        this_thread::sleep_for(chrono::milliseconds(30));
    }

    return (baseline_samples > 0) ? baseline_stddev / baseline_samples : 100.0;
}

// Capture images with finger-on/off detection (for enrollment — 5 distinct touches)
bool capture_usb_images(vector<Mat>& out_images, int required_touches, int timeout_sec) {
    ScopedUsbDevice usb(0x04f3, 0x0c4f);
    if (!usb.is_valid()) return false;

    int transferred = 0;
    unsigned char wake_cmd[] = {0x40, 0x31};
    libusb_bulk_transfer(usb.dev, 0x01, wake_cmd, sizeof(wake_cmd), &transferred, 1000);
    this_thread::sleep_for(chrono::milliseconds(300));

    unsigned char req1[] = {0x40, 0x3f};
    unsigned char req2[] = {0x00, 0x09};
    unsigned char buffer[12800];

    double baseline = estimate_baseline(usb.dev, req1, sizeof(req1), req2, sizeof(req2), buffer, sizeof(buffer));
    double touch_on_threshold = max(250.0, baseline * 2.2);
    double touch_off_threshold = min(touch_on_threshold * 0.6, baseline * 1.4);

    int touches = 0;
    bool waiting_for_off = false;
    auto start_time = chrono::steady_clock::now();

    while (touches < required_touches &&
           chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - start_time).count() < timeout_sec) {

        libusb_bulk_transfer(usb.dev, 0x01, req1, sizeof(req1), &transferred, 1000);
        libusb_bulk_transfer(usb.dev, 0x01, req2, sizeof(req2), &transferred, 1000);

        int res = libusb_bulk_transfer(usb.dev, 0x82, buffer, sizeof(buffer), &transferred, 1000);
        if (res == 0 && transferred == 12800) {
            Mat raw(80, 80, CV_16UC1, buffer);
            Scalar mean, stddev;
            meanStdDev(raw, mean, stddev);

            if (waiting_for_off) {
                if (stddev.val[0] < touch_off_threshold) {
                    waiting_for_off = false;
                    this_thread::sleep_for(chrono::milliseconds(200));
                }
            } else {
                if (stddev.val[0] > touch_on_threshold) {
                    Mat norm_img;
                    normalize(raw, norm_img, 0, 255, NORM_MINMAX, CV_8UC1);
                    out_images.push_back(norm_img);
                    touches++;
                    waiting_for_off = true;
                }
            }
        }
        this_thread::sleep_for(chrono::milliseconds(50));
    }

    return touches > 0;
}

// Capture multiple rapid frames from a single touch (for verification — best-of-N)
bool capture_verify_frames(vector<Mat>& out_images, int max_frames, int timeout_sec) {
    ScopedUsbDevice usb(0x04f3, 0x0c4f);
    if (!usb.is_valid()) return false;

    int transferred = 0;
    unsigned char wake_cmd[] = {0x40, 0x31};
    libusb_bulk_transfer(usb.dev, 0x01, wake_cmd, sizeof(wake_cmd), &transferred, 1000);
    this_thread::sleep_for(chrono::milliseconds(300));

    unsigned char req1[] = {0x40, 0x3f};
    unsigned char req2[] = {0x00, 0x09};
    unsigned char buffer[12800];

    double baseline = estimate_baseline(usb.dev, req1, sizeof(req1), req2, sizeof(req2), buffer, sizeof(buffer));
    double touch_threshold = max(250.0, baseline * 2.2);

    auto start_time = chrono::steady_clock::now();
    bool touch_detected = false;

    while (chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - start_time).count() < timeout_sec) {
        libusb_bulk_transfer(usb.dev, 0x01, req1, sizeof(req1), &transferred, 1000);
        libusb_bulk_transfer(usb.dev, 0x01, req2, sizeof(req2), &transferred, 1000);

        int res = libusb_bulk_transfer(usb.dev, 0x82, buffer, sizeof(buffer), &transferred, 1000);
        if (res == 0 && transferred == 12800) {
            Mat raw(80, 80, CV_16UC1, buffer);
            Scalar mean, stddev;
            meanStdDev(raw, mean, stddev);

            if (stddev.val[0] > touch_threshold) {
                Mat norm_img;
                normalize(raw, norm_img, 0, 255, NORM_MINMAX, CV_8UC1);
                out_images.push_back(norm_img);
                touch_detected = true;

                if ((int)out_images.size() >= max_frames) break;
                // Short delay between rapid captures from same touch
                this_thread::sleep_for(chrono::milliseconds(30));
                continue;
            } else if (touch_detected) {
                break; // Finger lifted, stop capturing
            }
        }
        this_thread::sleep_for(chrono::milliseconds(50));
    }

    return !out_images.empty();
}

// ===================== SIFT ENGINE API =====================

extern "C" {

int sift_engine_init(void) { return 1; }

int sift_engine_enroll(unsigned char** out_data) {
    if (!out_data) return 0;
    *out_data = nullptr;

    vector<Mat> images;
    if (!capture_usb_images(images, 5, 30)) return 0;

    Ptr<SIFT> sift = create_tuned_sift();
    Mat super_des;
    BFMatcher dedupe_matcher(NORM_L2);

    for (const auto& img : images) {
        Mat clean_img = process_image(img);
        vector<KeyPoint> kp;
        Mat des;
        sift->detectAndCompute(clean_img, noArray(), kp, des);

        if (des.empty() || kp.size() < 8) continue;

        // Quality gate: reject partial touches with low spatial coverage
        double quality = compute_quality(clean_img, kp);
        if (quality < 3.0) continue;

        // Deduplicate descriptors against existing super-template
        if (super_des.empty()) {
            super_des = des.clone();
        } else {
            Mat new_unique_des;
            for (int i = 0; i < des.rows; ++i) {
                Mat single_des = des.row(i);
                vector<DMatch> matches;
                dedupe_matcher.match(single_des, super_des, matches);
                if (matches.empty() || matches[0].distance > 100.0f) {
                    new_unique_des.push_back(single_des);
                }
            }
            if (!new_unique_des.empty()) {
                vconcat(super_des, new_unique_des, super_des);
            }
        }
    }

    if (super_des.empty()) return 0;

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
    if (rows <= 0 || rows > 10000 || cols <= 0 || cols > 256 || type < 0 || type > 30) return 0;

    size_t elem_size = CV_ELEM_SIZE(type);
    size_t expected_data_size = (size_t)rows * (size_t)cols * elem_size;
    if (expected_data_size > (size_t)(data_size - 12)) return 0;
    if (12 + (int)expected_data_size != data_size) return 0;

    Mat saved_des(rows, cols, type, (void*)(saved_data + 12));
    Mat saved_des_cloned = saved_des.clone();

    // Capture up to 3 rapid frames from a single touch
    vector<Mat> images;
    if (!capture_verify_frames(images, 3, 10)) return 0;

    Ptr<SIFT> sift = create_tuned_sift();

    // Process all captured frames and assess quality
    struct FrameResult {
        vector<KeyPoint> keypoints;
        Mat descriptors;
        double quality;
    };

    vector<FrameResult> results;
    for (const auto& img : images) {
        FrameResult fr;
        Mat clean_img = process_image(img);
        sift->detectAndCompute(clean_img, noArray(), fr.keypoints, fr.descriptors);

        if (fr.descriptors.empty() || fr.keypoints.size() < 5) continue;

        fr.quality = compute_quality(clean_img, fr.keypoints);
        results.push_back(fr);
    }

    if (results.empty()) return 0;

    // Sort by quality descending — try best frame first
    sort(results.begin(), results.end(), [](const FrameResult& a, const FrameResult& b) {
        return a.quality > b.quality;
    });

    // FLANN matcher — ~5x faster than BFMatcher for large super-templates
    // Build index once, reuse for all frame attempts
    FlannBasedMatcher matcher;
    matcher.add(vector<Mat>{saved_des_cloned});
    matcher.train();

    for (const auto& fr : results) {
        vector<vector<DMatch>> knn_matches;
        matcher.knnMatch(fr.descriptors, knn_matches, 2);

        int good_matches = 0;
        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i].size() >= 2) {
                if (knn_matches[i][0].distance < 0.70f * knn_matches[i][1].distance) {
                    good_matches++;
                }
            }
        }

        // Hybrid scoring: require both absolute count AND ratio
        // This adapts to varying keypoint counts across skin conditions
        double match_ratio = (double)good_matches / (double)fr.keypoints.size();
        if (good_matches >= 6 && match_ratio >= 0.25) {
            return 1;
        }
    }

    return 0;
}

}