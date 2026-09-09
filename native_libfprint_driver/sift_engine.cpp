#include "sift_engine.h"
#include <opencv2/opencv.hpp>
#include <libusb-1.0/libusb.h>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

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

Mat process_image(const Mat& img) {
    Mat blur, out;
    GaussianBlur(img, blur, Size(5, 5), 0);
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(blur, out);
    return out;
}

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
    
    // Estimate initial baseline noise stddev (idle sensor)
    double baseline_stddev = 0.0;
    int baseline_samples = 0;

    for (int i = 0; i < 5; ++i) {
        libusb_bulk_transfer(usb.dev, 0x01, req1, sizeof(req1), &transferred, 500);
        libusb_bulk_transfer(usb.dev, 0x01, req2, sizeof(req2), &transferred, 500);
        int res = libusb_bulk_transfer(usb.dev, 0x82, buffer, sizeof(buffer), &transferred, 500);
        if (res == 0 && transferred == 12800) {
            Mat raw(80, 80, CV_16UC1, buffer);
            Scalar mean, stddev;
            meanStdDev(raw, mean, stddev);
            baseline_stddev += stddev.val[0];
            baseline_samples++;
        }
        this_thread::sleep_for(chrono::milliseconds(30));
    }

    if (baseline_samples > 0) {
        baseline_stddev /= baseline_samples;
    } else {
        baseline_stddev = 100.0;
    }

    // Dynamic thresholds relative to baseline noise
    double touch_on_threshold = max(250.0, baseline_stddev * 2.2);
    double touch_off_threshold = min(touch_on_threshold * 0.6, baseline_stddev * 1.4);

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

extern "C" {

int sift_engine_init(void) { return 1; }

int sift_engine_enroll(unsigned char** out_data) {
    if (!out_data) return 0;
    *out_data = nullptr;

    vector<Mat> images;
    if (!capture_usb_images(images, 5, 30)) return 0;

    Ptr<SIFT> sift = SIFT::create();
    Mat super_des;
    BFMatcher dedupe_matcher(NORM_L2);

    for (const auto& img : images) {
        Mat clean_img = process_image(img);
        vector<KeyPoint> kp;
        Mat des;
        sift->detectAndCompute(clean_img, noArray(), kp, des);

        if (des.empty() || kp.size() < 5) continue;

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

    vector<Mat> images;
    if (!capture_usb_images(images, 1, 10)) return 0;

    Mat clean_img = process_image(images[0]);
    Ptr<SIFT> sift = SIFT::create();
    vector<KeyPoint> kp_current;
    Mat current_des;
    sift->detectAndCompute(clean_img, noArray(), kp_current, current_des);

    if (current_des.empty() || kp_current.size() < 5) return 0;

    // Query current fresh image keypoints against saved enrolled database
    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> knn_matches;
    matcher.knnMatch(current_des, saved_des_cloned, knn_matches, 2);

    int good_matches = 0;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i].size() >= 2) {
            if (knn_matches[i][0].distance < 0.70f * knn_matches[i][1].distance) {
                good_matches++;
            }
        }
    }

    return (good_matches >= 8) ? 1 : 0;
}

}