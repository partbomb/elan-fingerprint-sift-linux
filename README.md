# 🚀 ElanTech (04f3:0c4f) Linux Fingerprint Driver (OpenCV + SIFT)

![Linux](https://img.shields.io/badge/OS-Linux-informational?style=flat&logo=linux&logoColor=white&color=2bbc8a)
![C++](https://img.shields.io/badge/Language-C++-blue?style=flat&logo=c%2B%2B)
![OpenCV](https://img.shields.io/badge/Library-OpenCV-green?style=flat&logo=opencv)
![Status](https://img.shields.io/badge/Status-Working_Perfectly-success)

This repository provides a working, lightning-fast, and highly accurate fingerprint driver for ElanTech sensors (specifically `04f3:0c4f`) on Linux. 

It completely bypasses the broken default `libfprint` image-matching algorithm. Instead, it turns the sensor into a "Match-on-Chip" style device by handling raw USB communication and using **OpenCV (SIFT)** for biometric authentication.

---

## ⚡ Recent Enhancements & Fixes

* **2× Image Upscaling:** Upscale 80×80 → 160×160 before SIFT processing, yielding 2-3× more keypoints per frame.
* **Advanced Preprocessing Pipeline:** CLAHE (clip=3.0) + unsharp mask sharpening for enhanced fingerprint ridge contrast.
* **Tuned SIFT Parameters:** `nOctaveLayers=5`, `contrastThreshold=0.03`, `sigma=1.2` — optimized for tiny sensor images.
* **Image Quality Scoring:** Frames scored by keypoint count × spatial coverage; low-quality (partial touch, smudge) frames are rejected.
* **Multi-Frame Best-of-3 Verification:** Captures up to 3 rapid frames from a single touch, tries matching with the highest quality frame first.
* **FLANN Matcher:** KD-tree based matcher replaces brute-force — ~5× faster on large super-templates.
* **Hybrid Match Scoring:** `good_matches ≥ 6 AND match_ratio ≥ 25%` — adapts to varying keypoint counts across skin conditions.
* **Non-Blocking Async Architecture (`GThread`):** USB capture and SIFT processing in background threads, keeping `fprintd` responsive.
* **RAII LibUSB Lifecycle:** Automatic kernel driver re-attachment and safe interface release.
* **Dynamic Noise Baselining:** Adaptive touch thresholds based on sensor noise sampling.

> ⚠️ **Re-enrollment required:** The new preprocessing pipeline produces different SIFT descriptors. Run `fprintd-delete "$USER"` and re-enroll after upgrading.

---

## 🛑 The Problem
The official Linux `libfprint` driver fails with this sensor because:
1. It assumes the sensor is a "dumb" image provider.
2. It tries to extract standard minutiae from a tiny 80x80 pixel matrix, which almost always fails (`score 0/24`, `verify-no-match`).

## 💡 The Solution
I hacked the driver architecture:
1. **Raw USB Capture:** I use `libusb` to send specific hex commands to wake up the sensor and grab the raw image buffer.
2. **SIFT Feature Extraction:** Instead of standard minutiae, I use OpenCV's Scale-Invariant Feature Transform (SIFT). It finds unique micro-patterns on your fingerprint, regardless of the angle.
3. **The "Super-Template":** During enrollment, the code waits for 5 separate touches, deduplicates descriptors, and concatenates unique keypoints into one reference template.
4. **Fast Verification:** When unlocking, it takes 1 quick frame and compares it against the Super-Template using a strict K-Nearest Neighbors (KNN) matcher.

---

## 🕵️‍♂️ How I Reverse-Engineered the Protocol (Wireshark)
If you want to port this method to another unsupported sensor (e.g., Goodix, Egis), here is how I found the secret USB commands using **Wireshark** on Windows:

1. **Setup:** Install Wireshark with the **USBPcap** module on a Windows machine where the official driver works.
2. **Capture:** Start capturing the USB traffic for your specific USB Root Hub.
3. **Action:** Open Windows Hello, start enrolling a fingerprint, and touch the sensor a few times. Stop the capture.
4. **Analysis:** Filter the Wireshark packets by your device address (e.g., `usb.device_address == 5`).
5. **Finding the Commands:** Look at the packets sent *from* the host *to* the sensor just before I touched it. This is how I found the wake-up commands:
   * `0x40 0x31` (Wake up / Init)
   * `0x40 0x3f` and `0x00 0x09` (Request image data)
6. **Finding the Image Payload:** Look for the largest packets returning from the sensor. I found packets with exactly **12800 bytes** of leftover capture data.
   * *Math check:* `80 x 80 pixels = 6400 pixels`. Since the sensor uses 16-bit grayscale, `6400 * 2 bytes = 12800 bytes`. This confirmed I found the raw image!

---

## 🛠️ Installation Guide (Native C++ libfprint Driver)

**Requirements:** `libusb-1.0-0-dev`, `libopencv-dev`, `meson`, `ninja-build`

1. Clone the official `libfprint` v1.94 repository:
    ```bash
    git clone https://gitlab.freedesktop.org/libfprint/libfprint.git
    cd libfprint
    git checkout v1.94.1
    ```

2. Copy the files from this repo's `native_libfprint_driver` folder into your `libfprint` source code:
   * Put `elan.c`, `sift_engine.cpp`, and `sift_engine.h` into `libfprint/drivers/`.
   * Replace `libfprint/meson.build` and the root `meson.build` with the modified ones provided here.

3. Build and install natively:
    ```bash
    meson setup builddir --wipe
    ninja -C builddir
    sudo ninja -C builddir install
    sudo ldconfig
    sudo systemctl restart fprintd
    ```

4. **Important:** Clear your old broken prints first!
    ```bash
    fprintd-delete "$USER"
    ```

5. Go to your KDE/GNOME system settings and enroll your fingerprint normally! (Touch the sensor 5 times during enrollment).

---

## 🐍 Python Scripts
* `elan_enroll.py`: Interactive CLI tool for enrolling 5 touches with finger-lift detection and CLAHE preprocessing.
* `elan_verify.py`: Quick PAM/standalone verification script using SIFT matching.

---

## Для русскоязычных пользователей
Если у вас не работает сканер отпечатков пальцев ElanTech 04f3:0c4f в Linux (Ubuntu, Fedora, Mint) и стандартный драйвер libfprint выдает ошибку, этот репозиторий поможет решить проблему. Внесены исправления для предотвращения зависания системного интерфейса `fprintd` и корректного сопоставления отпечатков.

