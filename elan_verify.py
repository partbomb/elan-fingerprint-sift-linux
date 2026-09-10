#!/usr/bin/env python3
import sys
import logging
import os

logging.basicConfig(filename='/tmp/elan_auth.log', level=logging.DEBUG,
                    format='%(asctime)s - %(message)s')

try:
    import usb.core
    import usb.util
    import time
    import numpy as np
    import cv2
except Exception as e:
    logging.error(f"Не удалось импортировать необходимые библиотеки: {e}")
    sys.exit(1)

DIR = '/etc/elan_fingerprint'

# Tuned SIFT for 80x80 fingerprint sensor images
sift = cv2.SIFT_create(
    nfeatures=0,             # unlimited
    nOctaveLayers=5,         # more scale levels for tiny images
    contrastThreshold=0.03,  # lower = more sensitive to subtle ridges
    edgeThreshold=15,        # reject sensor boundary artifacts
    sigma=1.2                # slightly lower for sharper features
)

templates = []

# FLANN matcher parameters for SIFT (float32) descriptors
FLANN_INDEX_KDTREE = 1
flann_index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
flann_search_params = dict(checks=50)


def process_image(img):
    """Advanced fingerprint preprocessing: upscale + CLAHE + unsharp mask.
    Must be identical to C++ process_image() in sift_engine.cpp.
    """
    # 1. Upscale 2x — gives SIFT one more octave on 80x80 images
    upscaled = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    # 2. CLAHE with higher clip limit for ridge contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(upscaled)
    # 3. Unsharp mask — sharpen ridge edges
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 1.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    return sharpened


def compute_quality(img, keypoints):
    """Compute frame quality score: keypoint richness × spatial coverage."""
    if len(keypoints) < 5:
        return 0.0
    pts = np.array([kp.pt for kp in keypoints])
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    kp_area = (max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1])
    img_h, img_w = img.shape[:2]
    img_area = img_h * img_w
    coverage = kp_area / img_area if img_area > 0 else 0.0
    return len(keypoints) * max(coverage, 0.1)


# 1. Load all enrollment templates into memory
try:
    if os.path.exists(DIR):
        for filename in sorted(os.listdir(DIR)):
            if not filename.endswith('.png'):
                continue
            path = os.path.join(DIR, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                clean = process_image(img)
                kp, des = sift.detectAndCompute(clean, None)
                if des is not None and len(kp) >= 5:
                    templates.append((kp, des, filename))
                    logging.debug(f"Загружен эталон {filename}: {len(kp)} keypoints")
except Exception as e:
    logging.error(f"Ошибка загрузки эталонов: {e}")


def verify(captured_img):
    """Verify a captured frame against all enrolled templates.
    Uses FLANN matcher with hybrid scoring (absolute count + ratio).
    """
    captured_clean = process_image(captured_img)
    kp2, des2 = sift.detectAndCompute(captured_clean, None)

    if des2 is None or len(kp2) < 8:
        logging.debug(f"Frame rejected: only {len(kp2) if kp2 else 0} keypoints")
        return False

    # Quality gate: reject low-quality frames before expensive matching
    quality = compute_quality(captured_clean, kp2)
    if quality < 3.0:
        logging.debug(f"Frame rejected: low quality {quality:.2f}")
        return False

    flann = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)

    for kp1, des1, filename in templates:
        if des1 is None or len(des1) < 2:
            continue

        matches = flann.knnMatch(des2, des1, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        # Hybrid scoring: absolute count AND match ratio
        match_ratio = len(good_matches) / len(kp2) if len(kp2) > 0 else 0.0
        logging.info(f"Сверка с {filename}: {len(good_matches)} точек, ratio={match_ratio:.2f}, quality={quality:.1f}")

        if len(good_matches) >= 6 and match_ratio >= 0.25:
            return True

    return False


def main():
    if not templates:
        logging.error("Нет сохраненных эталонов для проверки!")
        sys.exit(1)

    dev = usb.core.find(idVendor=0x04f3, idProduct=0x0c4f)
    if dev is None:
        logging.error("Сканер не найден!")
        sys.exit(1)

    kernel_detached = False
    if dev.is_kernel_driver_active(0):
        try:
            dev.detach_kernel_driver(0)
            kernel_detached = True
        except Exception:
            pass

    try:
        dev.set_configuration()
        EP_OUT, EP_IN = 0x01, 0x82

        # Инициализация сенсора
        dev.write(EP_OUT, [0x40, 0x31])
        time.sleep(0.3)

        # Динамический замер уровня шума
        baseline_samples = []
        for _ in range(5):
            dev.write(EP_OUT, [0x40, 0x3f])
            dev.write(EP_OUT, [0x00, 0x09])
            try:
                data = dev.read(EP_IN, 12800, timeout=500)
                raw_data = np.frombuffer(data, dtype='<u2').reshape((80, 80))
                baseline_samples.append(np.std(raw_data))
            except usb.core.USBError:
                pass
            time.sleep(0.03)

        baseline_std = np.mean(baseline_samples) if baseline_samples else 100.0
        touch_on_threshold = max(250.0, baseline_std * 2.2)

        start_time = time.time()

        while time.time() - start_time < 5.0:
            try:
                dev.write(EP_OUT, [0x40, 0x3f])
                dev.write(EP_OUT, [0x00, 0x09])

                data = dev.read(EP_IN, 12800, timeout=1000)
                raw_data = np.frombuffer(data, dtype='<u2').reshape((80, 80))

                if np.std(raw_data) > touch_on_threshold:
                    captured_img = cv2.normalize(raw_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    if verify(captured_img):
                        logging.info("Авторизация успешна!")
                        sys.exit(0)
            except usb.core.USBError:
                pass

            time.sleep(0.05)

        logging.warning("Ошибка авторизации: совпадений не найдено или таймаут")
        sys.exit(1)

    finally:
        usb.util.dispose_resources(dev)
        if kernel_detached:
            try:
                dev.attach_kernel_driver(0)
            except Exception:
                pass

if __name__ == "__main__":
    main()
