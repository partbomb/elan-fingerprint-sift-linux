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

# Tuned SIFT for upscaled 160x160 fingerprint sensor images (parity with C++ & enroll.py)
sift = cv2.SIFT_create(
    nfeatures=150,           # cap to top 150 strongest keypoints to prevent ratio dilution
    nOctaveLayers=4,         # scale levels suited for 160x160 ridge patterns
    contrastThreshold=0.035, # sensitive to diagonal/rotated ridges while rejecting noise
    edgeThreshold=12,        # reject sensor boundary artifacts
    sigma=1.6                # standard SIFT base blur for stable gradient orientations
)

templates = []
super_des = None
bf_matcher = cv2.BFMatcher(cv2.NORM_L2)


def process_image(img):
    """Advanced fingerprint preprocessing: denoise + 2x upscale + CLAHE.
    Must be identical to C++ process_image() in sift_engine.cpp.
    """
    # 1. Light 3x3 Gaussian denoise preserves diagonal ridges under 20-40 deg rotation
    denoised = cv2.GaussianBlur(img, (3, 3), 0.8)
    # 2. Upscale 2x — gives SIFT one more octave on 80x80 images
    upscaled = cv2.resize(denoised, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    # 3. CLAHE with 4x4 tile grid (40x40px tiles) for rotation-stable ridge contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    enhanced = clahe.apply(upscaled)
    return enhanced


def rotate_sensor_frame(img, angle_deg):
    """Rotate an 80x80 sensor frame around its center to align rotated finger touches."""
    if abs(angle_deg) < 0.1:
        return img
    h, w = img.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


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


def is_valid_raw_frame(raw_data):
    """Reject empty, constant, or rail-saturated USB frames."""
    if raw_data is None or raw_data.size != 6400:
        return False
    min_val, max_val = int(raw_data.min()), int(raw_data.max())
    if max_val <= min_val:
        return False
    mean_val = float(np.mean(raw_data))
    if mean_val < 1.0 or mean_val > 65000.0:
        return False
    return True


# 1. Load all enrollment templates into memory and build deduplicated super-template
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
                if des is not None and len(kp) >= 6:
                    templates.append((kp, des, filename))
                    if super_des is None:
                        super_des = des.copy()
                    else:
                        unique_rows = []
                        for i in range(len(des)):
                            single_des = des[i:i+1]
                            m = bf_matcher.match(single_des, super_des)
                            if not m or m[0].distance > 200.0:
                                unique_rows.append(single_des)
                        if unique_rows:
                            super_des = np.vstack([super_des] + unique_rows)
                    logging.debug(f"Загружен эталон {filename}: {len(kp)} keypoints")
except Exception as e:
    logging.error(f"Ошибка загрузки эталонов: {e}")


def verify(captured_img):
    """Verify a captured frame against the super-template across multiple rotation angles.
    Uses exact L2 BFMatcher with multi-angle alignment (-55 to +55 deg).
    """
    for angle in (0.0, -20.0, 20.0, -38.0, 38.0, -55.0, 55.0):
        rotated = rotate_sensor_frame(captured_img, angle)
        captured_clean = process_image(rotated)
        kp2, des2 = sift.detectAndCompute(captured_clean, None)

        if des2 is None or len(kp2) < 6:
            continue

        quality = compute_quality(captured_clean, kp2)
        if quality < 1.8:
            continue

        # Check against combined deduplicated super-template (parity with C++ driver)
        if super_des is not None and len(super_des) >= 2:
            matches = bf_matcher.knnMatch(des2, super_des, k=2)
            good_matches = [
                m[0] for m in matches
                if len(m) == 2 and m[0].distance < 0.77 * m[1].distance
            ]
            match_ratio = len(good_matches) / len(kp2)
            logging.info(f"Сверка (угол {angle:.0f}°): {len(good_matches)} точек, ratio={match_ratio:.2f}, quality={quality:.1f}")
            if (len(good_matches) >= 5 and match_ratio >= 0.06) or len(good_matches) >= 7:
                return True

        # Fallback: check individual enrollment templates
        for kp1, des1, filename in templates:
            if des1 is None or len(des1) < 2:
                continue

            matches = bf_matcher.knnMatch(des2, des1, k=2)
            good_matches = [
                m[0] for m in matches
                if len(m) == 2 and m[0].distance < 0.77 * m[1].distance
            ]

            match_ratio = len(good_matches) / len(kp2) if len(kp2) > 0 else 0.0
            if (len(good_matches) >= 5 and match_ratio >= 0.06) or len(good_matches) >= 7:
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
        time.sleep(0.05)

        # Быстрый замер уровня шума (игнорируем кадры, если палец уже лежит на сенсоре)
        baseline_samples = []
        for _ in range(2):
            dev.write(EP_OUT, [0x40, 0x3f])
            dev.write(EP_OUT, [0x00, 0x09])
            try:
                data = dev.read(EP_IN, 12800, timeout=500)
                if len(data) != 12800:
                    continue
                raw_data = np.frombuffer(data, dtype='<u2').reshape((80, 80))
                if is_valid_raw_frame(raw_data):
                    std_val = np.std(raw_data)
                    if std_val < 220.0:
                        baseline_samples.append(std_val)
            except usb.core.USBError:
                pass
            time.sleep(0.01)

        baseline_std = np.mean(baseline_samples) if baseline_samples else 100.0
        touch_on_threshold = max(250.0, baseline_std * 2.2)

        start_time = time.time()
        touch_settled = False

        while time.time() - start_time < 5.0:
            try:
                dev.write(EP_OUT, [0x40, 0x3f])
                dev.write(EP_OUT, [0x00, 0x09])

                data = dev.read(EP_IN, 12800, timeout=1000)
                if len(data) != 12800:
                    continue
                raw_data = np.frombuffer(data, dtype='<u2').reshape((80, 80))
                if not is_valid_raw_frame(raw_data):
                    continue

                if np.std(raw_data) > touch_on_threshold:
                    if not touch_settled:
                        touch_settled = True
                        time.sleep(0.025)
                        continue
                    captured_img = cv2.normalize(raw_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    if verify(captured_img):
                        logging.info("Авторизация успешна!")
                        sys.exit(0)
                else:
                    touch_settled = False
            except usb.core.USBError:
                pass

            time.sleep(0.025)

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
