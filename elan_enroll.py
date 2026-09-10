#!/usr/bin/env python3
import usb.core
import usb.util
import time
import numpy as np
import cv2
import sys
import os

DIR = '/etc/elan_fingerprint'
REQUIRED_TOUCHES = 5

# Tuned SIFT for 80x80 fingerprint sensor images (same as C++ and verify.py)
sift = cv2.SIFT_create(
    nfeatures=0,
    nOctaveLayers=5,
    contrastThreshold=0.03,
    edgeThreshold=15,
    sigma=1.2
)


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


def main():
    if not os.path.exists(DIR):
        try:
            os.makedirs(DIR)
        except PermissionError:
            print(f"Ошибка: Недостаточно прав для создания директории {DIR}. Запустите через sudo.")
            sys.exit(1)

    dev = usb.core.find(idVendor=0x04f3, idProduct=0x0c4f)
    if dev is None:
        print("Сканер ElanTech (04f3:0c4f) не найден!")
        sys.exit(1)

    kernel_detached = False
    if dev.is_kernel_driver_active(0):
        try:
            dev.detach_kernel_driver(0)
            kernel_detached = True
        except Exception as e:
            print(f"Предупреждение: Не удалось отключить драйвер ядра: {e}")

    try:
        dev.set_configuration()
        EP_OUT, EP_IN = 0x01, 0x82

        print("Сканер готов.")
        print(f"Необходимо приложить палец {REQUIRED_TOUCHES} раз для завершения регистрации.")

        dev.write(EP_OUT, [0x40, 0x31])
        time.sleep(0.3)

        # Baseline noise estimation
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
        touch_off_threshold = min(touch_on_threshold * 0.6, baseline_std * 1.4)

        touches = 0
        waiting_for_off = False
        start_time = time.time()

        while touches < REQUIRED_TOUCHES and (time.time() - start_time) < 60.0:
            try:
                dev.write(EP_OUT, [0x40, 0x3f])
                dev.write(EP_OUT, [0x00, 0x09])
                data = dev.read(EP_IN, 12800, timeout=1000)
                raw_data = np.frombuffer(data, dtype='<u2').reshape((80, 80))
                stddev = np.std(raw_data)

                if waiting_for_off:
                    if stddev < touch_off_threshold:
                        waiting_for_off = False
                        print("Палец убран. Приложите еще раз...")
                        time.sleep(0.2)
                else:
                    if stddev > touch_on_threshold:
                        img_8bit = cv2.normalize(raw_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                        # Quality gate: check if frame has enough features
                        processed = process_image(img_8bit)
                        kp, des = sift.detectAndCompute(processed, None)

                        if des is not None and len(kp) >= 8:
                            quality = compute_quality(processed, kp)
                            if quality >= 3.0:
                                # Save raw 8-bit image (processing applied during verification)
                                existing_files = [f for f in os.listdir(DIR) if f.startswith('template_') and f.endswith('.png')]
                                max_idx = 0
                                for f in existing_files:
                                    try:
                                        idx = int(f[len('template_'):-len('.png')])
                                        if idx > max_idx:
                                            max_idx = idx
                                    except ValueError:
                                        pass
                                next_idx = max_idx + 1
                                save_path = os.path.join(DIR, f'template_{next_idx}.png')

                                cv2.imwrite(save_path, img_8bit)
                                touches += 1
                                print(f"[{touches}/{REQUIRED_TOUCHES}] Эталон сохранен ({len(kp)} keypoints, quality={quality:.1f})")
                                waiting_for_off = True
                            else:
                                print(f"Низкое качество кадра (quality={quality:.1f}). Приложите палец ровнее...")
                        else:
                            print(f"Недостаточно деталей ({len(kp) if kp else 0} keypoints). Приложите палец плотнее...")

            except usb.core.USBError:
                pass

            time.sleep(0.05)

        if touches == REQUIRED_TOUCHES:
            print("Успех! Все эталоны пальца успешно зарегистрированы.")
        else:
            print("Превышено время ожидания регистрации.")

    finally:
        usb.util.dispose_resources(dev)
        if kernel_detached:
            try:
                dev.attach_kernel_driver(0)
            except Exception:
                pass

if __name__ == "__main__":
    main()
