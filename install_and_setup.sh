#!/usr/bin/env bash
set -e

TARGET_USER="${SUDO_USER:-$USER}"
LIBFPRINT_BUILD="/home/kuanysh/Downloads/libfprint/builddir"

echo "=== 1. Установка быстрого драйвера Elan Trojan SIFT (прикладывание 0.2с вместо свайпа) ==="
ninja -C "$LIBFPRINT_BUILD" install
cp -f "$LIBFPRINT_BUILD/libfprint/libfprint-2.so.2.0.0" /usr/lib/x86_64-linux-gnu/libfprint-2.so.2.0.0
ldconfig

echo "=== 2. Включение разблокировки по отпечатку для экрана блокировки KDE, входа и sudo ==="
pam-auth-update --enable fprintd

echo "=== 3. Очистка старого swipe-отпечатка и перезапуск fprintd ==="
systemctl stop fprintd || true
rm -rf /var/lib/fprint/"$TARGET_USER"
systemctl start fprintd

echo "=== 4. Проверка драйвера (должно быть 'Elan Trojan SIFT') ==="
fprintd-list "$TARGET_USER"

echo ""
echo "Теперь зарегистрируйте палец (просто ПРИКЛАДЫВАЙТЕ на 0.3с 5 раз, НЕ нужно проводить!):"
su - "$TARGET_USER" -c "fprintd-enroll -f right-index-finger"

echo ""
echo "=== Проверка разблокировки (просто коснитесь сканера на 0.2-0.5с) ==="
su - "$TARGET_USER" -c "fprintd-verify"
