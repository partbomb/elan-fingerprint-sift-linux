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
sed -i 's/max-tries=1/max-tries=3/g' /etc/pam.d/common-auth || true

echo "=== 3. Очистка старого отпечатка и перезапуск fprintd ==="
systemctl stop fprintd || true
rm -rf /var/lib/fprint/"$TARGET_USER"
systemctl start fprintd

echo "=== 4. Проверка драйвера (должно быть 'Elan Trojan SIFT') ==="
fprintd-list "$TARGET_USER"

echo ""
echo "Теперь зарегистрируйте палец — 8 коротких касаний (после каждого появится 'enroll-stage-passed')."
echo "Совет: прикладывайте палец прямо, чуть левее/правее и под небольшим наклоном (20-30°):"
fprintd-enroll -f right-index-finger "$TARGET_USER"

echo ""
echo "=== Проверка разблокировки (просто коснитесь сканера на 0.2-0.5с) ==="
fprintd-verify "$TARGET_USER"
