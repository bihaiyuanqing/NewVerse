#!/usr/bin/env bash
# 第二步（在 GPU 机上以 root 执行）：安装 NVIDIA 驱动。
# 实测环境：Debian 10 / 内核 5.15.152.ve.8u3-amd64 / 8×L20(46G) / 无外网。
# 可行路径：内网 mirrors.byted.org/nvidia 无 .run，改用 ubuntu2004(focal) 的
# R535 deb 解包——NVIDIA 用户态仅依赖 GLIBC_2.9，buster(2.28) 完全兼容；
# 内核模块用源码树对着本机 headers 编译。
# 用法：bash deploy/02_install_gpu_driver.sh
set -euo pipefail

DRIVER_BRANCH="535"
DRIVER_VER="535.154.05"
MIRROR="http://mirrors.byted.org/nvidia/ubuntu2004/x86_64"
WORK=/tmp/nvidia-driver
KVER="$(uname -r)"

echo "==> [1/7] 下载 focal R535 deb（内核源码 + 用户态）"
mkdir -p "$WORK" && cd "$WORK"
for pkg in \
  "nvidia-kernel-source-${DRIVER_BRANCH}_${DRIVER_VER}-0ubuntu1_amd64.deb" \
  "nvidia-kernel-common-${DRIVER_BRANCH}_${DRIVER_VER}-0ubuntu1_amd64.deb" \
  "nvidia-utils-${DRIVER_BRANCH}_${DRIVER_VER}-0ubuntu1_amd64.deb" \
  "libnvidia-compute-${DRIVER_BRANCH}_${DRIVER_VER}-0ubuntu1_amd64.deb"; do
  [ -f "$pkg" ] || curl -fL --retry 3 -O "$MIRROR/$pkg"
done

echo "==> [2/7] 解包"
for d in ksrc kcommon utils compute; do mkdir -p "$d"; done
dpkg -x "nvidia-kernel-source-${DRIVER_BRANCH}_${DRIVER_VER}-0ubuntu1_amd64.deb" ksrc/
dpkg -x "nvidia-kernel-common-${DRIVER_BRANCH}_${DRIVER_VER}-0ubuntu1_amd64.deb" kcommon/
dpkg -x "nvidia-utils-${DRIVER_BRANCH}_${DRIVER_VER}-0ubuntu1_amd64.deb" utils/
dpkg -x "libnvidia-compute-${DRIVER_BRANCH}_${DRIVER_VER}-0ubuntu1_amd64.deb" compute/

echo "==> [3/7] 修复 ve 内核 GPL-only 符号冲突"
# 非抢占 RCU 内核把 rcu_read_unlock_strict 导出为 GPL-only，专有 nvidia.ko
# 无法通过 modpost；未开 RCU_STRICT_GRACE_PERIOD 时该函数体为空，
# 改为静态内联空函数语义等价，仅影响树外模块编译（备份原头文件）。
RCU_H="/usr/src/linux-headers-${KVER}/include/linux/rcupdate.h"
[ -f "${RCU_H}.newverse.bak" ] || cp "$RCU_H" "${RCU_H}.newverse.bak"
sed -i 's|^void rcu_read_unlock_strict(void);|static inline void rcu_read_unlock_strict(void) { }|' "$RCU_H"
grep -q "static inline void rcu_read_unlock_strict" "$RCU_H"

echo "==> [4/7] 编译内核模块（在源码目录直接 make，勿用 -C M= 方式）"
rm -rf "/usr/src/nvidia-${DRIVER_VER}"
cp -r "ksrc/usr/src/nvidia-${DRIVER_VER}" /usr/src/
cd "/usr/src/nvidia-${DRIVER_VER}"
make clean >/dev/null 2>&1 || true
make -j"$(nproc)"
test -f nvidia.ko && test -f nvidia-uvm.ko

echo "==> [5/7] 安装模块（注意 ve 内核模块目录带 -amd64 后缀，需手动归位）"
make INSTALL_MOD_DIR=extra/nvidia modules_install || true
MOD_DST="/lib/modules/${KVER}/extra/nvidia"
mkdir -p "$MOD_DST"
SRC_BASE="$(find /lib/modules -maxdepth 4 -name 'nvidia.ko' -printf '%h\n' 2>/dev/null | head -1)"
if [ -n "$SRC_BASE" ] && [ "$SRC_BASE" != "$MOD_DST" ]; then
  mv "$SRC_BASE"/nvidia*.ko "$MOD_DST/"
  rm -rf "/lib/modules/${KVER%-amd64}"
fi
depmod -a "$KVER"

echo "==> [6/7] 部署用户态与 udev 规则"
cp -a "$WORK"/utils/usr/bin/* /usr/bin/
cp -a "$WORK"/compute/usr/lib/x86_64-linux-gnu/* /usr/lib/x86_64-linux-gnu/
find "$WORK/kcommon" -name '*.rules' -exec cp -v {} /etc/udev/rules.d/ \;
udevadm control --reload-rules 2>/dev/null || true
ldconfig

echo "==> [7/7] 加载模块并配置开机自动加载"
modprobe nvidia
modprobe nvidia_uvm
cat > /etc/modules-load.d/newverse-nvidia.conf <<'EOF'
nvidia
nvidia_uvm
EOF
nvidia-smi
echo "驱动安装完成：$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) 张 GPU 在线"
