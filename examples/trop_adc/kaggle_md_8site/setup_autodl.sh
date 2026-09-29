#!/bin/bash
# AutoDL OpenMM 环境一键安装
# 在 AutoDL SSH 终端执行：bash setup_autodl.sh
set -e

echo "=== Step 1: 安装 Miniforge ==="
if [ ! -d ~/miniforge3 ]; then
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p ~/miniforge3
    echo 'export PATH="$HOME/miniforge3/bin:$PATH"' >> ~/.bashrc
fi
export PATH="$HOME/miniforge3/bin:$PATH"

echo "=== Step 2: 创建 openmm 环境 ==="
if ! conda env list | grep -q openmm; then
    conda create -n openmm python=3.11 openmm numpy -c conda-forge -y
fi

echo "=== Step 3: 验证 CUDA ==="
eval "$(conda shell.bash hook)"
conda activate openmm
python -c "
import openmm
print(f'OpenMM {openmm.__version__}')
for i in range(openmm.Platform.getNumPlatform()):
    p = openmm.Platform.getPlatform(i)
    print(f'  Platform {i}: {p.getName()}')
"

echo ""
echo "=== 安装完成 ==="
echo "运行: conda activate openmm && python run_md.py"
