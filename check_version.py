import torch
import sys

print('=' * 30)
print(f'Python Version : {sys.version.split()[0]}')
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Version   : {torch.version.cuda}')

# 檢查 cuDNN
if torch.backends.cudnn.is_available():
    v = torch.backends.cudnn.version()
    # cuDNN 版本通常是整數 (如 8902 -> 8.9.2)
    major = v // 1000
    minor = (v % 1000) // 100
    patch = v % 100
    print(f'cuDNN Version  : {v} ({major}.{minor}.{patch})')
else:
    print('cuDNN Version  : Not Available')

# 檢查 NCCL
if torch.cuda.is_available():
    # NCCL 版本通常是 Tuple (如 (2, 29, 3))
    nccl_ver = torch.cuda.nccl.version()
    print(f'NCCL Version   : {nccl_ver[0]}.{nccl_ver[1]}.{nccl_ver[2]}')
else:
    print('NCCL Version   : Not Available (CUDA not found)')

print('=' * 30)
