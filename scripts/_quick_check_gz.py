#!/usr/bin/env python3
"""快速检查截断 gzip 的可恢复性：只解压头部和前几切片。"""
import struct, zlib
from pathlib import Path

TRUNCATED = Path("/xmlg/Lim/Project4/datasets/amos_mm/imagesTr/amos_5964.nii.gz")
raw = TRUNCATED.read_bytes()
print(f"文件大小: {len(raw)} bytes ({len(raw)/1024/1024:.1f} MB)")

# 增量解压
decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
data = b""
chunk = 1 << 20  # 1MB
try:
    for off in range(0, len(raw), chunk):
        data += decompressor.decompress(raw[off:off+chunk])
except zlib.error as e:
    print(f"解压在 {len(data)} bytes 处停止: {e}")

# 尝试 flush 获取剩余
try:
    data += decompressor.flush()
except:
    pass
print(f"已解压: {len(data)} bytes ({len(data)/1024/1024:.1f} MB)")

# NIfTI1 header
if len(data) < 352:
    print("数据不足")
    exit(1)

# 读取 dim
dim0 = struct.unpack_from("<h", data, 40)[0]
dims = [struct.unpack_from("<h", data, 42 + 2*i)[0] for i in range(dim0)]
print(f"dim: {dims}")

dtype_code = struct.unpack_from("<h", data, 70)[0]
bitpix = struct.unpack_from("<h", data, 72)[0]
vox_offset = struct.unpack_from("<f", data, 108)[0]
print(f"dtype_code={dtype_code}, bitpix={bitpix}, vox_offset={vox_offset}")

shape = tuple(dims[1:])
expected_bytes = 1
for d in shape:
    expected_bytes *= d
expected_bytes *= bitpix // 8
print(f"预期体素数据: {expected_bytes} bytes, shape={shape}")

available = len(data) - int(vox_offset)
print(f"可用体素数据: {available} bytes ({available/expected_bytes*100:.1f}%)")

if len(shape) == 3:
    slice_bytes = shape[0] * shape[1] * (bitpix // 8)
    complete_slices = available // slice_bytes
    print(f"完整 z 切片: {complete_slices}/{shape[2]}")
    print(f"足够64层均匀采样: {complete_slices >= 64}")