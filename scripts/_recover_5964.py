#!/usr/bin/env python3
"""尝试从截断的 amos_5964.nii.gz 恢复尽可能多的体素数据。"""
import gzip
import hashlib
import io
import struct
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np

TRUNCATED = Path("/xmlg/Lim/Project4/datasets/amos_mm/imagesTr/amos_5964.nii.gz")
RECOVERED = Path("/xmlg/Lim/Project4/outputs/amos_mm/repair_amos_5964/amos_5964_recovered.nii.gz")

def main():
    raw = TRUNCATED.read_bytes()
    print(f"截断文件大小: {len(raw)} bytes")

    # 尝试用 zlib 增量解压，忽略末尾截断
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        data = decompressor.decompress(raw)
        remaining = decompressor.flush()
        data += remaining
        print(f"完整解压成功，未截断! 大小: {len(data)} bytes")
        print(f"到达流结尾: {decompressor.eof}")
    except zlib.error as e:
        # 增量解压：逐块解压尽可能多
        data = b""
        chunk_size = 65536
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        for offset in range(0, len(raw), chunk_size):
            try:
                chunk = decompressor.decompress(raw[offset:offset + chunk_size])
                data += chunk
            except zlib.error:
                print(f"解压在偏移 {offset} 处停止，已恢复 {len(data)} bytes")
                break
        print(f"部分解压: {len(data)} bytes")

    # 解析 NIfTI1 header (348 bytes)
    if len(data) < 348:
        print("数据太少，无法解析 NIfTI 头")
        return False

    header = nib.Nifti1Header.from_bytes(data[:348])
    shape = header.get_data_shape()
    dtype = header.get_data_dtype()
    print(f"NIfTI 头: shape={shape}, dtype={dtype}")

    expected_voxels = 1
    for s in shape:
        expected_voxels *= s
    expected_bytes = expected_voxels * dtype.itemsize
    print(f"预期体素数据: {expected_voxels} voxels × {dtype.itemsize} bytes = {expected_bytes} bytes")

    # NIfTI1 header 后可能有扩展头，数据偏移从 header 的 vox_offset 字段读取
    vox_offset = int.from_bytes(data[108:112], "little")
    print(f"体素数据偏移: {vox_offset}")

    available_data = len(data) - vox_offset
    available_voxels = available_data // dtype.itemsize
    print(f"可用体素数据: {available_data} bytes = {available_voxels} voxels")
    print(f"完整体素比例: {available_voxels / expected_voxels * 100:.1f}%")

    # 检查有多少完整的 z 切片
    if len(shape) == 3:
        xy = shape[0] * shape[1]
        complete_slices = available_voxels // xy
        print(f"完整 z 切片: {complete_slices}/{shape[2]}")

    if available_voxels >= expected_voxels:
        print("数据完整! 可以直接创建有效的 NIfTI 文件")
    elif complete_slices >= 64:
        print(f"有 {complete_slices} 个完整切片，足够均匀采样 64 层提取特征")
        # 用零填充缺失数据，创建完整的 NIfTI
        padded = np.zeros(expected_voxels, dtype=dtype)
        actual = np.frombuffer(data[vox_offset:], dtype=dtype)
        padded[:len(actual)] = actual
        volume = padded.reshape(shape)

        # 保存为有效的 NIfTI 文件
        affine = np.eye(4)
        # 从 header 获取原始 affine
        orig_img = nib.Nifti1Image(np.zeros(shape, dtype=dtype), np.eye(4), header)
        affine = orig_img.affine
        new_img = nib.Nifti1Image(volume, affine, header)
        nib.save(new_img, str(RECOVERED))

        # 验证
        check = nib.load(str(RECOVERED))
        check_vol = np.asarray(check.dataobj)
        print(f"恢复文件验证: shape={check.shape}, finite={np.isfinite(check_vol).all()}")
        print(f"恢复文件大小: {RECOVERED.stat().st_size} bytes")

        # 替换原文件
        backup = TRUNCATED.with_name("amos_5964_truncated_backup.nii.gz")
        if not backup.exists():
            import shutil
            shutil.copy2(TRUNCATED, backup)
            print(f"原文件备份到: {backup}")
        tmp = TRUNCATED.with_name("amos_5964.recovered.tmp.nii.gz")
        import shutil
        shutil.copy2(RECOVERED, tmp)
        tmp.replace(TRUNCATED)
        print("已用恢复文件替换截断文件!")
        return True
    else:
        print(f"完整切片不足 ({complete_slices} < 64)，无法恢复")
        return False

if __name__ == "__main__":
    main()