"""
简单图像缩放工具。
Image utilities for simple resizing.

提供将 640x480 图像下采样到 320x240 的辅助函数（保持宽高比），
并针对深度图与 RGB 图分别采用合理的插值默认值。
Provides a helper to downscale 640x480 images to 320x240 (aspect-ratio preserved),
with sensible interpolation defaults for depth vs. RGB images.
"""

from typing import Tuple, Dict
import numpy as np
import cv2


def downscale_640x480_to_320x240(img: np.ndarray, is_depth: bool = True) -> np.ndarray:
    """
    将 640x480 图像按等比直接缩小为 320x240。 / Directly downscale a 640x480 image to 320x240 preserving aspect ratio.

    这是 downscale_to_target_size 的便捷包装函数。
    Convenience wrapper around downscale_to_target_size.
    """
    return downscale_to_target_size(img, target_height=240, target_width=320, is_depth=is_depth)


def scale_camera_intrinsics(fx: float, fy: float, ppx: float, ppy: float, 
                           scale_x: float = 0.5, scale_y: float = 0.5) -> Dict[str, float]:
    """
    根据图像缩放比例调整相机内参。 / Adjust camera intrinsics according to the image scaling factors.

    当图像从 640x480 下采样到 320x240 时，相机内参也需要相应调整：
    - 焦距 (fx, fy) 需要按比例缩放
    - 主点坐标 (ppx, ppy) 需要按比例缩放
    When an image is downsampled from 640x480 to 320x240, the camera intrinsics
    must be adjusted accordingly:
    - The focal lengths (fx, fy) are scaled by the same factors.
    - The principal-point coordinates (ppx, ppy) are scaled by the same factors.

    参数 / Args:
        fx: 原始 x 轴焦距 / Original focal length along the x axis.
        fy: 原始 y 轴焦距 / Original focal length along the y axis.
        ppx: 原始 x 轴主点坐标 / Original principal-point coordinate along the x axis.
        ppy: 原始 y 轴主点坐标 / Original principal-point coordinate along the y axis.
        scale_x: x 轴缩放比例（默认 0.5 对应 640->320） / Scale factor along the x axis (default 0.5 maps 640->320).
        scale_y: y 轴缩放比例（默认 0.5 对应 480->240） / Scale factor along the y axis (default 0.5 maps 480->240).

    返回 / Returns:
        包含缩放后内参的字典：{'fx': float, 'fy': float, 'ppx': float, 'ppy': float}
        A dict holding the scaled intrinsics: {'fx': float, 'fy': float, 'ppx': float, 'ppy': float}.
    """
    return {
        'fx': fx * scale_x,
        'fy': fy * scale_y,
        'ppx': ppx * scale_x,
        'ppy': ppy * scale_y
    }


def scale_camera_intrinsics_to_224x168(fx: float, fy: float, ppx: float, ppy: float) -> Dict[str, float]:
    """
    将 640x480 的相机内参缩放到 224x168。 / Scale 640x480 camera intrinsics to 224x168.

    缩放比例：
    - x 轴: 224/640 = 0.35
    - y 轴: 168/480 = 0.35
    Scale factors:
    - x axis: 224/640 = 0.35
    - y axis: 168/480 = 0.35

    参数 / Args:
        fx: 原始 x 轴焦距 / Original focal length along the x axis.
        fy: 原始 y 轴焦距 / Original focal length along the y axis.
        ppx: 原始 x 轴主点坐标 / Original principal-point coordinate along the x axis.
        ppy: 原始 y 轴主点坐标 / Original principal-point coordinate along the y axis.

    返回 / Returns:
        包含缩放后内参的字典：{'fx': float, 'fy': float, 'ppx': float, 'ppy': float}
        A dict holding the scaled intrinsics: {'fx': float, 'fy': float, 'ppx': float, 'ppy': float}.
    """
    scale_x = 224.0 / 640.0  # 缩放比例 0.35 / scale factor 0.35
    scale_y = 168.0 / 480.0  # 缩放比例 0.35 / scale factor 0.35
    
    return scale_camera_intrinsics(fx, fy, ppx, ppy, scale_x, scale_y)


def downscale_to_target_size(img: np.ndarray, target_height: int, target_width: int, is_depth: bool = True) -> np.ndarray:
    """
    将输入图像缩放到指定的目标尺寸。 / Resize the input image to the given target size.

    支持的输入形状：
      - (H, W)
      - (H, W, C)
      - (C, H, W)  会在内部转换为 (H, W, C) 进行 resize，之后再转换回 (C, H, W)
    Supported input shapes:
      - (H, W)
      - (H, W, C)
      - (C, H, W)  internally transposed to (H, W, C) for the resize, then transposed back to (C, H, W)

    参数 / Args:
        img: numpy.ndarray 图像/深度图 / Image or depth image as a numpy.ndarray.
        target_height: 目标高度 / Target height.
        target_width: 目标宽度 / Target width.
        is_depth: 若为 True，使用最近邻插值以避免深度值被平滑；若为 False，使用面积插值以获得更好的下采样质量。
                  If True, use nearest-neighbor interpolation to avoid smoothing depth values;
                  if False, use area interpolation for better downsampling quality.

    返回 / Returns:
        缩放后的图像，尺寸为 target_height × target_width，保持与输入相同的数据类型；
        若输入为 (C, H, W)，则返回同样的通道优先格式。
        The resized image of size target_height × target_width, keeping the same dtype as the input;
        if the input is (C, H, W), the result is returned in the same channel-first format.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy.ndarray")

    # 记录输入布局 / Record the input layout.
    channel_first = False
    if img.ndim == 3 and img.shape[0] <= 4 and img.shape[1] != target_height and img.shape[2] != target_width:
        # 可能是 (C, H, W)，但需要确认它尚未处于目标尺寸
        # Likely (C, H, W), but verify it is not already at the target size.
        if img.shape[0] <= 4 and img.shape[1] > img.shape[0] and img.shape[2] > img.shape[0]:
            channel_first = True
            img_work = np.transpose(img, (1, 2, 0))  # 转为 (H, W, C) / convert to (H, W, C)
        else:
            img_work = img
    else:
        img_work = img

    if img_work.ndim == 2:
        h, w = img_work.shape
        c = None
    elif img_work.ndim == 3:
        h, w, c = img_work.shape
    else:
        raise ValueError("Unsupported image ndim; expected 2D or 3D array")

    # 如果已经是目标尺寸，直接返回 / If it is already at the target size, return as is.
    if h == target_height and w == target_width:
        return img

    # OpenCV 的 dsize 顺序是 (width, height) / OpenCV's dsize is ordered as (width, height).
    dsize: Tuple[int, int] = (target_width, target_height)

    interpolation = cv2.INTER_NEAREST if is_depth else cv2.INTER_AREA

    resized = cv2.resize(img_work, dsize=dsize, interpolation=interpolation)

    if channel_first and resized.ndim == 3:
        resized = np.transpose(resized, (2, 0, 1))  # 转回 (C, H, W) / convert back to (C, H, W)

    # 保持 dtype 与输入一致 / Keep the dtype consistent with the input.
    if resized.dtype != img.dtype:
        resized = resized.astype(img.dtype, copy=False)

    return resized


def downscale_640x480_to_224x168(img: np.ndarray, is_depth: bool = True) -> np.ndarray:
    """
    将 640x480 图像缩小为 224x168（保持 4:3 宽高比）。 / Downscale a 640x480 image to 224x168 (keeping the 4:3 aspect ratio).

    这是 downscale_to_target_size 的便捷包装函数。
    Convenience wrapper around downscale_to_target_size.
    """
    return downscale_to_target_size(img, target_height=168, target_width=224, is_depth=is_depth)


def downscale_640x480_to_320x240(img: np.ndarray, is_depth: bool = True) -> np.ndarray:
    """
    将 640x480 图像按等比直接缩小为 320x240。 / Directly downscale a 640x480 image to 320x240 preserving aspect ratio.

    这是 downscale_to_target_size 的便捷包装函数。
    Convenience wrapper around downscale_to_target_size.
    """
    return downscale_to_target_size(img, target_height=240, target_width=320, is_depth=is_depth)


__all__ = [
    "downscale_to_target_size",
    "downscale_640x480_to_320x240", 
    "downscale_640x480_to_224x168",
    "scale_camera_intrinsics",
    "scale_camera_intrinsics_to_224x168",
]
