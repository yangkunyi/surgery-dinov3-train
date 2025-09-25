# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from bisect import bisect_right
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2
import torch


import zarr

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Normalize:
    return v2.Normalize(mean=mean, std=std)


class Cholec80(VisionDataset):
    """Cholec80 dataset backed by a Zarr store.

    Samples correspond to frame-aligned clips aggregated across all available
    videos within ``cholec80/rgb``. Each index returns a dictionary containing
    the mask and depth clips, the matching RGB frame clip, and the CoTracker
    outputs. CoTracker data follows the ``(T, N, 2)`` convention and is paired
    with the matching ``T`` RGB frames in clip order.
    """

    def __init__(
        self,
        split: Optional[str] = None,
        root: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        if root is None:
            raise ValueError("The 'root' argument must point to the dataset.zarr directory.")

        dataset_path = Path(root).expanduser()
        if not dataset_path.exists():
            raise FileNotFoundError(f"No Cholec80 Zarr store found at {dataset_path}")

        super().__init__(
            root=str(dataset_path),
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )

        self.split = split  # accepted for API compatibility; intentionally unused
        root_group = zarr.open(dataset_path, mode="r")
        cholec_group = self._require_group(root_group, "cholec80")
        self._rgb_group = self._require_group(cholec_group, "rgb")
        self._mask_group = self._require_group(cholec_group, "mask")
        self._depth_group = self._require_group(cholec_group, "depth")
        self._cotracker_group = self._require_group(cholec_group, "cotracker")

        self._video_names = sorted(self._rgb_group.keys())
        if not self._video_names:
            raise RuntimeError(f"No videos discovered in {dataset_path / 'cholec80/rgb'}")

        (
            self._rgb_arrays,
            self._mask_arrays,
            self._depth_arrays,
            self._tracks_arrays,
            self._visibility_arrays,
            self._frame_counts,
        ) = self._load_video_arrays()

        self._cumulative_lengths = self._build_cumulative_lengths(self._frame_counts)
        self._length = self._cumulative_lengths[-1]

        self.normalize = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                make_normalize_transform(),
            ]
        )
        self.rgb_spatial_transform = v2.Compose(
            [
                v2.Resize((256, 256), interpolation=v2.InterpolationMode.BICUBIC),
                v2.CenterCrop((256, 256)),
            ]
        )
        self.mask_spatial_transform = v2.Compose(
            [
                v2.Resize((256, 256), interpolation=v2.InterpolationMode.NEAREST),
                v2.CenterCrop((256, 256)),
            ]
        )

    def _load_video_arrays(
        self,
    ) -> Tuple[
        List["zarr.Array"],
        List["zarr.Array"],
        List["zarr.Array"],
        List["zarr.Array"],
        List["zarr.Array"],
        List[int],
    ]:
        rgb_arrays: List["zarr.Array"] = []
        mask_arrays: List["zarr.Array"] = []
        depth_arrays: List["zarr.Array"] = []
        tracks_arrays: List["zarr.Array"] = []
        visibility_arrays: List["zarr.Array"] = []
        frame_counts: List[int] = []

        for name in self._video_names:
            rgb_array = self._rgb_group[name]
            mask_array = self._mask_group[name]
            depth_array = self._depth_group[name]
            cotracker_video_group = self._require_group(self._cotracker_group, name)
            tracks_array = self._require_array(cotracker_video_group, "tracks")
            visibility_array = self._require_array(cotracker_video_group, "visibility")

            frame_count = int(rgb_array.shape[0])
            self._ensure_matching_frame_count(frame_count, mask_array, "mask", name)
            self._ensure_matching_frame_count(frame_count, depth_array, "depth", name)
            self._ensure_matching_frame_count(frame_count, tracks_array, "tracks", name)
            self._ensure_matching_frame_count(frame_count, visibility_array, "visibility", name)

            rgb_arrays.append(rgb_array)
            mask_arrays.append(mask_array)
            depth_arrays.append(depth_array)
            tracks_arrays.append(tracks_array)
            visibility_arrays.append(visibility_array)
            frame_counts.append(frame_count)

        return rgb_arrays, mask_arrays, depth_arrays, tracks_arrays, visibility_arrays, frame_counts

    @staticmethod
    def _require_group(parent: "zarr.hierarchy.Group", key: str) -> "zarr.hierarchy.Group":
        try:
            return parent[key]
        except KeyError as exc:  # pragma: no cover - fail fast in production
            raise RuntimeError(f"The provided Zarr store is missing the '{key}' group.") from exc

    @staticmethod
    def _require_array(parent: "zarr.hierarchy.Group", key: str) -> "zarr.Array":
        try:
            return parent[key]
        except KeyError as exc:  # pragma: no cover - fail fast in production
            raise RuntimeError(f"The provided Zarr store is missing the '{key}' array.") from exc

    @staticmethod
    def _ensure_matching_frame_count(
        reference_count: int,
        array: "zarr.Array",
        label: str,
        video_name: str,
    ) -> None:
        if int(array.shape[0]) != reference_count:
            raise RuntimeError(
                f"Mismatched frame count for '{label}' in video '{video_name}'. Expected {reference_count},"
                f" found {array.shape[0]}."
            )

    @staticmethod
    def _build_cumulative_lengths(frame_counts: Sequence[int]) -> List[int]:
        cumulative: List[int] = []
        running_total = 0
        for count in frame_counts:
            if count <= 0:
                raise RuntimeError("Encountered a video with no frames while indexing Cholec80.")
            running_total += count
            cumulative.append(running_total)
        if not cumulative:
            raise RuntimeError("Unable to index Cholec80 because no frames were discovered.")
        return cumulative

    def _resolve_index(self, index: int) -> Tuple[int, int]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError("Cholec80 index out of range")

        video_idx = bisect_right(self._cumulative_lengths, index)
        previous_total = 0 if video_idx == 0 else self._cumulative_lengths[video_idx - 1]
        frame_idx = index - previous_total
        return video_idx, frame_idx

    def __getitem__(self, index: int) -> Dict[str, Any]:
        video_idx, frame_idx = self._resolve_index(index)

        tracks_clip = np.asarray(self._tracks_arrays[video_idx][frame_idx], dtype=np.float32)
        clip_length = tracks_clip.shape[0]
        visibility_clip = np.asarray(self._visibility_arrays[video_idx][frame_idx], dtype=np.float32)
        if visibility_clip.shape[0] != clip_length:
            raise RuntimeError(
                f"Mismatched visibility clip length for video index {video_idx}: expected {clip_length},"
                f" found {visibility_clip.shape[0]}."
            )

        rgb_array = self._rgb_arrays[video_idx]
        rgb_clip_slice = rgb_array[frame_idx : frame_idx + clip_length]
        rgb_clip_np = np.asarray(rgb_clip_slice, dtype=np.uint8)
        if rgb_clip_np.shape[0] != clip_length:
            raise RuntimeError(
                f"Mismatched RGB clip length for video index {video_idx}: expected {clip_length},"
                f" found {rgb_clip_np.shape[0]}"
            )

        rgb_clip_images = [Image.fromarray(frame, mode="RGB") for frame in rgb_clip_np]
        resized_rgb_images = [self.rgb_spatial_transform(image) for image in rgb_clip_images]
        rgb_clip_tuple = tuple(resized_rgb_images)

        normalized_clip = torch.stack([self.normalize(image) for image in resized_rgb_images])

        transform_outputs: Optional[List[Any]] = None
        if self.transform is not None:
            transform_outputs = [self.transform(image) for image in rgb_clip_tuple]


        mask_clip = self._mask_arrays[video_idx][frame_idx : frame_idx + clip_length]
        if mask_clip.shape[0] != clip_length:
            raise RuntimeError(
                f"Mismatched mask clip length for video index {video_idx}: expected {clip_length},"
                f" found {mask_clip.shape[0]}"
            )
        mask_clip_np = []
        for mask_frame in mask_clip:
            mask_array = np.asarray(mask_frame, dtype=np.uint8).squeeze(-1)
            mask_image = Image.fromarray(mask_array, mode="L")
            resized_mask = self.mask_spatial_transform(mask_image)
            mask_clip_np.append(np.asarray(resized_mask, dtype=np.uint8)[..., None])
        mask_clip_np = np.stack(mask_clip_np, axis=0)

        depth_clip = self._depth_arrays[video_idx][frame_idx : frame_idx + clip_length]
        if depth_clip.shape[0] != clip_length:
            raise RuntimeError(
                f"Mismatched depth clip length for video index {video_idx}: expected {clip_length},"
                f" found {depth_clip.shape[0]}"
            )
        depth_clip_np = []
        for depth_frame in depth_clip:
            depth_array = np.asarray(depth_frame, dtype=np.uint8).squeeze(-1)
            depth_image = Image.fromarray(depth_array, mode="L")
            resized_depth = self.mask_spatial_transform(depth_image)
            depth_clip_np.append(np.asarray(resized_depth, dtype=np.uint8)[..., None])
        depth_clip_np = np.stack(depth_clip_np, axis=0)

        target: Dict[str, Any] = {
            "mask_clip": mask_clip_np,
            "depth_clip": depth_clip_np,
            "rgb_clip_tensor": normalized_clip,
            "tracks": tracks_clip,
            "visibility": visibility_clip,
            "clip_length": clip_length,
        }

        if transform_outputs is not None:
            target["transforms"] = tuple(transform_outputs)

        return target

    def __len__(self) -> int:
        return self._length
