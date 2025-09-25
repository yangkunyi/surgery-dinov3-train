# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def collate_data_and_cast(
    samples_list,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    n_tokens=None,
    mask_generator=None,
    random_circular_shift=False,
    local_batch_size=None,
):
    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack(
        [s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list]
    )  # [n_global_crops * B, ...]
    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])
    if "gram_teacher_crops" in samples_list[0][0]:
        collated_gram_teacher_crops = torch.stack(
            [s[0]["gram_teacher_crops"][i] for i in range(n_global_crops) for s in samples_list]
        )  # [n_global_crops, B, ...]
    else:
        collated_gram_teacher_crops = None

    if local_batch_size is not None:
        # multi-distillation case, number of masks is different because the number of samples masked
        # is different of the number of samples passed into the teacher initially
        B = n_global_crops * local_batch_size
    else:
        B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_max = probs[i + 1]
        mask = torch.BoolTensor(mask_generator(int(N * prob_max)))
        if random_circular_shift:  # apply le random circular shift to
            shift_x, shift_y = (
                random.randint(0, mask.shape[0] - 1),
                random.randint(0, mask.shape[1] - 1),
            )
            mask = torch.roll(mask, (shift_x, shift_y), (0, 1))
        masks_list.append(mask)
        upperbound += int(N * prob_max)
    for _ in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    out = {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
    if collated_gram_teacher_crops is not None:
        out["collated_gram_teacher_crops"] = collated_gram_teacher_crops.to(dtype)
    return out


def collate_cholec80_data(
    samples_list: Sequence[Dict[str, Any]],
    mask_ratio_tuple: Tuple[float, float],
    mask_probability: float,
    dtype: torch.dtype,
    n_tokens: Optional[int] = None,
    mask_generator=None,
    random_circular_shift: bool = False,
    local_batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Collate function tailored for the Cholec80 dataset.

    The function mirrors :func:`collate_data_and_cast` by first collating the
    multi-crop augmentation outputs (one entry per clip frame) and then stacking
    the clip-aware tensors (mask, depth, RGB, CoTracker metadata) so that the
    leading dimension corresponds to the batch size ``B``.
    """

    if not samples_list:
        raise ValueError("Expected a non-empty list of samples for collation.")

    clip_lengths = [sample["clip_length"] for sample in samples_list]
    clip_length = clip_lengths[0]
    if any(length != clip_length for length in clip_lengths[1:]):
        raise ValueError("All Cholec80 samples in the batch must share the same clip length.")

    frame_samples: List[Tuple[Dict[str, Any], Optional[Any]]] = []
    for sample in samples_list:
        transforms = sample.get("transforms")
        if transforms is None or len(transforms) != clip_length:
            raise ValueError("Cholec80 sample must carry a per-frame transforms tuple matching clip length.")
        for frame_transform in transforms:
            frame_samples.append((frame_transform, None))

    collated = collate_data_and_cast(
        frame_samples,
        mask_ratio_tuple=mask_ratio_tuple,
        mask_probability=mask_probability,
        dtype=dtype,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        random_circular_shift=random_circular_shift,
        local_batch_size=local_batch_size,
    )

    mask_clip = torch.from_numpy(np.stack([sample["mask_clip"] for sample in samples_list], axis=0))
    depth_clip = torch.from_numpy(np.stack([sample["depth_clip"] for sample in samples_list], axis=0))
    rgb_clip_tensor = torch.stack([sample["rgb_clip_tensor"] for sample in samples_list], dim=0)
    tracks = torch.from_numpy(np.stack([sample["tracks"] for sample in samples_list], axis=0))
    visibility = torch.from_numpy(np.stack([sample["visibility"] for sample in samples_list], axis=0))
    clip_length_tensor = torch.tensor(clip_lengths, dtype=torch.long)

    collated.update(
        {
            "mask_clip": mask_clip,
            "depth_clip": depth_clip,
            "rgb_clip_tensor": rgb_clip_tensor,
            "tracks": tracks,
            "visibility": visibility,
            "clip_length": clip_length_tensor,
        }
    )

    return collated


# def get_batch_subset(collated_data_batch, target_bs):
def get_batch_subset(collated_data_batch, divide_by):
    old_bs = collated_data_batch["collated_global_crops"].shape[0] // 2
    target_bs = (old_bs + divide_by - 1) // divide_by
    collated_global_crops = (
        collated_data_batch["collated_global_crops"].unflatten(0, (2, old_bs)).narrow(1, 0, target_bs).flatten(0, 1)
    )
    collated_local_crops = (
        collated_data_batch["collated_local_crops"].unflatten(0, (-1, old_bs)).narrow(1, 0, target_bs).flatten(0, 1)
    )

    masks_old_bs = collated_data_batch["collated_masks"].shape[0] // 2
    masks_target_bs = masks_old_bs // divide_by
    collated_masks = (
        collated_data_batch["collated_masks"]
        .unflatten(0, (2, masks_old_bs))
        .narrow(1, 0, masks_target_bs)
        .flatten(0, 1)
    )
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    while mask_indices_list.shape[0] == 0:
        _unbind = list(collated_data_batch["collated_masks"].unbind(0))
        random.shuffle(_unbind)
        _bind = torch.stack(_unbind, dim=0)
        collated_masks = _bind.unflatten(0, (2, masks_old_bs)).narrow(1, 0, masks_target_bs).flatten(0, 1)
        mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    upperbound = collated_data_batch["upperbound"]

    new_batch = {
        "collated_global_crops": collated_global_crops,
        "collated_local_crops": collated_local_crops,
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }

    if "global_batch_size" in collated_data_batch.keys():
        new_batch["global_batch_size"] = collated_data_batch["global_batch_size"] // divide_by

    return new_batch
