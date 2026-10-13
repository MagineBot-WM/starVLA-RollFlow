"""Compatibility registry for the already-trained LIBERO H64 checkpoint.

The checkpoint records ``libero_all_rollflow_h64`` as its training mixture,
while the current LIBERO example registry only contains the non-H64 alias.
PolicyNormProcessor needs the mixture to recover the training DataConfig when
it loads normalization statistics. The H64 model's per-key action transform is
the same RollFlow transform; the horizon is stored in the model config.
"""

DATASET_NAMED_MIXTURES = {
    "libero_all_rollflow_h64": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka_rollflow"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka_rollflow"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka_rollflow"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka_rollflow"),
    ],
}
