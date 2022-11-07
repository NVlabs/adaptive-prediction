import pickle as pkl
from collections import defaultdict
from pathlib import Path
from typing import Final

import numpy as np
from nuscenes.eval.prediction import splits
from tqdm import trange
from trajdata import AgentType
from trajdata.caching import EnvCache
from trajdata.data_structures import SceneMetadata
from trajdata.dataset_specific.nusc import NuscDataset, nusc_utils

###########################################################################
# Change these to match your computing environment!
TRAJDATA_CACHE_DIR: Final[str] = "/home/bivanovic/.unified_data_cache"
NUSC_RAW_DATA_DIR: Final[str] = "/home/bivanovic/datasets/nuScenes"
###########################################################################

# Load training and evaluation environments and scenes
attention_radius = defaultdict(
    lambda: 20.0
)  # Default range is 20m unless otherwise specified.
attention_radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] = 10.0
attention_radius[(AgentType.PEDESTRIAN, AgentType.VEHICLE)] = 20.0
attention_radius[(AgentType.VEHICLE, AgentType.PEDESTRIAN)] = 20.0
attention_radius[(AgentType.VEHICLE, AgentType.VEHICLE)] = 30.0

map_params = {"px_per_m": 2, "map_size_px": 100, "offset_frac_xy": (-0.75, 0.0)}


nusc_dataset = NuscDataset(
    "nusc_trainval", NUSC_RAW_DATA_DIR, parallelizable=False, has_maps=True
)
nusc_dataset.load_dataset_obj()


for split in ["train", "train_val", "val"]:
    prediction_challenge_tokens = set(
        splits.get_prediction_challenge_split(split, dataroot=NUSC_RAW_DATA_DIR)
    )

    within_challenge_split = list()

    for idx in trange(len(nusc_dataset.dataset_obj.scene)):
        scene_info = SceneMetadata(None, None, None, idx)
        scene = nusc_dataset.get_scene(scene_info)

        for frame_idx, frame in enumerate(
            nusc_utils.frame_iterator(nusc_dataset.dataset_obj, scene)
        ):
            for agent_info in nusc_utils.agent_iterator(
                nusc_dataset.dataset_obj, frame
            ):
                instance_token: str = agent_info["instance_token"]
                sample_token: str = agent_info["sample_token"]

                if f"{instance_token}_{sample_token}" in prediction_challenge_tokens:
                    scene_info_path = EnvCache.scene_metadata_path(
                        Path(""), nusc_dataset.name, scene.name, scene.dt
                    )

                    within_challenge_split.append(
                        (
                            str(scene_info_path),
                            1,
                            [
                                (
                                    instance_token,
                                    np.array([frame_idx, frame_idx], dtype=int),
                                )
                            ],
                        )
                    )

    print(split, len(within_challenge_split))
    with open(f"predchal_{split}_index.pkl", "wb") as f:
        pkl.dump(within_challenge_split, f)
