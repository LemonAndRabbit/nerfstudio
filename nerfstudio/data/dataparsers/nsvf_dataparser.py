"""Data parser for blender dataset"""
from __future__ import annotations

import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import imageio
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json

def _parse_pose(cname: str):
    lines = [[float(w) for w in line.strip().split()] for line in open(cname)]
    if len(lines[0]) == 2:
        lines = lines[1:]
    if len(lines[-1]) == 2:
        lines = lines[:-1]
    return np.array(lines).astype(np.float32)


@dataclass
class NSVFDataParserConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: NSVF)
    """target class to instantiate"""
    data: Path = Path("data/blender/lego")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""


@dataclass
class NSVF(DataParser):
    """Blender Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: NSVFDataParserConfig

    def __init__(self, config: NSVFDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        prefix_dict = {
            "train": "0",
            "val": "1",
            "test": "2",
        }

        prefix = prefix_dict[split]

        raw_rgb_path = os.listdir(self.data / "rgb")

        image_filenames = []
        poses = []
        for p in raw_rgb_path:
            if p[0] == prefix:
                image_filenames.append(self.data / "rgb" / p)
                cname = self.data / "pose" / (p[:-4] + ".txt")
                pose = _parse_pose(cname)
                poses.append(pose)


        # meta = load_from_json(self.data / f"transforms_{split}.json")
        # image_filenames = []
        # poses = []
        # for frame in meta["frames"]:
        #     fname = self.data / Path(frame["file_path"].replace("./", "") + ".png")
        #     image_filenames.append(fname)
        #     poses.append(np.array(frame["transform_matrix"]))
        poses = np.array(poses).astype(np.float32)
        poses[:,:3, 1:3] *= -1

        # img_0 = imageio.imread(image_filenames[0])
        # image_height, image_width = img_0.shape[:2]
        # camera_angle_x = float(meta["camera_angle_x"])
        # focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        # cx = image_width / 2.0
        # cy = image_height / 2.0
        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        # # in x,y,z order
        # camera_to_world[..., 3] *= self.scale_factor
        # scene_box = SceneBox(aabb=torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32))
        bbox_file = self.data / "bbox.txt"
        bbox = np.loadtxt(bbox_file)
        scene_box = SceneBox(aabb=torch.tensor(bbox, dtype=torch.float32)[:-1].reshape(2,3))


        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=875.,
            fy=875.,
            cx=400.,
            cy=400.,
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
        )

        return dataparser_outputs
