# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Data parser for nerfstudio datasets. """

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Optional, Type

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json

CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600


@dataclass
class Real360DataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: Real360Data)
    """target class to instantiate"""
    data: Path = Path("data/nerfstudio/poster")
    """Directory or explicit json file path specifying location of data."""
    # scale_factor: float = 1.0
    # """How much to scale the camera origins by."""
    downscale_factor: int = 4
    """How much to downscale images. In mipnerf dataparser, the default downscale factor = 4.0."""
    # scene_scale: float = 1.0
    # """How much to scale the region of interest by."""
    # orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    # """The method to use for orientation."""
    # center_method: Literal["poses", "focus", "none"] = "poses"
    # """The method to use to center the poses."""
    # auto_scale_poses: bool = True
    # """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_factor: int = 8
    """How many numbers of input images per test images."""


@dataclass
class Real360Data(DataParser):
    """Nerfstudio DatasetParser"""

    config: Real360DataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):
        # pylint: disable=too-many-statements

        meta = load_from_json(self.config.data / "transforms.json")
        data_dir = self.config.data

        image_filenames = []
        poses = []
        num_skipped_image_filenames = 0

        fx = float(meta["fl_x"])
        fy = float(meta["fl_y"])
        cx = float(meta["cx"])
        cy = float(meta["cy"])
        height = int(meta["h"])
        width = int(meta["w"])

        for frame in meta["frames"]:
            filepath = PurePath(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)
            if not fname.exists():
                num_skipped_image_filenames += 1
                continue

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))

        if num_skipped_image_filenames >= 0:
            CONSOLE.log(f"Skipping {num_skipped_image_filenames} files in dataset split {split}.")
        assert (
            len(image_filenames) != 0
        ), """
        No image files found. 
        You should check the file_paths in the transforms.json file to make sure they are correct.
        """

        # divide train and eval images based on given splict factor
        num_images = len(image_filenames)
        all_indices = np.arange(num_images)
        if split == "train":
            indices = all_indices[all_indices % self.config.train_split_factor != 0]
        elif split in ["val", "test"]:
            indices = all_indices[all_indices % self.config.train_split_factor == 0]
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        # change orientation
        # orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        # poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
        #     poses,
        #     method=orientation_method,
        #     center_method=self.config.center_method,
        # )

        # Scale poses
        # scale_factor = 1.0
        # if self.config.auto_scale_poses:
        #     scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        # scale_factor *= self.config.scale_factor

        # poses[:, :3, 3] *= scale_factor

        # in x,y,z order
        # assumes that the scene is centered at the origin
        # aabb_scale = self.config.scene_scale
        # scene_box = SceneBox(
        #     aabb=torch.tensor(
        #         [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
        #     )
        # )
        camera_locs = poses[:, :3, 3]
        aabb = torch.cat([camera_locs.min(dim=0).values, camera_locs.max(dim=0).values]).tolist()

        print(f"Auto aabb: {aabb}")
        scene_box = SceneBox(aabb=torch.tensor(aabb, dtype=torch.float32))

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        poses = poses[indices]


        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
        )
        return dataparser_outputs

    def _get_fname(self, filepath: PurePath, data_dir: PurePath, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        downsample_folder_prefix: prefix of the newly generated downsampled images
        """

        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(data_dir / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
                        break
                    if not (data_dir / f"{downsample_folder_prefix}{2**(df+1)}" / filepath.name).exists():
                        break
                    df += 1

                self.downscale_factor = 2**df
                CONSOLE.log(f"Auto image downscale factor of {self.downscale_factor}")
            else:
                self.downscale_factor = self.config.downscale_factor

        if self.downscale_factor > 1:
            return data_dir / f"{downsample_folder_prefix}{self.downscale_factor}" / filepath.name
        return data_dir / filepath
