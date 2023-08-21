"""Data parser for blender dataset"""
from __future__ import annotations

import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import imageio
import numpy as np
import torch
import cv2

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json

def load_opencv_calib(extrin_path, intrin_path):
    cameras = {}

    fse = cv2.FileStorage()
    fse.open(str(extrin_path), cv2.FileStorage_READ)

    fsi = cv2.FileStorage()
    fsi.open(str(intrin_path), cv2.FileStorage_READ)

    names = [
        fse.getNode("names").at(c).string() for c in range(fse.getNode("names").size())
    ]

    for camera in names:
        rot = fse.getNode(f"R_{camera}").mat()
        R = fse.getNode(f"Rot_{camera}").mat()
        T = fse.getNode(f"T_{camera}").mat()
        R_pred = cv2.Rodrigues(rot)[0]
        assert np.all(np.isclose(R_pred, R))
        K = fsi.getNode(f"K_{camera}").mat()
        cameras[camera] = {
            "Rt": np.concatenate([R, T], axis=1).astype(np.float32),
            "K": K.astype(np.float32),
        }
    return cameras

@dataclass
class ZJUDataParserConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: ZJUData)
    """target class to instantiate"""
    data: Path = Path("data/386")
    """Directory specifying location of data."""
    frame_idx: int = 0
    """Frame idx to load."""
    alpha_color: str = "white"
    """alpha color of background"""
    scale_factor: float = 1.0


@dataclass
class ZJUData(DataParser):
    """Blender Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: ZJUDataParserConfig
    TRAIN_CAMS = ["01", "02", "03", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21",]
    TEST_CAMS = ["04", "22", "23"]

    def __init__(self, config: ZJUDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color
        self.frame_idx = config.frame_idx

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        # prefix_dict = {
        #     "train": "0",
        #     "val": "1",
        #     "test": "1",
        # }

        # prefix = prefix_dict[split]
        if split == "train":
            cam_list = self.TRAIN_CAMS
        elif split == "test" or split == "val":
            cam_list = self.TEST_CAMS

        raw_rgb_path = os.listdir(self.data / "masked-images")

        camera_perp = load_opencv_calib(self.data / "extri.yml", self.data / "intri.yml")

        image_filenames = []
        poses = []
        intris = []
        for p in raw_rgb_path:
            if p in cam_list:
                image_filenames.append(self.data / "masked-images"/ p / f"{self.frame_idx:06d}.jpg")
                poses.append(camera_perp[p]["Rt"])
                intris.append(camera_perp[p]["K"])

        print(image_filenames)
        print(poses)
        print(intris)

        # meta = load_from_json(self.data / f"transforms_{split}.json")
        # image_filenames = []
        # poses = []
        # for frame in meta["frames"]:
        #     fname = self.data / Path(frame["file_path"].replace("./", "") + ".png")
        #     image_filenames.append(fname)
        #     poses.append(np.array(frame["transform_matrix"]))
        poses = np.array(poses).astype(np.float32)
        intris = np.array(intris).astype(np.float32)

        # img_0 = imageio.imread(image_filenames[0])
        # image_height, image_width = img_0.shape[:2]
        # camera_angle_x = float(meta["camera_angle_x"])
        # focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        # cx = image_width / 2.0
        # cy = image_height / 2.0
        camera_to_world = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(len(poses), 1, 1)
        camera_to_world[:, :3, :4] = torch.from_numpy(poses[:, :3, :4])  # camera to world transform
        camera_to_world = torch.linalg.inv(camera_to_world)[:, :3]
        camera_to_world[:,:3, 1:3] *= -1
        Ks = torch.from_numpy(intris)

        # # in x,y,z order
        # camera_to_world[..., 3] *= self.scale_factor
        # scene_box = SceneBox(aabb=torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32))
        bbox_file = self.data / "bbox.txt"
        bbox = np.loadtxt(bbox_file)
        scene_box = SceneBox(aabb=torch.tensor(bbox, dtype=torch.float32)[:-1].reshape(2,3))


        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=Ks[:,0,0],
            fy=Ks[:,1,1],
            cx=Ks[:,0,2],
            cy=Ks[:,1,2],
            height=1024,
            width=1024,
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
