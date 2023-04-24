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

"""
TensorRF implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type
import time

import numpy as np
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.encodings import (
    NeRFEncoding,
    TensorCPEncoding,
    TensorVMEncoding,
    TensorVMSplitEncoding,
    TriplaneEncoding,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.tensorf_field import TensoRFField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler, VolumetricSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc

import nerfacc
from nerfacc import ContractionType


@dataclass
class TensoRFModelConfig(ModelConfig):
    """TensoRF model config"""

    _target: Type = field(default_factory=lambda: TensoRFModel)
    """target class to instantiate"""
    init_resolution: int = 128
    """initial render resolution"""
    final_resolution: int = 300
    """final render resolution"""
    upsampling_iters: Tuple[int, ...] = (2000, 3000, 4000, 5500, 7000)
    """specifies a list of iteration step numbers to perform upsampling"""
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss": 1.0})
    """Loss specific weights."""
    num_samples: int = 50
    """Number of samples in field evaluation"""
    num_uniform_samples: int = 200
    """Number of samples in density evaluation"""
    num_den_components: int = 16
    """Number of components in density encoding"""
    num_color_components: int = 48
    """Number of components in color encoding"""
    appearance_dim: int = 27
    """Number of channels for color encoding"""
    tensorf_encoding: Literal["triplane", "vm", "cp"] = "vm"
    
    yzf_mode: bool = False
    """yzf mode where use uniform sampler"""
    yzf_mode2: bool = False
    """yzf mode where use occupancy sampler"""
    render_step_size: float = 0.01
    """Minimum step size for rendering, only used in yzf_mode2"""

    enable_single_jitter: bool = True
    """enable single jitter"""
    density_activation: str = 'softplus',
    """density activation"""

    shrinking: bool = False
    """enable shrinking from occupancy grid"""
    shrinking_iters: Tuple[int, ...] = (2000, 4000)
    """specifies a list of iteration step numbers to perform shrinking"""

    filtering: bool = False
    """enable filtering"""
    filtering_iters: Tuple[int, ...] = (0, 2000, 4000)
    """specifies a list of iteration step numbers to perform filtering"""



class TensoRFModel(Model):
    """TensoRF Model

    Args:
        config: TensoRF configuration to instantiate model
    """

    config: TensoRFModelConfig

    def __init__(
        self,
        config: TensoRFModelConfig,
        **kwargs,
    ) -> None:
        self.init_resolution = config.init_resolution
        self.upsampling_iters = config.upsampling_iters
        self.num_den_components = config.num_den_components
        self.num_color_components = config.num_color_components
        self.appearance_dim = config.appearance_dim
        self.upsampling_steps = (
            np.round(
                np.exp(
                    np.linspace(
                        np.log(config.init_resolution),
                        np.log(config.final_resolution),
                        len(config.upsampling_iters) + 1,
                    )
                )
            )
            .astype("int")
            .tolist()[1:]
        )
        self.yzf_mode = config.yzf_mode
        self.yzf_mode2 = config.yzf_mode2
        self.shrinking = config.shrinking
        self.shrinking_iters = config.shrinking_iters
        self.filtering = config.filtering
        self.filtering_iters = config.filtering_iters

        assert self.yzf_mode2 or not self.shrinking, "Must enable occupancy_grid to enable shrinking"
        assert not self.shrinking or set(self.shrinking_iters) <= set(self.upsampling_iters), \
            "Shrinking iterations must be a subset of upsampling iterations"
        assert self.shrinking or not self.filtering, "Must enable shrinking to enable filtering"
        nonzero_filtering_iters = set(self.filtering_iters)
        nonzero_filtering_iters.discard(0)
        assert not self.filtering or nonzero_filtering_iters <= set(self.shrinking_iters), \
            "Filtering iteration must be a subset of shrinking iterations"
        super().__init__(config=config, **kwargs)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        # the callback that we want to run every X iterations after the training iteration
        def reinitialize_optimizer(
            self, training_callback_attributes: TrainingCallbackAttributes, step: int  # pylint: disable=unused-argument
        ):
            index = self.upsampling_iters.index(step)
            resolution = self.upsampling_steps[index]

            # upsample the position and direction grids
            # self.field.density_encoding.upsample_grid(resolution)
            # self.field.color_encoding.upsample_grid(resolution)
            self.field.upsample_grid(resolution)

            # reinitialize the encodings optimizer
            optimizers_config = training_callback_attributes.optimizers.config
            enc = training_callback_attributes.pipeline.get_param_groups()["encodings"]
            lr_init = optimizers_config["encodings"]["optimizer"].lr

            training_callback_attributes.optimizers.optimizers["encodings"] = optimizers_config["encodings"][
                "optimizer"
            ].setup(params=enc)
            if optimizers_config["encodings"]["scheduler"]:
                training_callback_attributes.optimizers.schedulers["encodings"] = (
                    optimizers_config["encodings"]["scheduler"]
                    .setup()
                    .get_scheduler(
                        optimizer=training_callback_attributes.optimizers.optimizers["encodings"], lr_init=lr_init
                    )
                )

        def update_occupancy_grid(step: int):
            # TODO: needs to get access to the sampler, on how the step size is determinated at each x. See
            # https://github.com/KAIR-BAIR/nerfacc/blob/127223b11401125a9fce5ce269bb0546ee4de6e8/examples/train_ngp_nerf.py#L190-L213
            self.occupancy_grid.every_n_step(
                step=step,
                occ_eval_fn=lambda x: self.field.get_opacity(x, self.config.render_step_size),
            )

        def shrink_tensorf_grids(self, step: int):
            print('========> shrinking grids ...')
            cur_reso = self.field.color_encoding.resolution
            aabb = self.field.aabb

            xyzs = torch.stack(torch.meshgrid(
                torch.linspace(0, 1, cur_reso[0].item()),
                torch.linspace(0, 1, cur_reso[1].item()),
                torch.linspace(0, 1, cur_reso[2].item()),
                indexing = 'ij'
            ), -1).to(aabb.device)

            xyzs = xyzs * (aabb[1] - aabb[0]) + aabb[0]

            step_size = ((aabb[1] - aabb[0]) / cur_reso).mean()*0.5/25
            alpha = self.field.get_opacity(xyzs, step_size).squeeze(-1)

            xyzs = xyzs.transpose(0,2).contiguous()
            alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None]

            alpha = 1 - torch.exp(-alpha)

            ks = 3
            alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view((cur_reso[2], cur_reso[1], cur_reso[0]))
            alpha[alpha>=0.005] = 1
            alpha[alpha<0.005] = 0

            valid_xyzs = xyzs[alpha>0.5]
            new_aabb = torch.cat([valid_xyzs.amin(0), valid_xyzs.amax(0)]).view((2,3))

            new_aabb = self.field.shrink_grid(new_aabb)
            self.sampler.scene_aabb = new_aabb.flatten()

            print(f"  New aabb: {self.sampler.scene_aabb}")

        def filter_training_rays(self, training_callback_attributes: TrainingCallbackAttributes, step: int):

            print('========> filtering rays ...')
            tt = time.time()
            assert training_callback_attributes.pipeline.datamanager.train_image_dataloader.cache_all_images, \
                NotImplementedError("Current ray filtering only works with cache_all_images=True")

            scene_aabb = self.field.aabb
            cached_data = training_callback_attributes.pipeline.datamanager.train_image_dataloader.cached_collated_batch
            ray_generator = training_callback_attributes.pipeline.datamanager.train_ray_generator
            training_callback_attributes.pipeline.datamanager.train_pixel_sampler.clear_mask_cache()

            all_ray_indices = torch.stack(torch.meshgrid(
                torch.arange(cached_data['image'].shape[0]),
                torch.arange(cached_data['image'].shape[1]),
                torch.arange(cached_data['image'].shape[2]),
                indexing = 'ij'
            ), dim=-1) # [100,800,800,3]
            all_ray_indices[...,0] = cached_data['image_idx'][..., None, None]

            all_ray_indices = all_ray_indices.view((-1,3))

            N = cached_data['image'].shape[0] * cached_data['image'].shape[1] * cached_data['image'].shape[2]

            mask_filtered = []
            idx_chunks = torch.split(torch.arange(N), 10240*5)
            for idx_chunk in idx_chunks:
                all_rays = ray_generator(all_ray_indices[idx_chunk])
                rays_o = all_rays.origins
                rays_d = all_rays.directions

                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (scene_aabb[1] - rays_o) / vec
                rate_b = (scene_aabb[0] - rays_o) / vec

                t_min = torch.minimum(rate_a, rate_b).amax(-1)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)
                mask_inbbox = t_max > t_min

                mask_filtered.append(mask_inbbox.cpu())

            mask_filtered = torch.cat(mask_filtered).view(cached_data['image'].shape[:-1] + (1,))

            print(f'  Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')
            
            cached_data['mask'] = mask_filtered


        callbacks = []

        if self.shrinking:
            # shrinking grid size should be done before the upsampling and optimizer reinitialization
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    iters=self.shrinking_iters,
                    func=shrink_tensorf_grids,
                    args=[self,],
                )
            )

        if self.filtering:
            callbacks.append( # after the shrinking
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    iters=self.filtering_iters,
                    func=filter_training_rays,
                    args=[self,training_callback_attributes],
                )
            )

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=self.upsampling_iters,
                func=reinitialize_optimizer,
                args=[self, training_callback_attributes],
            )
        )

        if self.yzf_mode2:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=update_occupancy_grid,
                )
            )
        return callbacks

    def update_to_step(self, step: int, state_dict) -> None:
        if step < self.upsampling_iters[0]:
            return

        new_iters = list(self.upsampling_iters) + [step + 1]
        new_iters.sort()

        index = new_iters.index(step + 1)
        new_grid_resolution = self.upsampling_steps[index - 1]

        if '_model.field.aabb' in state_dict:
            self.field.aabb[:] = state_dict['_model.field.aabb']
            self.sampler.scene_aabb[:] = state_dict['_model.field.aabb'].flatten()

        self.field.upsample_grid(new_grid_resolution)  # type: ignore

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # setting up fields
        if self.config.tensorf_encoding == "vm":
            density_encoding = TensorVMSplitEncoding(
                resolution=self.init_resolution,
                num_components=self.num_den_components,
            )
            color_encoding = TensorVMSplitEncoding(
                resolution=self.init_resolution,
                num_components=self.num_color_components,
            )
        elif self.config.tensorf_encoding == "cp":
            density_encoding = TensorCPEncoding(
                resolution=self.init_resolution,
                num_components=self.num_den_components,
            )
            color_encoding = TensorCPEncoding(
                resolution=self.init_resolution,
                num_components=self.num_color_components,
            )
        elif self.config.tensorf_encoding == "triplane":
            density_encoding = TriplaneEncoding(
                resolution=self.init_resolution,
                num_components=self.num_den_components,
            )
            color_encoding = TriplaneEncoding(
                resolution=self.init_resolution,
                num_components=self.num_color_components,
            )
        else:
            raise ValueError(f"Encoding {self.config.tensorf_encoding} not supported")

        feature_encoding = NeRFEncoding(in_dim=self.appearance_dim, num_frequencies=2, min_freq_exp=0, max_freq_exp=2)
        direction_encoding = NeRFEncoding(in_dim=3, num_frequencies=2, min_freq_exp=0, max_freq_exp=2)

        self.field = TensoRFField(
            self.scene_box.aabb,
            feature_encoding=feature_encoding,
            direction_encoding=direction_encoding,
            density_encoding=density_encoding,
            color_encoding=color_encoding,
            appearance_dim=self.appearance_dim,
            head_mlp_num_layers=2,
            head_mlp_layer_width=128,
            use_sh=False,
            density_activation=self.config.density_activation,
        )

        # samplers
        if self.yzf_mode:
            self.sampler = UniformSampler(num_samples=self.config.num_uniform_samples, single_jitter=True)
        elif self.yzf_mode2:
            flat_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

            # Occupancy Grid
            self.occupancy_grid = nerfacc.OccupancyGrid(
                roi_aabb=flat_aabb,
                resolution=128,
                contraction_type=ContractionType.AABB
            )

            # TODO: need to support other contraction types
            vol_sampler_aabb = self.scene_box.aabb
            self.sampler = VolumetricSampler(
                scene_aabb=vol_sampler_aabb,
                occupancy_grid=self.occupancy_grid,
                density_fn=self.field.density_fn,
            )
        else:
            self.sampler_uniform = UniformSampler(num_samples=self.config.num_uniform_samples, single_jitter=True)
            self.sampler_pdf = PDFSampler(num_samples=self.config.num_samples, single_jitter=self.config.enable_single_jitter, include_original=False)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected" if self.yzf_mode2 else "median")

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # colliders
        if self.config.enable_collider:
            self.collider = AABBBoxCollider(scene_box=self.scene_box)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}

        param_groups["fields"] = (
            list(self.field.mlp_head.parameters())
            + list(self.field.B.parameters())
            + list(self.field.field_output_rgb.parameters())
        )
        param_groups["encodings"] = list(self.field.color_encoding.parameters()) + list(
            self.field.density_encoding.parameters()
        )

        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        if self.yzf_mode:
            return self.get_outputs_mode1(ray_bundle)
        elif self.yzf_mode2:
            return self.get_outputs_mode2(ray_bundle)
        else:
            return self.get_outputs_old(ray_bundle)

    def get_outputs_mode1(self, ray_bundle: RayBundle):
        ray_samples_uniform = self.sampler(ray_bundle)
        dens, _ = self.field.get_density(ray_samples_uniform)
        weights = ray_samples_uniform.get_weights(dens)
        coarse_accumulation = self.renderer_accumulation(weights)
        acc_mask = torch.where(coarse_accumulation < 0.0001, False, True).reshape(-1)

        field_outputs_fine = self.field.forward(
            ray_samples_uniform, mask=acc_mask, bg_color=colors.WHITE.to(weights.device)
        )

        weights_fine = weights

        accumulation = self.renderer_accumulation(weights_fine)
        depth = self.renderer_depth(weights_fine, ray_samples_uniform)

        rgb = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )

        rgb = torch.where(accumulation < 0, colors.WHITE.to(rgb.device), rgb)
        accumulation = torch.clamp(accumulation, min=0)

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth}
        return outputs

    def get_outputs_mode2(self, ray_bundle: RayBundle):
        num_rays = len(ray_bundle)
        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=2,
                far_plane=6,
                render_step_size=self.config.render_step_size,
                cone_angle=0)
        
        field_outputs = self.field(ray_samples)

        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            packed_info=packed_info,
            sigmas=field_outputs[FieldHeadNames.DENSITY],
            t_starts=ray_samples.frustums.starts,
            t_ends=ray_samples.frustums.ends,
        )

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        depth = self.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
        )
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth}
        return outputs

    def get_outputs_old(self, ray_bundle: RayBundle):
        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        dens, _ = self.field.get_density(ray_samples_uniform)
        weights = ray_samples_uniform.get_weights(dens)
        coarse_accumulation = self.renderer_accumulation(weights)
        acc_mask = torch.where(coarse_accumulation < 0.0001, False, True).reshape(-1)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights)

        # fine field:
        field_outputs_fine = self.field.forward(
            ray_samples_pdf, mask=acc_mask, bg_color=colors.WHITE.to(weights.device)
        )

        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])

        accumulation = self.renderer_accumulation(weights_fine)
        depth = self.renderer_depth(weights_fine, ray_samples_pdf)

        rgb = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )

        rgb = torch.where(accumulation < 0, colors.WHITE.to(rgb.device), rgb)
        accumulation = torch.clamp(accumulation, min=0)

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth}
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb"].device
        image = batch["image"].to(device)

        rgb_loss = self.rgb_loss(image, outputs["rgb"])

        l1_loss = self.field.l1_loss() * 8e-5

        loss_dict = {"rgb_loss": rgb_loss, "l1_loss": l1_loss}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        assert self.config.collider_params is not None
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = torch.Tensor([0.5,])

        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
            "lpips": float(lpips.item()),
        }
        images_dict = {"img": combined_rgb, "accumulation": acc, "depth": depth}
        return metrics_dict, images_dict
