# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

""" Functions to create a grid/atlas of colors and features. 
    Heavily inspired by https://github.com/google-research/google-research/blob/master/snerg/snerg/baking.py
    
    Modified by EIC lab at Georgia Tech to use vanilla PyTorch instead of JAX
"""

import gc

import numpy as np
import torch


def build_3d_grid(min_xyz,
                  voxel_size,
                  grid_size,
                  worldspace_t_opengl=np.eye(4),
                  output_dtype=np.float32):
    """Builds a tensor containing a regular grid of 3D locations.

    Args:
        min_xyz: The minimum XYZ location of the grid.
        voxel_size: The side length of a voxel.
        grid_size: A numpy array containing the grid dimensions [H, W, D].
        worldspace_t_opengl: An optional 4x4 transformation matrix that maps the
        native coordinate space of the NeRF model to an OpenGL coordinate system,
        where y is down, and negative-z is pointing towards the scene.
        output_dtype: The data type of the resulting grid tensor.

    Returns:
        A [H, W, D, 3] numpy array for the XYZ coordinates of each grid cell center.
    """
    x_span = min_xyz[0] + voxel_size / 2 + np.arange(grid_size[0], dtype=output_dtype) * voxel_size
    y_span = min_xyz[1] + voxel_size / 2 + np.arange(grid_size[1], dtype=output_dtype) * voxel_size
    z_span = min_xyz[2] + voxel_size / 2 + np.arange(grid_size[2], dtype=output_dtype) * voxel_size
    xv, yv, zv = np.meshgrid(x_span, y_span, z_span, indexing='ij')
    positions_hom = np.stack([xv, yv, zv, np.ones_like(zv)], axis=-1)
    positions_hom = positions_hom.reshape((-1, 4)).dot(worldspace_t_opengl)
    return positions_hom[Ellipsis, 0:3].reshape((xv.shape[0], xv.shape[1], xv.shape[2], -1))

def render_voxel_block(mlp_model, mlp_params, block_coordinates_world,
                       voxel_size, scene_params):
    """Extracts a grid of colors, features and alpha values from a SNeRG model.

    Args:
      mlp_model: A nerf.model_utils.MLP that predicts per-sample color, density,
        and the SNeRG feature vector.
      mlp_params: A dict containing the MLP parameters for the per-sample MLP.
      block_coordinates_world: A [H, W, D, 3] numpy array of XYZ coordinates (see
        build_3d_grid).
      voxel_size: The side length of a voxel.
      scene_params: A dict for scene specific params (bbox, rotation, resolution).

    Returns:
      output_rgb_and_features: A [H, W, D, C] numpy array for the colors and
        computed features at each voxel.
      output_alpha: A [H, W, D, 1] numpy array for the alpha values at each voxel.
    """
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # retrieve scene params
    chunk_size = scene_params['chunk_size']
    channels = scene_params['_channels']
    output_dtype = torch.tensor(scene_params['dtype'])

    batch_size = chunk_size * num_devices
    actual_num_rays = int(torch.tensor(block_coordinates_world.shape[0:3]).prod())
    rounded_num_rays = batch_size * ((actual_num_rays + batch_size - 1) // batch_size)

    origins = block_coordinates_world.reshape((-1, 3)).copy()
    origins.resize((rounded_num_rays, 3))
    origins = origins.reshape((-1, num_devices, batch_size // num_devices, 3))

    rgb_and_features = np.zeros((origins.shape[0], origins.shape[1], origins.shape[2], channels), dtype=output_dtype)
    sigma = torch.zeros_like(rgb_and_features[:, :, :, 0])

    # *********** NOT REFACTORED YET *************

    # for i in range(origins.shape[0]):
    #   batch_origins = origins[i]
    #   batch_origins = batch_origins.reshape(num_devices, 1, -1, 3)

      
    #   host_batch_origins = batch_origins[host_id *
    #                                    jax.local_device_count():(host_id + 1) *
    #                                    jax.local_device_count()]

    #   batch_rgb, batch_sigma = model_utils.pmap_model_fn(mlp_model, mlp_params,
    #                                                    host_batch_origins,
    #                                                    scene_params)
      


    #   rgb_and_features[i] = np.array(batch_rgb[0], dtype=output_dtype).reshape(rgb_and_features[i].shape)
    #   sigma[i] = np.array(batch_sigma[0], dtype=output_dtype).reshape(sigma[i].shape)

    # rgb_and_features = rgb_and_features.reshape((-1, channels))
    # sigma = sigma.reshape((-1))
    # rgb_and_features = rgb_and_features[:actual_num_rays]
    # sigma = sigma[:actual_num_rays]

    # alpha = 1.0 - np.exp(-sigma * voxel_size)
    # output_rgb_and_features = rgb_and_features.reshape(
    #     (block_coordinates_world.shape[0], block_coordinates_world.shape[1],
    #     block_coordinates_world.shape[2], channels))
    # output_alpha = alpha.reshape(block_coordinates_world.shape[0:3])

    return output_rgb_and_features * np.expand_dims(output_alpha, -1), output_alpha



def posenc(x, min_deg, max_deg, legacy_posenc_order=False):
    """Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].

    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

    Args:
      x: torch.Tensor, variables to be encoded. Note that x should be in [-pi, pi].
      min_deg: int, the minimum (inclusive) degree of the encoding.
      max_deg: int, the maximum (exclusive) degree of the encoding.
      legacy_posenc_order: bool, keep the same ordering as the original tf code.

    Returns:
      encoded: jnp.ndarray, encoded variables.
    """
    if min_deg == max_deg:
        return x
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)])
    if legacy_posenc_order:
        xb = x[Ellipsis, None, :] * scales[:, None]
        four_feat = torch.sin(torch.stack([xb, xb + 0.5 * torch.tensor(torch.pi)], dim=-2)).reshape(list(x.shape[:-1]) + [-1])
    else:
        xb = torch.reshape((x[Ellipsis, None, :] * scales[:, None]), list(x.shape[:-1]) + [-1])
        four_feat = torch.sin(torch.cat([xb, xb + 0.5 * torch.tensor(torch.pi)], dim=-1))
    return torch.cat([x] + [four_feat], dim=-1)