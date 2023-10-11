import os
from pathlib import Path

from nerfstudio.cameras.rays import Frustums, RaySamples

import torch
import torch.onnx





def get_field_input():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_rays = 1024
    num_samples = 256
    positions = torch.rand((num_rays, num_samples, 3), dtype=torch.float32, device=device)
    directions = torch.rand_like(positions)
    frustums = Frustums(
        origins=positions,
        directions=directions,
        starts=torch.zeros((*directions.shape[:-1], 1), device=device),
        ends=torch.zeros((*directions.shape[:-1], 1), device=device),
        pixel_area=torch.ones((*directions.shape[:-1], 1), device=device),
    )
    ray_samples = RaySamples(
        frustums=frustums,
        camera_indices=torch.zeros(
            (num_rays, 1, 1),
            device=device,
            dtype=torch.int32,
        ),
    )

    aabb_scale = 1.0
    aabb = torch.tensor(
        [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]],
        dtype=torch.float32,
        device=device,
    )

    return ray_samples, aabb