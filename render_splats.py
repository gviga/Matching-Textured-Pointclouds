import torch
import math
import trimesh
import numpy as np
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)

# Function to render point clouds
@torch.no_grad()
def run_rendering(device, pointclouds, num_views, H, W, add_angle_azi=0, add_angle_ele=0, use_normal_map=False, return_images=False):
    # Create pointclouds object with colors as features

    # Get bounding box and scaling factors
    bbox = pointclouds.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bb_diff = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2.0
    scaling_factor = 0.65
    distance = torch.sqrt((bb_diff * bb_diff).sum())
    distance *= scaling_factor

    # Calculate camera positions
    steps = int(math.sqrt(num_views))
    end = 360 - 360 / steps
    elevation = torch.linspace(start=0, end=end, steps=steps).repeat(steps) + add_angle_ele
    azimuth = torch.linspace(start=0, end=end, steps=steps)
    azimuth = torch.repeat_interleave(azimuth, steps) + add_angle_azi
    bbox_center = bbox_center.unsqueeze(0)
    rotation, translation = look_at_view_transform(
        dist=distance, azim=azimuth, elev=elevation, device=device, at=bbox_center
    )
    camera = PerspectiveCameras(R=rotation, T=translation, device=device)

    # Rasterization settings
    rasterization_settings = PointsRasterizationSettings(
        image_size=H,
        radius=0.01,
        points_per_pixel=1,  # Increase for better blending of points
        bin_size=None,  # Automatic bin size calculation
        max_points_per_bin=None  # Automatic bin size calculation
    )

    # Render pipeline
    rasterizer = PointsRasterizer(cameras=camera, raster_settings=rasterization_settings)
    batch_renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=(1, 1, 1))
    )
    camera_centre = camera.get_camera_center()

    # Render point clouds
    batch_points = pointclouds.extend(num_views)
    rendered_images = batch_renderer(batch_points)

    fragments = rasterizer(batch_points)
    raw_depth = fragments.zbuf
    # Extract RGB images from rendered images
    rgb_images = rendered_images[..., :3]

    return rendered_images, rgb_images, camera, raw_depth


# Batch render function
def batch_render(device, pc, num_views, H, W, use_normal_map=False, return_images=False):
    trials = 0
    add_angle_azi = 0
    add_angle_ele = 0
    while trials < 5:
        try:
            return run_rendering(device, pc, num_views, H, W, use_normal_map=use_normal_map, return_images=return_images)
        except torch.linalg.LinAlgError as e:
            trials += 1
            print("LinAlgError during rendering, retrying", trials)
            add_angle_azi = torch.randn(1)
            add_angle_ele = torch.randn(1)
            continue

