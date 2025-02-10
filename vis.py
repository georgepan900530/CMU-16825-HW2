import dis
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch3d
import pytorch3d.io
from pytorch3d.vis.plotly_vis import plot_scene
from tqdm.auto import tqdm
import starter.utils
import os
import imageio
from starter.utils import (
    get_device,
    get_mesh_renderer,
    load_cow_mesh,
    get_points_renderer,
    unproject_depth_image,
)
import argparse
import torch
import plotly.io as pio
from PIL import Image, ImageDraw
import math
from starter.render_generic import *


def DegreeRenders(
    output_file,
    cow_path="data/cow.obj",
    device=None,
    image_size=256,
    color=[0.7, 0.7, 1],
    fps=15,
    mesh=None,
    distance=3,
    has_textures=None,
    steps=5,
    elev=0,
    full_sphere=False,
):
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    if not mesh:
        vertices, faces = load_cow_mesh(cow_path)
        vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
        faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
        if not has_textures:
            textures = torch.ones_like(vertices)  # (1, N_v, 3)
            textures = textures * torch.tensor(color)  # (1, N_v, 3)
        else:
            temp = vertices.squeeze(0)
            textures = torch.zeros_like(temp)
            z_min = torch.min(temp[:, 2]).item()
            z_max = torch.max(temp[:, 2]).item()
            color1, color2 = has_textures[0], has_textures[1]
            for i in range(temp.shape[0]):
                alpha = (temp[i, 2] - z_min) / (z_max - z_min)
                textures[i] = color1 * alpha + color2 * (1 - alpha)
            textures = textures.unsqueeze(0)
        mesh = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures),
        )
    mesh = mesh.to(device)

    num_views = 360 // steps
    if full_sphere:
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=distance,
            elev=np.linspace(-180, 180, num_views, endpoint=False),
            azim=np.linspace(-180, 180, num_views, endpoint=False),
        )
    else:
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=distance,
            elev=elev,
            azim=np.linspace(-180, 180, num_views, endpoint=False),
        )
    # Prepare the camera:
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -distance]], device=device)

    rend = renderer(mesh.extend(num_views), cameras=many_cameras, lights=lights)
    rend = rend.cpu().numpy()[..., :3]
    rend = (rend * 255).astype(np.uint8)
    rend = list(rend)
    duration = 1000 // fps
    imageio.mimsave(output_file, rend, duration=duration, loop=0)


def vis_voxel(
    voxel,
    output_file,
    threshold=0.5,
    image_size=256,
    distance=3,
    fps=15,
    steps=5,
    elev=0,
    full_sphere=False,
    color=[0.7, 0.7, 1],
    device=None,
):
    if device is None:
        device = get_device()

    # Use cubify to convert a voxel grid to a mesh, threshold is used to determine if the voxel is occupied or not
    mesh = pytorch3d.ops.cubify(voxel, device=device, thresh=threshold)
    vertices, faces = mesh.verts_list()[0], mesh.faces_list()[0]
    vertices = vertices.unsqueeze(0)
    faces = faces.unsqueeze(0)
    textures = torch.ones_like(vertices, device=device)
    textures = textures * torch.tensor(color, device=device)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    DegreeRenders(
        output_file,
        mesh=mesh,
        image_size=image_size,
        distance=distance,
        fps=fps,
        steps=steps,
        elev=elev,
        full_sphere=full_sphere,
        color=color,
    )


def vis_point_cloud(
    pc,
    output_file,
    device=None,
    fps=15,
    image_size=256,
    distance=8,
    background_color=(1, 1, 1),
    elev=0,
    steps=20,
):
    if device is None:
        device = get_device()

    renderer = get_points_renderer(
        image_size=image_size, device=device, background_color=background_color
    )

    r = torch.tensor([0, 0, np.pi])
    r = pytorch3d.transforms.euler_angles_to_matrix(r, "XYZ")
    views = []
    for i in range(-180, 180, steps):
        for j in range(-180, 180, steps):
            R, T = pytorch3d.renderer.look_at_view_transform(
                dist=distance,
                elev=j,
                azim=i,
            )
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(
                R=r @ R, T=T, device=device
            )
            rend = renderer(pc, cameras=cameras)
            rend = rend.cpu().numpy()[0, ..., :3]
            rend = (rend * 255).astype(np.uint8)
        views.append(rend)
    duration = 1000 // fps
    imageio.mimsave(output_file, views, duration=duration, loop=0)
