import sys
sys.path.append('')

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
# modified from https://github.com/benjiebob/SMALViewer/blob/master/p3d_renderer.py
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
from pytorch3d.io import IO
from pytorch3d.structures import Pointclouds

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    PerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRasterizer, BlendParams,
    PointLights, SoftPhongShader, SoftSilhouetteShader, TexturesVertex,MeshRenderer
)

import torch
import torch.nn as nn
import numpy as np

import cv2

class PointCloudRenderer(nn.Module):
    def __init__(self, img_size=800, device='cuda'):
        super(PointCloudRenderer, self).__init__()
        self.device = device

        # Initialize a camera.
        R, T = look_at_view_transform(20, 10, 0)
        cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)

        raster_settings = PointsRasterizationSettings(
            image_size=img_size, 
            radius = 0.003,
            points_per_pixel = 10
        )

        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        self.renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor()
        ).to(device)

    def forward(self, points):
        points_rgb = torch.ones_like(points)
        point_cloud = Pointclouds(points=points, features=points_rgb).to(self.device)
        img = self.renderer(point_cloud)
        return img

    def save_obj(self, points, save_loc):
        points_rgb = torch.ones_like(points)
        point_cloud = Pointclouds(points=points, features=points_rgb).to(self.device)

        IO().save_pointcloud(point_cloud, save_loc)

class ObjRenderer(nn.Module):
    def __init__(self, img_size=800, device='cuda'):
        super(ObjRenderer, self).__init__()
        self.device = device

        raster_settings = RasterizationSettings(
            image_size=img_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            bin_size=None
        )

        R, T = look_at_view_transform(20, 10, 0)
        cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)
        lights = PointLights(device=device, location=[[2.0, 2.0, 0.0]])

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=R.device, 
                cameras=cameras,
                lights=lights,
            )
        ).to(device)

    def forward(self, mesh):
        verts = mesh.verts_padded()
        faces = mesh.faces_padded()

        verts_rgb = torch.ones_like(verts) 
        textures = TexturesVertex(verts_features=verts_rgb).to(self.device)

        mesh = Meshes(verts=verts, faces=faces, textures = textures)
        img = self.renderer(mesh)[:, :,:,:3]

        return img

    def save_obj(self, mesh, save_loc):
        IO().save_mesh(mesh, save_loc)

def images_to_grid(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    im = torchvision.transforms.ToPILImage()(grid) 
    # im = ImageOps.invert(im)
    im.save(path)

def render_obj(render, points, save_loc):
    render.save_obj(points, save_loc)


def visualise_point_cloud(points, img_loc, save_grid=True):
    device = points.device
    renerer = PointCloudRenderer(device=device)
    images = renerer(points).cpu()

    if save_grid:
        images_to_grid(images.permute(0, 3,1,2), img_loc + '.png', nrow=4)

    else:
        for i in images:
            plt.imsave(img_loc + "_{:03d}.png".format(i), images[i].numpy())

def visualise_mesh(mesh, img_loc, save_grid=True):
    device = mesh.device

    renerer = ObjRenderer(device=device)
    images = renerer(mesh).cpu()

    if save_grid:
        images_to_grid(images.permute(0, 3,1,2), img_loc + '.png', nrow=4)

    else:
        for i in images:
            plt.imsave(img_loc + "_{:03d}.png".format(i), images[i].numpy())




if __name__ == '__main__':
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.io import load_obj



    verts, faces, _ = load_obj("../ShapeNetCore/02691156/1a04e3eab45ca15dd86060f189eb133/models/model_normalized.obj")

    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale

    mesh = Meshes(verts=[verts], faces=[faces.verts_idx])

    pc = sample_points_from_meshes(mesh, 5000)
    
    save_dir = './outputs/test'
    
    visualise_point_cloud(pc,save_dir + 'pc')
    visualise_mesh(mesh,save_dir + 'mesh')

