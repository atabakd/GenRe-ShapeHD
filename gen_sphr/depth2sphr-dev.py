from __future__ import print_function, division

import numpy as np
from util.util_sph import render_spherical
from toolbox import spherical_proj
from toolbox.cam_bp.cam_bp.functions import SphericalBackProjection
# import cv2
import skimage
import json
import time
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from gen_sphr import binvox_rw
import torch


def plot_voxel(voxel):
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")
  ax.set_aspect('equal')

  # voxel = voxel[20:80,20:80,20:80].copy()
  from skimage.measure import block_reduce
  ds_voxel = block_reduce(voxel, (4,4,4))
  ax.voxels(ds_voxel , edgecolor="k", facecolors=[1, 0, 0, 0.1])
  # ax.view_init(90, 270)
  ax.view_init(0, 180)
  plt.draw()

  # plt.savefig('voxelized_plane')
  plt.show()
  # for angle in range(0, 360):
  #   ax.view_init(0, angle)
  #   plt.draw()
  #   plt.pause(.001)

def depth2sphere(depth_address):
  mine = True
  if mine:
    box = True
    if box:

      depth = skimage.io.imread(depth_address, as_gray=True)#[np.newaxis, np.newaxis, ...]
      mask = skimage.io.imread(depth_address.replace("depth", "bin_mask").replace("1.png", ".png"), as_gray=True)
      with open("/mnt/Extra/dsl/GenRe-ShapeHD/gen_sphr/data/0000.json", "r") as j_file:
        data = json.load(j_file)[0]

    else:
      depth_address = depth_address.replace("000001-depth-001", "000381-depth-021")
      depth = skimage.io.imread(depth_address, as_gray=True)  # [np.newaxis, np.newaxis, ...]
      mask = skimage.io.imread(depth_address.replace("depth", "bin_mask").replace("1.png", ".png"), as_gray=True)
      with open("/mnt/Extra/dsl/GenRe-ShapeHD/gen_sphr/data/0006.json", "r") as j_file:
        data = json.load(j_file)[996]
  #
    from scipy import ndimage as nd

    # depth = nd.zoom(depth, 2, mode="nearest")
    # mask = nd.zoom(mask, 2, mode="nearest")
    # plt.imshow(depth)
    # plt.show()
    depth_minmax = data["depth_minmax"]
    print("my depth minmax is: ", depth_minmax)
    depth = skimage.util.invert(depth)
    # depth = skimage.util.pad(depth, ([np.abs(np.subtract(*depth.shape))//2]*2, (0, 0)), 'constant', constant_values=0.)
    # mask = skimage.util.pad(mask, ([np.abs(np.subtract(*mask.shape))//2]*2, (0, 0)), 'constant', constant_values=0.)
  # plt.imshow(mask_img)
  # plt.show()
  #   plt.imshow(depth)
  #   plt.show()

  # plt.imshow(mask)
  # plt.show()
  # plt.imshow(depth)
  # plt.show()
  else:

    depth_minmax = np.load('data/02691156_fff513f407e00e85a9ced22d91ad7027_view019.npy')
    depth = skimage.io.imread('data/depth.png', as_gray=True)
    mask = skimage.io.imread('data/mask.png', as_gray=True)
    print("his depth minmax is: ", depth_minmax)

  # plt.imshow(mask)
  # plt.show()
  # plt.imshow(depth)
  # plt.show()

  depth = depth[None, None, :, :]
  # print(type(depth))
  # print(depth.shape)

  # There is a better way for sure. I'm not an expert in python... Save and Load was the easiest way for me...
  # np.savez('/home/pvicente/software/git/GenRe-ShapeHD/tmpData/mine/tmpdata2.npz', depth=depth,
  #          depth_minmax=depth_minmax)
  # data = np.load('/home/pvicente/software/git/GenRe-ShapeHD/tmpData/mine/tmpdata2.npz')
  data = {'depth': depth, 'depth_minmax': depth_minmax}

  sph = render_spherical(data, mask,
                         res=129 if mine else 128
                         )  # , I change the code in Perlis to receive the res as well, The default is 64
  plt.imshow(sph)
  # plt.show()
  plt.savefig('depth2sphere.png')
  sphere2vox(sph.astype(np.float32), res=128 if mine else 128)

  # from skimage.measure import block_reduce
  # sph = block_reduce(sph, (4,4,4))
  # vox2sphere(sph.astype(np.float64))


def vox2sphere(invoxel=None):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  if invoxel is None:
    mine = True
    if mine:
      # with open('data/rotated_mesh.binvox', 'rb') as f:
      # with open('data/incomplete_mesh.binvox', 'rb') as f:
      #   m1 = binvox_rw.read_as_3d_array(f)
      #
      # voxel = m1.data.astype(np.float32)
      box = True
      if box:
        data = np.load("/mnt/Extra/dsl/GenRe-ShapeHD/gen_sphr/data/rotated_mesh_voxelized.npz")
      else:
        data = np.load("/mnt/Extra/dsl/GenRe-ShapeHD/gen_sphr/data/rotated_mesh_voxelized_drill.npz")
      voxel = data['voxel'].astype(np.float64)

    else:
      data = np.load(
        'data/02691156_fff513f407e00e85a9ced22d91ad7027_view019_gt_rotvox_samescale_128.npz')
      voxel = data['voxel']
    # plot_voxel(voxel)

  else:
    voxel = invoxel
  projector = spherical_proj.render_spherical().to(device)
  voxeltensor = torch.from_numpy(voxel).float().to(device)
  voxeltensor = voxeltensor.unsqueeze(0)
  voxeltensor = voxeltensor.unsqueeze(0)
  spherical = projector.forward(voxeltensor)
  spherical = spherical.squeeze().to("cpu").numpy()
  spherical = np.flipud(spherical)
  # spherical = np.fliplr(spherical)
  plt.imshow(spherical)
  plt.savefig('vox2sphere.png')  # if invoxel is None else plt.savefig('depth2sphere.png')
  # plt.show()
  sphere2vox(spherical)
  # np.savez('data/depth_centered.npz', spherical=spherical)
  # spherical = np.load('data/depth_centered.npz')['spherical']
  # plt.imshow(spherical)
  # plt.show()


def sphere2vox(sphere, res=128):
  # Init
  sphere = sphere.astype(np.float32)
  grid = spherical_proj.gen_sph_grid(res)
  proj_spherical = SphericalBackProjection().apply
  # margin = 16
  batch_size = 1
  sph_np = sphere
  sph = torch.from_numpy(sph_np.copy())
  #
  # print(sph.shape)
  sph = sph.unsqueeze(0)
  sph = sph.unsqueeze(0).cuda()
  # print(sph.shape)
  grid = grid.expand(1, -1, -1, -1, -1)
  grid = grid[0, :, :, :, :]
  grid = grid.expand(batch_size, -1, -1, -1, -1).cuda()

  # crop_sph = sph[:, :, margin:h - margin, margin:w - margin]
  proj_df, cnt = proj_spherical(1 - sph, grid, res)
  mask = torch.clamp(cnt.detach(), 0, 1)
  proj_df = (-proj_df + 1 / res) * res
  proj_df = proj_df * mask
  # print(proj_df.shape)
  proj_df = proj_df.squeeze()
  proj_df = proj_df.squeeze()
  voxel = proj_df.cpu().numpy()
  plot_voxel(voxel)
  # fig = plt.figure()
  # ax = fig.gca(projection='3d')
  #
  # ax.set_xlabel("x")
  # ax.set_ylabel("y")
  # ax.set_zlabel("z")
  # ax.set_aspect('equal')
  # ax.voxels(voxel, edgecolor="k")
  # # ori = [0, 90, 180, 270]
  # # for i in ori:
  # #   for j in ori:
  #
  # ax.view_init(0, 180)
  # plt.draw()
  # plt.show()
  # # plt.pause(600)


if __name__ =='__main__':
  depth2sphere("/mnt/Extra/dsl/GenRe-ShapeHD/gen_sphr/data/000001-depth-001.png")
  # vox2sphere()