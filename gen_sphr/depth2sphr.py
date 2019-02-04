from __future__ import print_function, division

import numpy as np
import skimage
import json
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import torch

import sys
import os
sys.path.append(os.getcwd())
# print(sys.path)

from util.util_sph import render_spherical
from toolbox import spherical_proj
from toolbox.cam_bp.cam_bp.functions import SphericalBackProjection


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
  ax.voxels(ds_voxel, edgecolor="k", facecolors=[1, 0, 0, 0.1])
  # ax.view_init(90, 270)
  ax.view_init(0, 180)
  plt.draw()

  # plt.savefig('voxelized_plane')
  plt.show()
  # for angle in range(0, 360):
  #   ax.view_init(0, angle)
  #   plt.draw()
  #   plt.pause(.001)

def depth2sphere(data):
  '''
  :param data: a dictionary of an individual sample, coming from json file
  :return: sph_map, sph_centered_map
  '''

  depth_path = data['mask'].replace('mask', 'depth').replace('.png', '1.png').replace('media', 'mnt')
  mask_path = data['mask'].replace('mask', 'bin_mask').replace('media', 'mnt')
  # import pdb; pdb.set_trace()
  if not os.path.exists(depth_path) or not os.path.exists(mask_path): return None
  # import pdb; pdb.set_trace()
  depth_img = skimage.io.imread(depth_path, as_gray=True)
  mask_img = skimage.io.imread(mask_path, as_gray=True)
  # print("my depth minmax is: ", depth_minmax)
  depth_img = skimage.util.invert(depth_img)

  depth_img = depth_img[None, None, :, :]
  data['depth_img'] = depth_img
  data['mask_img'] = mask_img

  sph_map, sph_centered_map = render_spherical(data, res=129)
  return sph_map, sph_centered_map

  # plt.imshow(sph)
  # plt.show()
  # plt.savefig('depth2sphere.png')
  # sphere2vox(sph.astype(np.float32), res=128 if mine else 128)

  # from skimage.measure import block_reduce
  # sph = block_reduce(sph, (4,4,4))
  # vox2sphere(sph.astype(np.float64))


def vox2sphere(data):
  vox_path = data['mask'].replace('mask', 'voxel').replace('png', 'npz').replace('media', 'mnt')
  if not os.path.exists(vox_path): return None
  voxel = np.load(vox_path)['voxel'].astype(np.float32)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # if invoxel is None:
  #   mine = True
  #   if mine:
  #     # with open('data/rotated_mesh.binvox', 'rb') as f:
  #     # with open('data/incomplete_mesh.binvox', 'rb') as f:
  #     #   m1 = binvox_rw.read_as_3d_array(f)
  #     #
  #     # voxel = m1.data.astype(np.float32)
  #     box = True
  #     if box:
  #       data = np.load("/mnt/Extra/dsl/GenRe-ShapeHD/gen_sphr/data/rotated_mesh_voxelized.npz")
  #     else:
  #       data = np.load("/mnt/Extra/dsl/GenRe-ShapeHD/gen_sphr/data/rotated_mesh_voxelized_drill.npz")
  #     voxel = data['voxel'].astype(np.float64)
  #
  #   else:
  #     data = np.load(
  #       'data/02691156_fff513f407e00e85a9ced22d91ad7027_view019_gt_rotvox_samescale_128.npz')
  #     voxel = data['voxel']
  #   # plot_voxel(voxel)
  #
  # else:
  #   voxel = invoxel
  projector = spherical_proj.render_spherical().to(device)
  voxeltensor = torch.from_numpy(voxel).float().to(device)
  voxeltensor = voxeltensor.unsqueeze(0)
  voxeltensor = voxeltensor.unsqueeze(0)
  spherical = projector.forward(voxeltensor)
  spherical = spherical.squeeze().to("cpu").numpy()
  spherical = np.flipud(spherical)
  return spherical
  # spherical = np.fliplr(spherical)
  # plt.imshow(spherical)
  # plt.savefig('vox2sphere.png')  # if invoxel is None else plt.savefig('depth2sphere.png')
  # plt.show()
  # sphere2vox(spherical)
  # np.savez('data/depth_centered.npz', spherical=spherical)
  # spherical = np.load('data/depth_centered.npz')['spherical']
  # plt.imshow(spherical)
  # plt.show()


def sphere2vox(sphere, res=128):
  # Init
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


if __name__ =='__main__':
  import argparse
  # print('/'.join(os.getcwd().split('/')))
  # print(os.getcwd())
  # print(sys.path)

  parser = argparse.ArgumentParser()
  parser.add_argument('--division_num', type=int, default=0,
                      help='division number 0<=division_num<60')
  # parser.add_argument('--output_path', type=str,
  #                     default='/media/hdd/YCBvideo/YCB_Video_Dataset/Generated_YCB_Video_Dataset',
  #                     help='output image path')

  args = parser.parse_args()

  with open(os.path.join('json', '{:04d}.json'.format(args.division_num)), "r") as j_file:
    print(os.path.join('json', '{:04d}.json'.format(args.division_num)))
    data_list = json.load(j_file)

  for data in data_list:
    sph_path = data['mask'].replace('mask', 'spherical').replace('.png', 'npz').replace('media', 'mnt')
    if os.path.exists(sph_path): continue
    d_sphere = depth2sphere(data)
    if d_sphere is None: continue
    d_sph_map, d_sph_centered_map = d_sphere
    vox_sphere = vox2sphere(data)
    if vox_sphere is None: continue
    if not os.path.exists(os.path.dirname(sph_path)):
      os.makedirs(sph_path)

    np.savez(sph_path, depth_spherical=d_sph_map, obj_spherical=vox_sphere, depth_spherical_centered=d_sph_centered_map)