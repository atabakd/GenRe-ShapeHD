import trimesh
from util.util_img import depth_to_mesh_df, resize
from skimage import measure
import numpy as np
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
  from skimage.measure import block_reduce
  ds_voxel = block_reduce(voxel, (16,16,16))
  ax.voxels(ds_voxel , edgecolor="k", facecolors=[1, 0, 0, 0.1])
  # ax.view_init(90, 270)
  ax.view_init(0, 180)
  plt.draw()

def render_model(mesh, sgrid):
  index_tri, index_ray, loc = mesh.ray.intersects_id(
    ray_origins=sgrid, ray_directions=-sgrid, multiple_hits=False, return_locations=True)
  loc = loc.reshape((-1, 3))

  grid_hits = sgrid[index_ray]
  dist = np.linalg.norm(grid_hits - loc, axis=-1)
  dist_im = np.ones(sgrid.shape[0])
  dist_im[index_ray] = dist
  im = dist_im
  return im


def make_sgrid(b, alpha, beta, gamma):
  res = b * 2
  pi = np.pi
  phi = np.linspace(0, 180, res * 2 + 1)[1::2]
  theta = np.linspace(0, 360, res + 1)[:-1]
  grid = np.zeros([res, res, 3])
  for idp, p in enumerate(phi):
    for idt, t in enumerate(theta):
      grid[idp, idt, 2] = np.cos((p * pi / 180))
      proj = np.sin((p * pi / 180))
      grid[idp, idt, 0] = proj * np.cos(t * pi / 180)
      grid[idp, idt, 1] = proj * np.sin(t * pi / 180)
  grid = np.reshape(grid, (res * res, 3))
  return grid


def plot_3d(points):
  from mpl_toolkits.mplot3d import Axes3D
  from matplotlib import pyplot as plt
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  filter_idxs = \
    np.where(points[..., 2].reshape(-1, 1) < points.max())[0]
  x = points[..., 0].reshape(-1, 1)[filter_idxs]
  y = points[..., 1].reshape(-1, 1)[filter_idxs]
  z = points[..., 2].reshape(-1, 1)[filter_idxs]
  ax.scatter(x, y, z, alpha=0.9)
  ax.set_xlabel('X Label')
  ax.set_ylabel('Y Label')
  ax.set_zlabel('Z Label')
  plt.show()


def render_spherical(data, mask, res=128, obj_path=None, debug=False):
  depth_im = data['depth'][0, 0, :, :]
  th = data['depth_minmax']
  depth_im = resize(depth_im, 480, 'vertical')
  im = resize(data['mask_img'], 480, 'vertical')
  gt_sil = np.where(im > 0.95, 1, 0)
  depth_im = depth_im * gt_sil
  depth_im = depth_im[:, :, np.newaxis]
  b = 64
  tdf = depth_to_mesh_df(depth_im, th, False, 1.0, cam_dist=0.5934864717638447, res=res)
  from scipy.ndimage import zoom
  # tdf = zoom(tdf, (4, 1, 1), order=2)
  # tdf = tdf[
  #       tdf.shape[0]//2-res//2:tdf.shape[0]//2+res//2,
  #       # :,
  #       #     tdf.shape[0] // 2 - res//2:tdf.shape[0] // 2 + res//2,
  #           :,
  #           # tdf.shape[0] // 2 - res//2:tdf.shape[0] // 2 + res//2
  #       :
  #       ]
  try:
    verts, faces, normals, values = measure.marching_cubes_lewiner(
      tdf, 0.99999 / res, spacing=(1 / res, 1 / res, 1 / res))
    mesh = trimesh.Trimesh(vertices=verts - 0.5, faces=faces)
  # mesh = trimesh.Trimesh(vertices=verts - 0.2, faces=faces)
  # mesh.rezero()
  # meshvoxel = trimesh.voxel.local_voxelize(mesh, mesh.center_mass, pitch=0.25 / 513, radius=256)[0]  # 25cm devided in 129 voxels
  # meshvoxel = trimesh.voxel.local_voxelize(mesh, (0., 0., 0.), pitch=0.25 / 513, radius=256)[0]  # 25cm devided in 513 voxels
  # pitch = 0.25/res # 25cm devided in # voxels
  # meshvoxel, origin = trimesh.voxel.local_voxelize(mesh, (0., 0., 0.), pitch=pitch, radius=256)
  # meshvoxel, origin = trimesh.voxel.local_voxelize(mesh, (0.12475634/8, 0.12475634/8, 0.12475634/8), pitch=pitch, radius=256)
  # return meshvoxel

    sgrid = make_sgrid(b, 0, 0, 0)
    im_depth = render_model(mesh, sgrid)
    im_depth = im_depth.reshape(2 * b, 2 * b)
    im_depth = np.where(im_depth > 1, 1, im_depth)
  except:
    im_depth = np.ones([2 * b, 2 * b])
  #   return im_depth
  return im_depth

# def render_spherical(data, mask, obj_path=None, debug=False, res=128):
#   depth_im = data['depth'][0, 0, :, :]
#   th = data['depth_minmax']
#   depth_im = resize(depth_im, 480, 'vertical')
#   im = resize(mask, 480, 'vertical')
#   gt_sil = np.where(im > 0.95, 1, 0)
#   depth_im = depth_im * gt_sil
#   depth_im = depth_im[:, :, np.newaxis]
#   b = res//2
#   tdf = depth_to_mesh_df(depth_im, th, False, 1.0, 2.2)
#   try:
#     verts, faces, normals, values = measure.marching_cubes_lewiner(
#       tdf, 0.999 / res, spacing=(1 / res, 1 / res, 1 / res))
#     mesh = trimesh.Trimesh(vertices=verts - 0.5, faces=faces)
#     sgrid = make_sgrid(b, 0, 0, 0)
#     im_depth = render_model(mesh, sgrid)
#     im_depth = im_depth.reshape(2 * b, 2 * b)
#     im_depth = np.where(im_depth > 1, 1, im_depth)
#   except:
#     im_depth = np.ones([res, res])
#     return im_depth
#   return im_depth

# https://codereview.stackexchange.com/questions/79032/generating-a-3d-point-cloud
def depth_to_3d(depth,
                depth_intrinsics=np.array([[567.6188, 0, 310.0724], [0, 568.1618, 242.7912], [0, 0, 1.]])
                ):
  """Transform a uint16 depth image into a point cloud with one point for each
  pixel in the image, using the camera transform for a camera
  centred at cx, cy with field of view fx, fy.
  depth is a 2-D ndarray with shape (rows, cols) containing
  depths in uint16. The result is a 3-D array with
  shape (rows, cols, 3). Pixels with invalid depth in the input have
  NaN for the z-coordinate in the result.
  """
  from skimage.transform import rescale, resize, downscale_local_mean
  # depth = resize(depth, (640/4, 480/4))

  cx, cy = depth_intrinsics[0, 2], depth_intrinsics[1, 2]
  # cx, cy = -9.13, 2.79
  fx, fy = depth_intrinsics[0, 0], depth_intrinsics[1, 1]
  rows, cols, _ = depth.shape
  c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
  valid = (depth >= 0) & (depth <= np.iinfo(np.uint16).max)
  z = np.where(valid, depth, np.nan)
  x = np.where(valid, z * (c - cx) / fx, 0)
  y = np.where(valid, z * (r - cy) / fy, 0)
  return np.dstack((x, y, z))
