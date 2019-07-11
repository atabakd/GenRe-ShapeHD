from __future__ import print_function, division
## ATABAK, check if the imports are correct
from toolbox import spherical_proj
from toolbox.cam_bp.cam_bp.modules.camera_backprojection_module import Camera_back_projection_layer
import torch
import numpy as np
import skimage
import json
from matplotlib import pyplot as plt


def depth2sphere_tool(depth_path, mask_path):
  proj_depth = Camera_back_projection_layer().cuda()
  render_spherical = spherical_proj.render_spherical().cuda()

  depth = skimage.io.imread(depth_path, as_gray=True)
  mask = skimage.io.imread(mask_path, as_gray=True)
  mine = False
  if mine:
    with open("data/0006.json", "r") as j_file: # DRILL
      data = json.load(j_file)[996]
    # with open("data/0000.json", "r") as j_file: # Box
    #   data = json.load(j_file)[0]
  
    #
    depth_minmax = data["depth_minmax"]
  else:
    depth_minmax = np.load('data/02691156_fff513f407e00e85a9ced22d91ad7027_view019.npy')
  print(depth_minmax)
  print(np.amin(depth))
  print(np.amax(depth))
  pred_abs_depth = get_abs_depth(depth,mask,depth_minmax)
  print(type(pred_abs_depth))
  pred_abs_depth = pred_abs_depth.type(torch.Tensor).cuda()
  
  #pred_abs_depth = pred_abs_depth.squeeze().cpu().numpy()
  print(depth_minmax)
  # print(np.amin(pred_abs_depth))
  # print(np.amax(pred_abs_depth))
  # plt.imshow(pred_abs_depth)
  # plt.show()
  proj = proj_depth(pred_abs_depth)
  proj_df = proj.squeeze()

  voxel = proj_df.cpu().numpy()
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.set_title('partial_voxel')
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")
  ax.set_aspect('equal')

  ax.voxels(voxel[::1,::1,::1], edgecolor="k") 
  ax.view_init(0, 180)
  plt.draw()
  plt.savefig('vox_partial_sampled.png')
  plt.show()
  plt.clf()
  print(proj.shape)
  sph = render_spherical(torch.clamp(proj * 50, 1e-5, 1 - 1e-5))
  sph = sph.squeeze().cpu().numpy()
  # sph = np.flipud(sph)
  print(sph.shape)
  # plt.ion()
  plt.imshow(sph)
  plt.savefig('depth2sph.png')
  # plt.show()

def get_abs_depth(pred, mask,pred_depth_minmax):
  pred_depth = pred
  #pred_depth = postprocess(pred_depth)
  #pred_depth_minmax = pred['depth_minmax'].detach()

  print('min_depth before proc')
  print(np.amin(pred_depth[mask > 0.5]))
  print(np.amax(pred_depth[mask > 0.5]))

  pred_depth = scale_depth(pred_depth,mask)

  print(np.amin(pred_depth[mask > 0.5]))
  print(np.amax(pred_depth[mask > 0.5]))

  pred_abs_depth = to_abs_depth(1 - pred_depth, pred_depth_minmax)
  print(pred_abs_depth.shape)
  print('inside abs')
  print(np.amin(pred_abs_depth.squeeze().cpu().numpy()))
  print(np.amax(pred_abs_depth.squeeze().cpu().numpy()))
  #silhou = torch.from_numpy(postprocess(mask)).unsqueeze(0)
  silhou = torch.from_numpy(mask).unsqueeze(0)
  silhou = silhou.unsqueeze(0)
  print(silhou.shape)
  pred_abs_depth[silhou < 0.5] = 0
  print('depth')
  print(np.amin(pred_abs_depth.squeeze().cpu().numpy()[mask > 0.5]))
  print(np.amax(pred_abs_depth.squeeze().cpu().numpy()[mask > 0.5]))
  # pred_abs_depth = pred_abs_depth.permute(0, 1, 3, 2)
  # pred_abs_depth = torch.flip(pred_abs_depth, [2])
  return pred_abs_depth

# def postprocess(tensor, bg=1.0, input_mask=None):
#   scale_25d = 0.00100
#   scaled = tensor / scale_25d
#   return scaled

def scale_depth(depth, mask):

  
  # bmin = torch.tensor(np.amin(depth[mask > 0.5]),dtype=torch.double)
  # bmax = torch.tensor(np.amax(depth[mask > 0.5]),dtype=torch.double)
  bmin = np.amin(depth[mask > 0.5])
  bmax = np.amax(depth[mask > 0.5])
  scale_depth = np.interp(depth,(bmin, bmax), (0,1))
  # depth_min = bmin.view(-1, 1, 1, 1) #+ 2.2 - (bmax - bmin)
  # depth_max = bmax.view(-1, 1, 1, 1) #+ 2.2 - (bmax - bmin)
  # scale_depth = depth * (depth_max - depth_min + 1e-4) #+ depth_min
  return scale_depth


def to_abs_depth(rel_depth, depth_minmax):

  #torch.from_numpy(voxel).float().to(device)
  bmin = torch.tensor(depth_minmax[0],dtype=torch.double)
  bmax = torch.tensor(depth_minmax[1],dtype=torch.double)

  mine = True
  hack = True
  if mine and hack:
    depth_min = 2.2 - bmin.view(-1, 1, 1, 1)/2#+ 2.2 - (bmax - bmin)
    depth_max = 2.2 + bmax.view(-1, 1, 1, 1)/2 #+ 2.2 - (bmax - bmin)

    mean = (bmin+bmax)/2
    print(mean.item())
    # depth_min = 2.2 + bmin - mean.view(-1, 1, 1, 1) #- (bmin+bmax)/2 #+ 2.2 - (bmax - bmin))
    # depth_max = 2.2 + bmax - mean.view(-1, 1, 1, 1) #- (bmin+bmax)/2 #+ 2.2 - (bmax - bmin)

    # depth_min = torch.tensor(2.15,dtype=torch.double).view(-1, 1, 1, 1)
    # depth_max = torch.tensor(2.45,dtype=torch.double).view(-1, 1, 1, 1)
  else:
    depth_min = bmin.view(-1, 1, 1, 1) #+ 2.2 - (bmax - bmin)
    depth_max = bmax.view(-1, 1, 1, 1) #+ 2.2 - (bmax - bmin)
  abs_depth = rel_depth * (depth_max - depth_min + 1e-4) + depth_min
  return abs_depth


# Calling the function
depth2sphere_tool('data/drill_center_depth2.png', 'data/drill_center_mask2.png')