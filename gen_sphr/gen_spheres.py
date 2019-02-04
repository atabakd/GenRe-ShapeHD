from __future__ import division, print_function
import subprocess

def gen_voxels(num):
  com = 'python gen_sphr/depth_to_sphere.py --division_num {}'.format(num)
  subprocess.call([com], shell=True)

if __name__ == '__main__':

  from joblib import Parallel, delayed
  Parallel(n_jobs=6)(delayed(gen_voxels)(i) for i in range(60))