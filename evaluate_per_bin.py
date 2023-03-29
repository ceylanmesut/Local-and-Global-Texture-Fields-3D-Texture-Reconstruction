import argparse
import pandas as pd
import os
import glob
import sys

from mesh2tex import config
from mesh2tex.eval import evaluate_generated_images

categories = {'02958343': 'cars', '03001627': 'chairs',
              '02691156': 'airplanes', '04379243': 'tables',
              '02828884': 'benches', '02933112': 'cabinets',
              '04256520': 'sofa', '03636649': 'lamps',
              '04530566': 'vessels'}

parser = argparse.ArgumentParser(
    description='Generate Color for given mesh.'
)

parser.add_argument('config', type=str, help='Path to config file.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
base_path = cfg['test']['vis_dir']

# loop through all bins
for i in range(24):

  binIdx = i
  print('Processing bin=============================================', str(i))
  base_path_temp = base_path + '_bin_' + str(binIdx)
  print(base_path_temp)

  sides = os.listdir(base_path_temp)
  for side in sides:
    if side == 'back' or side == 'front' or side == 'sideLeft' or side == 'sideRight' or side == 'top':
      path1 = base_path_temp + '/' + side + '/fake/' #os.path.join(category_path, 'fake/')
      path2 = base_path_temp + '/' + side + '/real/' #os.path.join(category_path, 'real/')
      evaluation = evaluate_generated_images('all', path1, path2)

      df = pd.DataFrame(evaluation, index=[side])
      df.to_pickle(os.path.join(base_path_temp + '/' + side, 'eval.pkl'))
      df.to_csv(os.path.join(base_path_temp + '/' + side, 'eval.csv'))
      print('EVALUATION OF  ' + side + ' FINISHED===============================')

  print('-----------------------------------------')
  print('Evaluation of bin ' + str(i) + ' finished')
