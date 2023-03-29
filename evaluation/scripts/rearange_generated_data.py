import os
import numpy as np

# The purpose of this script is to rearrange the data that is created by the generate_per_bin.py script such that it is compatible
# with evaluating with the evaluate_per_bin.py script.

# path to generated data of texture fields
test_list = '/srv/beegfs02/scratch/texture/data/single_view_3d_w_texture/src/texture_project/out/singleview/new_data' # modified according to our generated data.


bins = os.listdir(test_list)

# loop through each bin folder and rearrange the data
for Bin in bins:
	print(Bin)
	if Bin == 'vis' or Bin == 'log' or Bin == 'eval_fix':
		continue
	else:
		if not os.path.exists(test_list + '/' + Bin + '/front'): #001.png
			os.makedirs(test_list + '/' + Bin + '/front/real')
			os.makedirs(test_list + '/' + Bin + '/front/fake')
		if not os.path.exists(test_list + '/' + Bin + '/back'): # 003.png
			os.makedirs(test_list + '/' + Bin + '/back/real')
			os.makedirs(test_list + '/' + Bin + '/back/fake')
		if not os.path.exists(test_list + '/' + Bin + '/top'): # 004.png
			os.makedirs(test_list + '/' + Bin + '/top/real')
			os.makedirs(test_list + '/' + Bin + '/top/fake')
		if not os.path.exists(test_list + '/' + Bin + '/sideLeft'): # 000.png
			os.makedirs(test_list + '/' + Bin + '/sideLeft/real')
			os.makedirs(test_list + '/' + Bin + '/sideLeft/fake')
		if not os.path.exists(test_list + '/' + Bin + '/sideRight'): # 002.png
			os.makedirs(test_list + '/' + Bin + '/sideRight/real')
			os.makedirs(test_list + '/' + Bin + '/sideRight/fake')
		# go through the fake directory
		fakes = os.listdir(test_list + '/' + Bin + '/fake')
		for fake in fakes:
			if fake.endswith('000.png'):
				os.rename(test_list + '/' + Bin + '/fake/' + fake, test_list + '/' + Bin + '/sideLeft/fake/' + fake)
			elif fake.endswith('001.png'):
				os.rename(test_list + '/' + Bin + '/fake/' + fake, test_list + '/' + Bin + '/front/fake/' + fake)
			elif fake.endswith('002.png'):
				os.rename(test_list + '/' + Bin + '/fake/' + fake, test_list + '/' + Bin + '/sideRight/fake/' + fake)
			elif fake.endswith('003.png'):
				os.rename(test_list + '/' + Bin + '/fake/' + fake, test_list + '/' + Bin + '/back/fake/' + fake)
			elif fake.endswith('004.png'):
				os.rename(test_list + '/' + Bin + '/fake/' + fake, test_list + '/' + Bin + '/top/fake/' + fake)
		# go through the real directory
		reals = os.listdir(test_list + '/' + Bin + '/real')
		for real in reals:
			if real.endswith('000.png'):
				os.rename(test_list + '/' + Bin + '/real/' + real, test_list + '/' + Bin + '/sideLeft/real/' + real)
			elif real.endswith('001.png'):
				os.rename(test_list + '/' + Bin + '/real/' + real, test_list + '/' + Bin + '/front/real/' + real)
			elif real.endswith('002.png'):
				os.rename(test_list + '/' + Bin + '/real/' + real, test_list + '/' + Bin + '/sideRight/real/' + real)
			elif real.endswith('003.png'):
				os.rename(test_list + '/' + Bin + '/real/' + real, test_list + '/' + Bin + '/back/real/' + real)
			elif real.endswith('004.png'):
				os.rename(test_list + '/' + Bin + '/real/' + real, test_list + '/' + Bin + '/top/real/' + real)