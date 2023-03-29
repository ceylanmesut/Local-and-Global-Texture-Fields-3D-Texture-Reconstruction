import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','sans-serif':['Times New Roman']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# The purpose of this script is to take the test objects of texture fields and sort the input images (24 pc) according to their
# azimuth angle which is in the metadata file. Since the azimuth angle varies between objects we need to check the variation within
# each bin when the azimuth angles are sorted. Therefore, this script plots the histograms so that we can establish if our approach is 
# still valid.

# path to test data of texture fields
test_list = '/srv/beegfs02/scratch/texture/data/single_view_3d_w_texture/3rdparty/texture_fields/shapenet/synthetic_cars_nospecular/test.lst'

# path to the rendered data (where we also have the metadata stored)
data_path = '/srv/beegfs02/scratch/texture/data/ShapeNetRendering/02958343'

test_objs = []

with open(test_list) as testlist:
	for line in testlist:
		if line.endswith('\n'):
			test_objs.append(line[:-1])
		else:
			test_objs.append(line)

histogram_data = np.zeros((24, len(test_objs)))
print(histogram_data.shape)

for i, obj_id in enumerate(test_objs):
	metadata_path = data_path + '/' + obj_id + '/rendering/rendering_metadata.txt' 
	metadata = np.loadtxt(metadata_path)
	azimuth = metadata[:, 0]
	azimuth.sort()
	histogram_data[:, i] = azimuth
	# if azimuth[0] > 45:
	# 	print(obj_id)


for k in range(24):
	# print(np.amax(histogram_data[0,:]))
	# plot histograms of the azimuth angles
	#cm = plt.get_cmap('hsv')
	n, bins, patches = plt.hist(histogram_data[k,:], bins = 25)
	plt.axvline(x=np.mean(histogram_data[k,:]), color='g')
	plt.axvline(x=np.median(histogram_data[k,:]), color='k')
	#bin_centers = 0.5 * (bins[:-1] + bins[1:])
	# col = bin_centers - min(bin_centers)
	# col /= max(col)
	# for c, p in zip(col, patches):
	#     plt.setp(p, 'facecolor', cm(c))

	plt.ylabel("$\#$ of objects", fontsize=10)
	plt.xlabel("Angle (deg)", fontsize=10)
	plt.grid(True)
	plt.savefig('../results/conditioned_input_image_aziumuth_histograms/bin' + str(k)+ '.png')
	plt.clf()
