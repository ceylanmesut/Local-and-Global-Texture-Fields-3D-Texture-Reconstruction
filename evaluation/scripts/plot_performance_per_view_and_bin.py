import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','sans-serif':['Times New Roman']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# The purpose of this script is to take the output of the script evaluate_per_bin.py and plot, for each of the views "sideLeft",
# "sideRight", "front", "back", "top" the FID, SSIM, L1 and featL1 scores over the bins (i.e. as we change the angle of the input image)

# path to the evaluation results
eval_base_path = '/srv/beegfs02/scratch/texture/data/single_view_3d_w_texture/src/texture_project/out/singleview/new_data/'

# # output path
out_dir = '/srv/beegfs02/scratch/texture/data/single_view_3d_w_texture/src/texture_project/grids/'

# create arrays to store the retrieved data
arrayDict = {'front': np.zeros((4, 24)), 'sideLeft': np.zeros((4,24)), 'sideRight': np.zeros((4,24)), 'back': np.zeros((4,24)), 'top': np.zeros((4,24))}

viewList = ['front', 'sideLeft', 'sideRight', 'back', 'top']

# loop through all bins to retrieve the evaluation data
for i in range(24):


	if i<= 17: # ADDED BY MESUT 
		i = i + 6 #Relating 0th bin index to 6th image index. 
	else:
		i=i-18 # ADDED BY MESUT 

	# loop though all views 
	for view in viewList:

		eval_path = eval_base_path + 'car_trained_1July_bin_' + str(i) + '/' + view + '/eval.csv' 
		# open the .csv file and retrieve the data
		with open(eval_path, mode='r') as csv_file:
			csv_reader = csv.DictReader(csv_file)
			for row in csv_reader:

				if i >= 6 and i <=23: # ADDED BY MESUT 
					i = i - 6 #Relating 0th bin index to 6th image index. 
				else:
					i= i+18 # ADDED BY MESUT 

				arrayDict[view][0,i] = row['FID']
				arrayDict[view][1,i] = row['SSIM']
				arrayDict[view][2,i] = row['L1']
				arrayDict[view][3,i] = row['FeatL1']

# plot the results and save as images
x = np.linspace(0, 345, 24)

# loop through the viewList again and make plots per view
for view in viewList:
	data = arrayDict[view]
	FIDarray = data[0, :]
	# print(FIDarray)
	SSIMarray = data[1, :]
	L1array = data[2, :]
	FeatL1array = data[3, :]
	plt.plot(x, FIDarray, label='FID')
	plt.plot(x, SSIMarray, label='SSIM')
	plt.plot(x, L1array, label='L1')
	plt.plot(x, FeatL1array, label='FeatL1')
	plt.legend(loc='upper left')
	plt.yscale('log')
	plt.ylabel('Score per metric')
	plt.xlabel('Angle (deg)')
	plt.grid(True)
	plt.savefig(out_dir + view + '.png') # make as eps later
	plt.clf()

# plot the histograms per bin (only run this once). We have the same conditioned images per view
# I need to loop through the objects in the test.lst and for each object I need to go to the azimuth sorting directory.
# for each file in the azimuth sorting directory, open the file, if it is empty, move on to the next. If not empty, note the
# image number and then go to the metadata of this object id in the other shapenetrendering folder and look at the line of the 
# image number the angle of that image. Append this number to a bin-specific array, and move on to the next bin until all
# bins in all test objects are looped over. Then plot the histograms per bin in the same plot which different colors for each bin.

# test_list = '/srv/beegfs02/scratch/texture/data/single_view_3d_w_texture/3rdparty/texture_fields/shapenet/synthetic_cars_nospecular/test.lst'
# data_path = '/srv/beegfs02/scratch/texture/data/single_view_3d_w_texture/3rdparty/texture_fields/shapenet/synthetic_cars_nospecular/'
# metadata_path = '/srv/beegfs02/scratch/texture/data/ShapeNetRendering/02958343'


# # path to the rendered data which we will print to our .lst file
# render_path = '/srv/beegfs02/scratch/texture/data/single_view_3d_w_texture/3rdparty/texture_fields/shapenet/synthetic_cars_nospecular'
# test_objs = []

# with open(test_list) as testlist:
# 	for line in testlist:
# 		if line.endswith('\n'):
# 			test_objs.append(line[:-1])
# 		else:
# 			test_objs.append(line)

# histogram_data = {}

# # loop over all objects
# for obj in test_objs:
# 	azimuth_path = data_path + obj + '/azimuth_sorting'
# 	# loop over all azimuth bins
# 	bins = os.listdir(azimuth_path)
# 	for Bin in bins:
# 		if Bin[:-4] not in histogram_data.keys():
# 			histogram_data.update({Bin[:-4] : []})
# 		with open(azimuth_path + '/' + Bin, 'r') as f:
# 			line = f.readline()
# 			if line == '':
# 				continue
# 			else:
# 				line = line[:-1]
# 				idx = int(line[-7:-4])
# 				# fetch the angle at this row idx in the metadata of the object
# 				metadata = metadata_path + '/' + obj + '/rendering/rendering_metadata.txt'
# 				metadata = np.loadtxt(metadata)
# 				azimuth_angle = metadata[idx, 0]
# 				# append the angle to the appropriate bin array
# 				histogram_data[Bin[:-4]].append(azimuth_angle)

# # plot the histograms
# histo = []
# for key in histogram_data:
# 	#plt.hist(histogram_data[key], bins = 25)
# 	for val in histogram_data[key]:
# 		histo.append(val)


# plt.hist(histo, bins=1000)
# plt.ylabel("$\#$ of images", fontsize=10)
# plt.xlabel("Angle (deg)", fontsize=10)
# plt.grid(True)
# plt.savefig(out_dir + 'histograms_of_bins_new.png') # make eps later
# histogram_data = np.zeros((24, len(test_objs)))
# print(histogram_data.shape)

# for i, obj_id in enumerate(test_objs):
# 	metadata_path = data_path + '/' + obj_id + '/rendering/rendering_metadata.txt' 
# 	metadata = np.loadtxt(metadata_path)
# 	azimuth = metadata[:, 0]
# 	azimuth.sort()
# 	histogram_data[:, i] = azimuth
# 	# if azimuth[0] > 45:
# 	# 	print(obj_id)


# for k in range(24):
# 	# print(np.amax(histogram_data[0,:]))
# 	# plot histograms of the azimuth angles
# 	#cm = plt.get_cmap('hsv')
# 	n, bins, patches = plt.hist(histogram_data[k,:], bins = 25)
# 	plt.axvline(x=np.mean(histogram_data[k,:]), color='g')
# 	plt.axvline(x=np.median(histogram_data[k,:]), color='k')
# 	#bin_centers = 0.5 * (bins[:-1] + bins[1:])
# 	# col = bin_centers - min(bin_centers)
# 	# col /= max(col)
# 	# for c, p in zip(col, patches):
# 	#     plt.setp(p, 'facecolor', cm(c))

# 	plt.ylabel("$\#$ of objects", fontsize=10)
# 	plt.xlabel("Angle (deg)", fontsize=10)
# 	plt.grid(True)
# 	plt.savefig('../results/conditioned_input_image_aziumuth_histograms/bin' + str(k)+ '.png')
# 	plt.clf()
