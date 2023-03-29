import os
import numpy as np

# The purpose of this script is to take the filter the input images into different bins afther their azimuth angle.
# The output is 24 txt-files containing, for each row, the path to an input image that fulfills the bin requirement. The bins are:
# 15k+-7.5 for k:{1,...,24}.

# path to test data of texture fields
test_list = '/srv/beegfs02/scratch/texture/data/single_view_3d_w_texture/3rdparty/texture_fields/shapenet/synthetic_cars_nospecular/test.lst'

# path to the metadata
data_path = '/srv/beegfs02/scratch/texture/data/ShapeNetRendering/02958343'

# path to the rendered data which we will print to our .lst file
render_path = '/srv/beegfs02/scratch/texture/data/single_view_3d_w_texture/3rdparty/texture_fields/shapenet/synthetic_cars_nospecular'

test_objs = []

with open(test_list) as testlist:
	for line in testlist:
		if line.endswith('\n'):
			test_objs.append(line[:-1])
		else:
			test_objs.append(line)

# loop over the 24 bins
for i in range(24):
	# create .lst file for the bin
	with open('../config/test_' + str(i), 'w') as f:
		# loop over the test list objects
		for p, obj_id in enumerate(test_objs):
			# for each tests list object, create a folder in the object folder called "azimuth_sorting"
			if not os.path.exists(render_path + '/' + obj_id + '/azimuth_sorting'):
				os.makedirs(render_path + '/' + obj_id + '/azimuth_sorting')
			metadata_path = data_path + '/' + obj_id + '/rendering/rendering_metadata.txt' 
			metadata = np.loadtxt(metadata_path)
			azimuth = metadata[:, 0]
			with open(render_path + '/' + obj_id + '/azimuth_sorting/bin_' + str(i) + '.lst', 'w') as a:
				# check what entries fulfill the bin requirement by looping through the azimuth angles
				for k, angle in enumerate(azimuth):
					if i == 0:
						if angle <= 7.5 or angle > 352.5:
							if k < 10:
								f.write(render_path + '/' + obj_id + '/input_image/00' + str(k) + '.jpg' )
								f.write('\n')
								a.write(render_path + '/' + obj_id + '/input_image/00' + str(k) + '.jpg' )
								a.write('\n')
							else:
								f.write(render_path + '/' + obj_id + '/input_image/0' + str(k) + '.jpg' )
								f.write('\n')
								a.write(render_path + '/' + obj_id + '/input_image/0' + str(k) + '.jpg' )
								a.write('\n')
					else:
						if angle <= 15*i + 7.5 and angle > 15*i - 7.5:
							if k < 10:
								f.write(render_path + '/' + obj_id + '/input_image/00' + str(k) + '.jpg' )
								f.write('\n')
								a.write(render_path + '/' + obj_id + '/input_image/00' + str(k) + '.jpg' )
								a.write('\n')
							else:
								f.write(render_path + '/' + obj_id + '/input_image/0' + str(k) + '.jpg' )
								f.write('\n')
								a.write(render_path + '/' + obj_id + '/input_image/0' + str(k) + '.jpg' )
								a.write('\n')
