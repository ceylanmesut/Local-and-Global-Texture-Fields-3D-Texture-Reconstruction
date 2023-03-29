import os
import numpy as np
import cv2
from skimage.transform import resize # added by Mesut to resize real images into 224x224

# The purpose of this script is to take the predicted images and create a grid of images where each row describes the 
# requested pose and each column describes the pose of the conditioned image

# obj_id = 'cbe2dc469c47bb80425b2c354eccabaf'
# obj_id = 'cc067578ad92517bbe25370c898e25a5'
#obj_id =  'cc32e1fb3be07a942ea8e4c752a397ac'


# Objects required to have grid images.
obj_id = 'cc39c510115f0dadf774728aa9ef18b6' # this is the yellow mini cooper!
#obj_id = 'ccc11d9428b8a61d2abd704290638859' # this is the red car with writing on sides!
#obj_id = 'cdd00143a3e1e33bbecf71e2e014ff6f' #this is the white one "Honda"


#root = '/srv/beegfs02/scratch/texture/data/single_view_3d_w_texture/3rdparty/texture_fields/out/singleview/car_trained'
root = '/srv/beegfs02/scratch/texture/data/single_view_3d_w_texture/src/texture_project/out/singleview/new_data' #up-to-date folder path.




bins = os.listdir(root)

#valid_bins = ['eval_fix_bin_0', 'eval_fix_bin_6', 'eval_fix_bin_12', 'eval_fix_bin_18'] # front, sideRight, back, sideLeft  

# valid bins modified according to new bins.
valid_bins = ['car_trained_1July_bin_6', 'car_trained_1July_bin_12', 'car_trained_1July_bin_18', 'car_trained_1July_bin_0'] 
# bin0:sideleft bin6:front bin12:sideright bin18:back

requested_poses = ['front', 'sideLeft', 'back', 'sideRight', 'top']

image = []

for bin_ in bins:
	if bin_ in valid_bins:
		imtemp = []
		for pose in requested_poses:
			path = root + '/' + bin_ + '/' + pose + '/fake/' + obj_id
			if pose == 'sideLeft':
				path = path + '000.png'
			elif pose == 'front':
				path = path + '001.png'
			elif pose == 'sideRight':
				path = path + '002.png'
			elif pose == 'back':
				path = path + '003.png'
			elif pose == 'top':
				path = path + '004.png'

			im = cv2.imread(path)
			imtemp.append(im)
		col = imtemp[0]
		col = np.concatenate((col, imtemp[1]), axis=0)  
		col = np.concatenate((col, imtemp[2]), axis=0) 
		col = np.concatenate((col, imtemp[3]), axis=0)
		col = np.concatenate((col, imtemp[4]), axis=0) 

		image.append(col)

imageout = image[3] #col 1 - 3
imageout = np.concatenate((imageout, image[0]), axis=1) #col 2 - 
imageout = np.concatenate((imageout, image[2]), axis=1) # col 3 
imageout = np.concatenate((imageout, image[1]), axis=1) # col 4 - 1 

# add the gt images as the left most column 

for bin_ in bins:
	if bin_ in valid_bins:

		#print("REAL PART PRINTING BINS", bin_)
		imtemp = []
		for pose in requested_poses:
			path = root + '/' + bin_ + '/' + pose + '/real/' + obj_id
			if pose == 'sideLeft':
				path = path + '000.png'
			elif pose == 'front':
				path = path + '001.png'
			elif pose == 'sideRight':
				path = path + '002.png'
			elif pose == 'back':
				path = path + '003.png'
			elif pose == 'top':
				path = path + '004.png'


			im = cv2.imread(path)
			# For resizing the real images.
			#image=resize_image(im, 224,224)
			image = cv2.resize(im, (224,224)) #cv2 resizing function used

			imtemp.append(image)
		col = imtemp[0]
		col = np.concatenate((col, imtemp[1]), axis=0)
		col = np.concatenate((col, imtemp[2]), axis=0)
		col = np.concatenate((col, imtemp[3]), axis=0)
		col = np.concatenate((col, imtemp[4]), axis=0)
		# print(col.shape)
		break

imageout = np.concatenate((col, imageout), axis=1)

# add the conditioned images on the top row.

cond_root = '/srv/beegfs02/scratch/texture/data/single_view_3d_w_texture/3rdparty/texture_fields/shapenet/synthetic_cars_nospecular/' + obj_id + '/image_224/'


front = cond_root + '006.png'
back = cond_root + '018.png'
sideLeft = cond_root + '000.png'
sideRight = cond_root + '012.png'


cond = 255*np.ones((224, 224, 3)) # all shapes were 256
cond = np.concatenate((cond, cv2.resize(cv2.imread(front), (224, 224))), axis=1)
cond = np.concatenate((cond, cv2.resize(cv2.imread(sideRight), (224, 224))), axis=1)
cond = np.concatenate((cond, cv2.resize(cv2.imread(back), (224, 224))), axis=1)
cond = np.concatenate((cond, cv2.resize(cv2.imread(sideLeft), (224, 224))), axis=1)

print(cond.shape)
print(imageout.shape)

imageout = np.concatenate((cond, imageout), axis=0)
# print(imageout.shape)

if not cv2.imwrite('/srv/beegfs02/scratch/texture/data/single_view_3d_w_texture/src/texture_project/grids/new.png', imageout):
	
	raise Exception("Could not write image")


