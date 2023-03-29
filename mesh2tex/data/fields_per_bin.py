import os
import glob
import random
import numpy as np
import trimesh
import imageio
from mesh2tex.data.core import Field


# Make sure loading xlr works
imageio.plugins.freeimage.download()


# Basic index field
class IndexField(Field):
    def load(self, model_path, idx):
        return idx

    def check_complete(self, files):
        return True


class MeshField(Field):
    def __init__(self, folder_name, transform=None):
        self.folder_name = folder_name
        self.transform = transform

    def load(self, model_path, idx):
        folder_path = os.path.join(model_path, self.folder_name)
        file_path = os.path.join(folder_path, 'model.off')
        mesh = trimesh.load(file_path, process=False)
        if self.transform is not None:
            mesh = self.transform(mesh)

        data = {
            'vertices': np.array(mesh.vertices),
            'faces': np.array(mesh.faces),
        }

        return data

    def check_complete(self, files):
        complete = (self.folder_name in files)
        return complete


# Added by Mesut 
# Image and Depth field
# It is created to import condition image and depth simultaneously
class Image_and_Depth_Field(Field):
    
    def __init__(self, folder_name_img, folder_name_depth, transform_img=None, transform_depth=None,
                 extension_img='png', extension_depth='exr', random_view=True,
                 with_camera=False,
                 imageio_kwargs=dict()):
        
        #print('New image and depth dataloader is used for conditinonal block.=====================================')
        self.folder_name_img = folder_name_img
        self.folder_name_depth = folder_name_depth
        self.transform_depth = transform_depth
        self.transform_img = transform_img
        self.extension_img = extension_img
        self.extension_depth = extension_depth
        self.random_view = random_view
        self.with_camera = with_camera
        self.imageio_kwargs = dict()

    def load(self, model_path, idx, binIdx): # binIdx added
        folder_img = os.path.join(model_path, self.folder_name_img)
        files_img = glob.glob(os.path.join(folder_img, '*.%s' % self.extension_img))
        files_img.sort()
        folder_depth = os.path.join(model_path, self.folder_name_depth)
        files_depth = glob.glob(os.path.join(folder_depth, '*.%s' % self.extension_depth))
        files_depth.sort()
        if self.random_view:
            idx_img = random.randint(0, len(files_img)-1)
        else:
            #idx_img = 0
            idx_img = binIdx # added by Mesut

        #     # below added by Mesut
        #     bin_path = model_path + '/azimuth_sorting/bin_' + str(binIdx) + '.lst' # Line Added by ERIK SANDSTROEM
        #     #print("BIN PATH===============================", bin_path)
        #     with open(bin_path, 'r') as f:# Line Added by ERIK SANDSTROEM
        #         line = f.readline()# Line Added by ERIK SANDSTROEM
        #         #print(line[:-1])
        #         if line:# Line Added by ERIK SANDSTROEM
        #             #print('selects image in bin')
        #             filename = line[:-1]# Line Added by ERIK SANDSTROEM
        #         else:# Line Added by ERIK SANDSTROEM
        #             #print('Skip object id for this bin')
        #             #print('selects random image bc image in bin not found')
        #             #filename = files[idx_img]
        #             #print('Will return none for obj id: ', model_path.split('/')[-1])# Line Added by ERIK SANDSTROEM
        #             return# Line Added by ERIK SANDSTROEM
        #             #filename = files[idx_img]
        # #print('conditioned image: ', filename) # ADDED BY ERIK SANDSTROEM
        

    
        camera_file = os.path.join(folder_depth, 'cameras.npz')
        camera_dict = np.load(camera_file)

        
        filename_img = files_img[idx_img] # Commented by Mesut
        filename_depth = files_depth[idx_img] # Not sure check again
        

        image = imageio.imread(filename_img, **self.imageio_kwargs)
        #image = imageio.imread(filename, **self.imageio_kwargs)
        image = np.asarray(image)

        # Added by Mesut
        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)
            image = np.concatenate([image, image, image], axis=2) 
        
        if image.shape[2] == 4:
            image = image[:,:,:3]

        depth = imageio.imread(filename_depth, **self.imageio_kwargs)

        depth = np.asarray(depth)
        
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255
        else:
            image = image.astype(np.float32)

        if self.transform_img is not None:
            image = self.transform_img(image)

        if self.transform_depth is not None:
            depth = self.transform_depth(depth)

        image = image.transpose(2, 0, 1)
        depth = depth.transpose(2, 0, 1)

        camera_file = os.path.join(folder_depth, 'cameras.npz')
        camera_dict = np.load(camera_file)
        Rt = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
        K = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)

        data = {
            'img': image,
            'depth': depth
            }

        data['world_mat'] = Rt
        data['camera_mat'] = K
    
        return data



# Image field
class ImagesField(Field):
    def __init__(self, folder_name, transform=None,
                 extension='jpg', random_view=True,
                 with_camera=False,
                 imageio_kwargs=dict()):
        self.folder_name = folder_name
        self.transform = transform
        self.extension = extension
        self.random_view = random_view
        self.with_camera = with_camera
        self.imageio_kwargs = dict()

    def load(self, model_path, idx, binIdx):
        folder = os.path.join(model_path, self.folder_name)
        files = glob.glob(os.path.join(folder, '*.%s' % self.extension))
        files.sort()
        #print(model_path) # /srv/beegfs02/scratch/texture/data/single_view_3d_w_texture/3rdparty/texture_fields/shapenet/synthetic_cars_nospecular/cc6489afc2f925823505e48e4fe20c28
        #print(self.folder_name) input_image
        #print(files) # lists the full path names (24) in the input_image folder

        #print("WE ARE IN IMAGES FIELD in FIELDS PER BIN")
        if self.random_view:
            idx_img = random.randint(0, len(files)-1)
        else:
            idx_img = binIdx

            # ADDED BY MESUT-WRITTEN BY ERIK
            # check if the current object ID has a valid input image for the given bin
            bin_path = model_path + '/azimuth_sorting/bin_' + str(binIdx) + '.lst' # Line Added by ERIK SANDSTROEM
            with open(bin_path, 'r') as f:# Line Added by ERIK SANDSTROEM
                line = f.readline()# Line Added by ERIK SANDSTROEM
                #print(line[:-1])
                if line:# Line Added by ERIK SANDSTROEM
                    #print('selects image in bin')
                    filename = line[:-1]# Line Added by ERIK SANDSTROEM
                else:# Line Added by ERIK SANDSTROEM
                    #print('Skip object id for this bin')
                    #print('selects random image bc image in bin not found')
                    #filename = files[idx_img]
                    print('Will return none for obj id: ', model_path.split('/')[-1])# Line Added by ERIK SANDSTROEM
                    return# Line Added by ERIK SANDSTROEM
                    #filename = files[idx_img]
        #print('conditioned image: ', filename) # ADDED BY ERIK SANDSTROEM

        image = imageio.imread(filename, **self.imageio_kwargs)
        image = np.asarray(image)

        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)
            image = np.concatenate([image, image, image], axis=2)  

        if image.shape[2] == 4:
            image = image[:, :, :3]

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255
        else:
            image = image.astype(np.float32)

        if self.transform is not None:
            image = self.transform(image)
        image = image.transpose(2, 0, 1)
        data = {
            None: image
        }

        if self.with_camera:
            camera_file = os.path.join(folder, 'cameras.npz')
            camera_dict = np.load(camera_file)
            Rt = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
            K = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)
            data['world_mat'] = Rt
            data['camera_mat'] = K

        return data

    def check_complete(self, files):
        complete = (self.folder_name in files)
        # TODO: check camera
        return complete


# 3D Fields
class PointCloudField(Field):
    def __init__(self, file_name, transform=None, with_transforms=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms

    def load(self, model_path, idx):
        file_path = os.path.join(model_path, self.file_name)

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        data = {
            None: points.T,
            'normals': normals.T,
        }

        if self.with_transforms:
            data['loc'] = pointcloud_dict['loc'].astype(np.float32)
            data['scale'] = pointcloud_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        complete = (self.file_name in files)
        return complete


class DepthImageVisualizeField(Field):
    def __init__(self, folder_name_img, folder_name_depth, transform_img=None, transform_depth=None,
                 extension_img='jpg', extension_depth='exr', random_view=True,
                 with_camera=False,
                 imageio_kwargs=dict()):
        self.folder_name_img = folder_name_img
        self.folder_name_depth = folder_name_depth
        self.transform_depth = transform_depth
        self.transform_img = transform_img
        self.extension_img = extension_img
        self.extension_depth = extension_depth
        self.random_view = random_view
        self.with_camera = with_camera
        self.imageio_kwargs = dict()

    def load(self, model_path, idx):
        folder_img = os.path.join(model_path, self.folder_name_img)
        files_img = glob.glob(os.path.join(folder_img, '*.%s' % self.extension_img))
        files_img.sort()
        folder_depth = os.path.join(model_path, self.folder_name_depth)
        files_depth = glob.glob(os.path.join(folder_depth, '*.%s' % self.extension_depth))
        files_depth.sort()
        if self.random_view:
            idx_img = random.randint(0, len(files_img)-1)
        else:
            idx_img = 0
        
        image_all = []
        depth_all = []
        Rt = []
        K = []
        camera_file = os.path.join(folder_depth, 'cameras.npz')
        camera_dict = np.load(camera_file)

        for i in range(len(files_img)):
            filename_img = files_img[i]
            filename_depth = files_depth[i]

            image = imageio.imread(filename_img, **self.imageio_kwargs)
            image = np.asarray(image)
            
            if image.shape[2] == 4:
                image = image[:,:,:3]

            depth = imageio.imread(filename_depth, **self.imageio_kwargs)

            depth = np.asarray(depth)
            
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255
            else:
                image = image.astype(np.float32)

            if self.transform_img is not None:
                image = self.transform_img(image)

            if self.transform_depth is not None:
                depth = self.transform_depth(depth)

            image = image.transpose(2, 0, 1)
            depth = depth.transpose(2, 0, 1)
            image_all.append(image)
            depth_all.append(depth)

            camera_file = os.path.join(folder_depth, 'cameras.npz')
            camera_dict = np.load(camera_file)
            Rt.append(camera_dict['world_mat_%d' % i].astype(np.float32))
            K.append(camera_dict['camera_mat_%d' % i].astype(np.float32))

        data = {
            'img': np.stack(image_all),
            'depth': np.stack(depth_all)
            }

        data['world_mat'] = np.stack(Rt)
        data['camera_mat'] = np.stack(K)

        return data


# Image field
class DepthImageField(Field):
    def __init__(self, folder_name_img, folder_name_depth, transform_img=None, transform_depth=None,
                 extension_img='jpg', extension_depth='exr', random_view=True,
                 with_camera=False,
                 imageio_kwargs=dict()):
        self.folder_name_img = folder_name_img
        self.folder_name_depth = folder_name_depth
        self.transform_depth = transform_depth
        self.transform_img = transform_img
        self.extension_img = extension_img
        self.extension_depth = extension_depth
        self.random_view = random_view
        self.with_camera = with_camera
        self.imageio_kwargs = dict()

    def load(self, model_path, idx):
        folder_img = os.path.join(model_path, self.folder_name_img)
        files_img = glob.glob(os.path.join(folder_img, '*.%s' % self.extension_img))
        files_img.sort()
        folder_depth = os.path.join(model_path, self.folder_name_depth)
        files_depth = glob.glob(os.path.join(folder_depth, '*.%s' % self.extension_depth))
        files_depth.sort()
        if self.random_view:
            idx_img = random.randint(0, len(files_img)-1)
        else:
            idx_img = 0

        filename_img = files_img[idx_img]
        filename_depth = files_depth[idx_img]

        image = imageio.imread(filename_img, **self.imageio_kwargs)
        image = np.asarray(image)
        if image.shape[2] == 4:
            image = image[:,:,:3]

        depth = imageio.imread(filename_depth, **self.imageio_kwargs)

        depth = np.asarray(depth)
        
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255
        else:
            image = image.astype(np.float32)

        if self.transform_img is not None:
            image = self.transform_img(image)

        if self.transform_depth is not None:
            depth = self.transform_depth(depth)

        image = image.transpose(2, 0, 1)
        #TODO adapt depth transpose
        depth = depth.transpose(2, 0, 1)

        data = {
            'img': image,
            'depth': depth
            }

        if self.with_camera:
            camera_file = os.path.join(folder_depth, 'cameras.npz')
            camera_dict = np.load(camera_file)
            Rt = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
            K = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)
            data['world_mat'] = Rt
            data['camera_mat'] = K

        return data

    def check_complete(self, files):
        complete = (self.folder_name_img in files)
        # TODO: check camera
        return complete
