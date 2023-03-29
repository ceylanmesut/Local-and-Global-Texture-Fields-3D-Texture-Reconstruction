# Depth test
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import trimesh
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import  resize, downscale_local_mean
from numpy import inf

def resize_image(img, height, width):

    if img.ndim == 3:
        h, w, c = img.shape
        img2 = np.zeros([height, width, c])
        for k in range(c):
            img2[:, :, k] = resize(img[:, :, k], (height, width), anti_aliasing=True)
    else:
        h, w = img.shape
        img2 = resize(img, (height, width), anti_aliasing=True)

    return img2




def projection(loc3d, cam_K, cam_W):
    '''
    Berk: I have written this code to map loc3d from shape branch to depth
    loc3d : tensor of size B x 3 x N x M
    cam_K (torch.FloatTensor): tensor of size B x 3 x 4 representing camera matrices
    cam_W (torch.FloatTensor): tensor of size B x 3 x 4 representing world matrices
    '''
    device = loc3d.device

    batch_size, _, N, M = loc3d.size()
    loc3d = loc3d.reshape(batch_size, 3, N * M)
    
    ones_matrix = torch.ones([batch_size, 1, N * M])
    
    loc3d = torch.cat((loc3d, ones_matrix), dim=1)


    zero_one_row = torch.tensor([[0., 0., 0., 1.]])
    zero_one_row = zero_one_row.expand(batch_size, 1, 4).to(device)

    # add row to world mat
    cam_W = torch.cat((cam_W, zero_one_row), dim=1)


    pixel_mapping = torch.bmm(cam_K,  torch.bmm(cam_W, loc3d))
    
    c =  pixel_mapping[:, 2, :]/(112)
    
    
    pixel_mapping[:, 0, :] = torch.div(pixel_mapping[:, 0, :], c) #*137
    pixel_mapping[:, 1, :] = torch.div(pixel_mapping[:, 1, :], c) #*137
    pixel_mapping[:, 2, :] = torch.div(pixel_mapping[:, 2, :], c)

    
    out_mask =torch.zeros([batch_size, 1, N, M])
    converted_depth = torch.zeros([batch_size, 1, N*M])
    
    for i in range (N*M):
        if torch.max(pixel_mapping[:, :, i]) < N-1:  
            
            index_h = N - torch.round(pixel_mapping[:, 1, i]).type(torch.LongTensor)
            index_w = torch.round(pixel_mapping[:, 0, i]).type(torch.LongTensor)
            #print(index_w)
            converted_depth[:, :, index_h *M + index_w] =  c[:, i] * 2  *137/112
    #converted_depth = c * 2 
    
    #print(torch.mean(converted_depth))

    converted_depth = torch.reshape(converted_depth, [batch_size, 1, N, M])
   # out_mask = out_mask.reshape(batch_size, 1, N, M)
    #out_mask = out_mask .permute(0, 1, 2, 3)

    return converted_depth

def depth_map_to_3d( depth, cam_K, cam_W):
        """Derive 3D locations of each pixel of a depth map.

        Args:
            depth (torch.FloatTensor): tensor of size B x 1 x N x M
                with depth at every pixel
            cam_K (torch.FloatTensor): tensor of size B x 3 x 4 representing
                camera matrices
            cam_W (torch.FloatTensor): tensor of size B x 3 x 4 representing
                world matrices
        Returns:
            loc3d (torch.FloatTensor): tensor of size B x 3 x N x M
                representing color at given 3d locations
            mask (torch.FloatTensor):  tensor of size B x 1 x N x M with
                a binary mask if the given pixel is present or not
        """
       
        assert(depth.size(1) == 1)
        batch_size, _, N, M = depth.size()
        device = depth.device
        # Turn depth around. This also avoids problems with inplace operations
        depth = - depth .permute(0, 1, 3, 2)
        
        zero_one_row = torch.tensor([[0., 0., 0., 1.]])
        zero_one_row = zero_one_row.expand(batch_size, 1, 4).to(device)

        # add row to world mat
        cam_W = torch.cat((cam_W, zero_one_row), dim=1)

        # clean depth image for mask
        mask = (depth.abs() != float("Inf")).float()
        depth[depth == float("Inf")] = 0
        depth[depth == -1*float("Inf")] = 0

        # 4d array to 2d array k=N*M
        d = depth.reshape(batch_size, 1, N * M)

        # create pixel location tensor
        px, py = torch.meshgrid([torch.arange(0, N), torch.arange(0, M)])
        px, py = px.to(device), py.to(device)

        p = torch.cat((
            px.expand(batch_size, 1, px.size(0), px.size(1)), 
            (M - py).expand(batch_size, 1, py.size(0), py.size(1))
        ), dim=1)
        p = p.reshape(batch_size, 2, py.size(0) * py.size(1))
        p = (p.float() / M * 2)      
        
        # create terms of mapping equation x = P^-1 * d*(qp - b)
        P = cam_K[:, :2, :2].float().to(device)    
        q = cam_K[:, 2:3, 2:3].float().to(device)   
        b = cam_K[:, :2, 2:3].expand(batch_size, 2, d.size(2)).to(device)
        Inv_P = torch.inverse(P).to(device)   

        rightside = (p.float() * q.float() - b.float()) * d.float()
        x_xy = torch.bmm(Inv_P, rightside)
        
        # add depth and ones to location in world coord system
        x_world = torch.cat((x_xy, d, torch.ones_like(d)), dim=1)

        # derive loactoion in object coord via loc3d = W^-1 * x_world
        Inv_W = torch.inverse(cam_W)
        loc3d = torch.bmm(
            Inv_W.expand(batch_size, 4, 4),
            x_world
        ).reshape(batch_size, 4, N, M)

        loc3d = loc3d[:, :3].to(device)
        mask = mask.to(device)
        return loc3d, mask
    
    
idx_img = 2

depth = imageio.imread('C:/Users/41782/Desktop/103402871ed03ed117a54d13fb550a39/depth/002.exr')
depth = np.asarray(depth)
depth = resize_image(depth, 224, 224)
depth = np.squeeze(depth[:,:,1])

depth = np.expand_dims(depth,  0)
depth = np.expand_dims(depth,  0)

camera_file = 'C:/Users/41782/Desktop/103402871ed03ed117a54d13fb550a39/depth/cameras.npz'
camera_dict = np.load(camera_file)
cam_W = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
cam_K = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)

cam_W = np.expand_dims(cam_W,  0)
cam_K = np.expand_dims(cam_K,  0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

depth =torch.tensor(depth, device=device).float()
cam_W =torch.tensor(cam_W, device=device).float()
cam_K =torch.tensor(cam_K, device=device).float()


loc3d, mask = depth_map_to_3d(depth, cam_K, cam_W)

######################

idx_img = 3

camera_file = 'C:/Users/41782/Desktop/103402871ed03ed117a54d13fb550a39/depth_224/cameras.npz'
camera_dict = np.load(camera_file)
cam_W = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
cam_K = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)

cam_W = np.expand_dims(cam_W,  0)
cam_K = np.expand_dims(cam_K,  0)

cam_W =torch.tensor(cam_W, device=device).float()
cam_K =torch.tensor(cam_K, device=device).float()
converted_depth =  projection( loc3d, cam_K, cam_W)
    

depth = imageio.imread('C:/Users/41782/Desktop/103402871ed03ed117a54d13fb550a39/depth_224/00'+ str(idx_img)+ '0001.exr')
depth = np.asarray(depth)
depth = np.squeeze(depth[:,:,1])
#print (np.mean(depth[depth !=  inf]))


imgplot = plt.imshow(np.array(converted_depth.squeeze().squeeze()))
#imgplot = plt.imshow(np.array(depth))
#imgplot = plt.imshow(out_mask.squeeze())
