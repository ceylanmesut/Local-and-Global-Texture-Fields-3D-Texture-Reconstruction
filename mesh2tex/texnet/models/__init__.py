from PIL import Image 
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import trimesh
import numpy as np
from mesh2tex.texnet.models import (
    image_encoder, decoder, discriminator, vae_encoder
)

encoder_dict = {
    'resnet18': image_encoder.Resnet18,
}

decoder_dict = {
    #'each_layer_c': decoder.DecoderEachLayerC,
    'each_layer_c_larger': decoder.DecoderEachLayerCLarger,
}

discriminator_dict = {
    'resnet_conditional': discriminator.Resnet_Conditional,
}

vae_encoder_dict = {
    'resnet': vae_encoder.Resnet,
}


class TextureNetwork(nn.Module):
    def __init__(self, decoder, geometry_encoder, encoder=None,
                 vae_encoder=None, p0_z=None, white_bg=True):
        super().__init__()
        
        if p0_z is None:
            p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))

        self.decoder = decoder
        self.encoder = encoder
        self.geometry_encoder = geometry_encoder
        self.vae_encoder = vae_encoder
        self.p0_z = p0_z
        self.white_bg = white_bg
        
        
    def save_image(self, image, name):
        image = image.permute(0, 2, 3, 1)
        image = image.data.cpu().numpy()
        image = np.squeeze(image[0, :, :, : ])
        image[image == np.inf] = 0
        image = image - np.min(image)
        #print(np.min(image))        
        #print(np.max(image))

        image = 255* image/np.max(image)
        #print(image.shape)
        im = Image.fromarray(image.astype('uint8'))
        im.save('/home/bekaya/images/' + name + '.png')

    def forward(self, depth, cam_K, cam_W, geometry,
                condition_image=None, condition_depth = None,
                condition_cam_K = None, condition_cam_W = None,
                sample=True):
        """Generate an image .

        Args:
            depth (torch.FloatTensor): tensor of size B x 1 x N x M
                representing depth of at pixels
            cam_K (torch.FloatTensor): tensor of size B x 3 x 4 representing
                camera projectin matrix
            cam_W (torch.FloatTensor): tensor of size B x 3 x 4 representing
                camera world matrix
            geometry (dict): representation of geometry
            condition
            sample (Boolean): wether to sample latent code or take MAP
        Returns:
            img (torch.FloatTensor): tensor of size B x 3 x N x M representing
                output image
        """


        
        batch_size, _, N, M = depth.size()
        assert(depth.size(1) == 1)
        assert(cam_K.size() == (batch_size, 3, 4))
        assert(cam_W.size() == (batch_size, 3, 4))

        loc3d, mask = self.depth_map_to_3d(depth, cam_K, cam_W)
        geom_descr = self.encode_geometry(geometry)
        #print('Geometry encoded')

        if self.encoder is not None:
            # Use the condition image to get global feature and local features.
            gf, lf = self.encode(condition_image)  

            # Transfer features to gpu
            gf = gf.cuda()    
            lf = lf.cuda() 

            # Warp the depth map from CAD model using condition image camera parameters
            projected_depth, pixel_mapping = self.projection(loc3d, condition_cam_K, condition_cam_W)
        
            # Create an occlusion map 
            occlusion_map = self.occlusion_module(condition_depth, projected_depth)
            
            # Filter out occluded parts in local features block
            lf = self.filter_local_features(lf, occlusion_map)
            
        # Reshape 3d points and local features 
        lf = self.arrange_local_features(lf, pixel_mapping)
            
        loc3d = loc3d.view(batch_size, 3, N * M)
        #lf = lf.view(batch_size, 256, N*M)
        
        # Call modified Texture-fields decoder
        x = self.decode(loc3d, geom_descr, gf, lf)  
        x = x.view(batch_size, 3, N, M)
    
        # Adjust background 
        if self.white_bg is False:
            x_bg = torch.zeros_like(x)
        else:
            x_bg = torch.ones_like(x)

        # Adjust the background
        img = (mask * x).permute(0, 1, 3, 2) + (1 - mask.permute(0, 1, 3, 2)) * x_bg
   
        return img
    
    
    def filter_local_features(self, lf, occlusion_map):
        # Multiply occluded local features with zero 
        lf = lf * occlusion_map.expand_as(lf)
        return lf
        
        
        
    def occlusion_module(self, input_depth, projected_depth, margin = 0.02):
        '''
        Parameters
        ----------
        input_depth : condition depth map 
        projected_depth : projected version of loc3d, warped depth from 3d model 
        margin : The default is 0.05.
        Returns
        -------
        occlusion map: 1 if local feature is used, 0 otherwise
        '''
        mask = 1 - (projected_depth == 0)
        # if projected depth is larger, it means it is occluded
        # otherwise it is nor occluded, so we put 1 to use local feature
        occlusion_map = torch.abs(projected_depth - input_depth) < margin
        occlusion_map = occlusion_map * mask
        return occlusion_map.type(torch.FloatTensor).cuda()
 
           
    def projection(self, loc3d, cam_K, cam_W):
        # This function uses camera parameters to project 3d points
        # It simply multuplies the points with camera matrices and gets depth maps       

        device = loc3d.device
        
        # Add dimension to camera parameters if batch size is 1
        if len(cam_K.shape) == 2:
            cam_K = cam_K.unsqueeze(0)
        if len(cam_W.shape) == 2:
            cam_W = cam_W.unsqueeze(0)

        batch_size, _, N, M = loc3d.size()

        # Reshape point cloud
        loc3d = loc3d.reshape(batch_size, 3, N * M)
        
        # Create ones matrix and concatenate with point data
        ones_matrix = torch.ones([batch_size, 1, N * M]).cuda()
        loc3d = torch.cat((loc3d, ones_matrix), dim=1)
    
    
        zero_one_row = torch.tensor([[0., 0., 0., 1.]]).cuda()
        zero_one_row = zero_one_row.expand(batch_size, 1, 4).cuda()
        # add row to world mat
        cam_W = torch.cat((cam_W, zero_one_row), dim=1)
    
        # Multiply with camera parameters. 
        pixel_mapping = torch.bmm(cam_K,  torch.bmm(cam_W, loc3d))
        
        c =  pixel_mapping[:, 2, :]/(112)
        pixel_mapping[:, 0, :] = torch.div(pixel_mapping[:, 0, :], c) 
        pixel_mapping[:, 1, :] = torch.div(pixel_mapping[:, 1, :], c) 
        pixel_mapping[:, 2, :] = torch.div(pixel_mapping[:, 2, :], c)
    
        # Create converted depth tensor
        converted_depth = torch.zeros([batch_size, 1, N*M]).cuda()
        
        # For each sample, register points to the depth map
        # We also keep a pixel_mapping data to keep track which point is projected to which location
        for batch in range(batch_size):
            #for i in range (N*M):
            #if torch.max(pixel_mapping[:, :, i]) < N-1:  
     
                index_h = N - torch.round(pixel_mapping[batch, 1, :]).type(torch.LongTensor)
                index_w = torch.round(pixel_mapping[batch, 0, :]).type(torch.LongTensor)
                index_h = torch.clamp(index_h, 0, N-1)
                index_w = torch.clamp(index_w, 0, M-1)
                converted_depth[batch, 0, index_h *M + index_w] =  c[batch, :] * 2  *112/137
    
        converted_depth = torch.reshape(converted_depth, [batch_size, 1, N, M])
    
        return converted_depth.cuda(), pixel_mapping.cuda()
     
    
    def arrange_local_features(self, lf, pixel_mapping):
        '''
        This function is respoinsible for assigning local features to the correspoinding 
        3d point in the input depth coming from 3D model.
        
        lf: local features (bs x 256 x 224 x 224)
        pixel_mapping : (bs x 3 x h*w)
        '''
        bs, c, h, w = lf.shape
        
        # Get row and clumn information from pixel_mapping data
        index_h = torch.clamp(h-torch.round(pixel_mapping[:, 1, :]), 0, h-1)
        index_w = torch.clamp(torch.round(pixel_mapping[:, 0, :]), 0, w-1)

        # Get the pixel number
        pixel_index = index_h * w + index_w
        
        lf = torch.reshape(lf, [bs, c, h*w])
        
        arranged_local_features = torch.zeros(lf.shape).cuda()

        # Arrange(permute) local features for each sample
        for batch in range(bs):
            # Create a permutation list 
            permutation_list = pixel_index[batch, :].squeeze().type(torch.LongTensor)
            # Arrange the placement of the features and add them
            arranged_local_features[batch, :, :] = arranged_local_features[batch, :, :] + lf[batch, :, permutation_list]        

        arranged_local_features = torch.reshape(arranged_local_features, [bs, c, h, w])
        return arranged_local_features
    

    def load_mesh2facecenter(in_path):
        mesh = trimesh.load(in_path, process=False)
        faces_center = mesh.triangles_center
        return mesh, faces_center

    def elbo(self, image_real, depth, cam_K, cam_W, geometry):
        batch_size, _, N, M = depth.size()

        assert(depth.size(1) == 1)
        assert(cam_K.size() == (batch_size, 3, 4))
        assert(cam_W.size() == (batch_size, 3, 4))

        loc3d, mask = self.depth_map_to_3d(depth, cam_K, cam_W)
        geom_descr = self.encode_geometry(geometry)

        q_z = self.infer_z(image_real, geom_descr)
        z = q_z.rsample()

        loc3d = loc3d.view(batch_size, 3, N * M)
        x = self.decode(loc3d, geom_descr, z)
        x = x.view(batch_size, 3, N, M)

        if self.white_bg is False:
            x_bg = torch.zeros_like(x)
        else:
            x_bg = torch.ones_like(x)

        image_fake = (mask * x).permute(0, 1, 3, 2) + (1 - mask.permute(0, 1, 3, 2)) * x_bg

        recon_loss = F.mse_loss(image_fake, image_real).sum(dim=-1)
        kl = dist.kl_divergence(q_z, self.p0_z).sum(dim=-1)
        elbo = recon_loss.mean() + kl.mean()/float(N*M*3)
        return elbo, recon_loss.mean(), kl.mean()/float(N*M*3), image_fake

    def encode(self, cond):
        """Encode mesh using sampled 3D location on the mesh.

        Args:
            input_image (torch.FloatTensor): tensor of size B x 3 x N x M
                input image

        Returns:
            c (torch.FloatTensor): tensor of size B x C with encoding of
                the input image
        """
        gf, lf = self.encoder(cond)
        return gf, lf

    def encode_geometry(self, geometry):
        """Encode mesh using sampled 3D location on the mesh.

        Args:
            geometry (dict): representation of teometry
        Returns:
            geom_descr (dict): geometry discriptor

        """
        geom_descr = self.geometry_encoder(geometry)
        return geom_descr

    def decode(self, loc3d, c, gf, lf):
        """Decode image from 3D locations, conditional encoding and latent
        encoding.

        Args:
            loc3d (torch.FloatTensor): tensor of size B x 3 x K
                with 3D locations of the query
            c (torch.FloatTensor): tensor of size B x C with the encoding of
                the 3D meshes
            z (torch.FloatTensor): tensor of size B x Z with latent codes

        Returns:
            rgb (torch.FloatTensor): tensor of size B x 3 x N representing
                color at given 3d locations
        """
        rgb = self.decoder(loc3d, c, gf, lf)
        return rgb

    def depth_map_to_3d(self, depth, cam_K, cam_W):
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
        depth = -depth .permute(0, 1, 3, 2)
        
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

    def get_z_from_prior(self, size=torch.Size([]), sample=True):
        """Draw latent code z from prior either using sampling or
        using the MAP.

        Args:
            size (torch.Size): size of sample to draw.
            sample (Boolean): wether to sample or to use the MAP

        Return:
            z (torch.FloatTensor): tensor of shape *size x Z representing
                the latent code
        """
        if sample:
            z = self.p0_z.sample(size)
        else:
            z = self.p0_z.mean
            z = z.expand(*size, *z.size())

        return z

    def infer_z(self, image, c, **kwargs):
        if self.vae_encoder is not None:
            mean_z, logstd_z = self.vae_encoder(image, c, **kwargs)
        else:
            batch_size = image.size(0)
            mean_z = torch.empty(batch_size, 0).to(self._device)
            logstd_z = torch.empty(batch_size, 0).to(self._device)

        q_z = dist.Normal(mean_z, torch.exp(logstd_z))
        return q_z

    def infer_z_transfer(self, image, c, **kwargs):
        if self.vae_encoder is not None:
            mean_z, logstd_z = self.vae_encoder(image, c, **kwargs)
        else:
            batch_size = image.size(0)
            mean_z = torch.empty(batch_size, 0).to(self._device)
        return mean_z
