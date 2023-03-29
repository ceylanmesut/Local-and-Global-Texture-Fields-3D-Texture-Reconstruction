import os
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision.utils import save_image
from mesh2tex import geometry
from mesh2tex.training import BaseTrainer
import mesh2tex.utils.FID.feature_l1 as feature_l1
import mesh2tex.utils.SSIM_L1.ssim_l1_score as SSIM


class Trainer(BaseTrainer):
    '''
    Subclass of Basetrainer for defining train_step, eval_step and visualize
    '''
    def __init__(self, model_g, model_d,
                 optimizer_g, optimizer_d,
                 ma_beta=0.99, gp_reg=10.,
                 w_pix=0., w_gan=0., w_vae=0.,
                 experiment='conditional',
                 gan_type='standard',
                 loss_type='L1',
                 multi_gpu=False,
                 **kwargs):

        # Initialize base trainer
        super().__init__(**kwargs)

        # Models and optimizers
        self.model_g = model_g
        self.model_d = model_d

        self.model_g_ma = copy.deepcopy(model_g)

        for p in self.model_g_ma.parameters():
            p.requires_grad = False
        self.model_g_ma.eval()

        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.loss_type = loss_type
        self.experiment = experiment
        # Attributes
        self.gp_reg = gp_reg
        self.ma_beta = ma_beta
        self.gan_type = gan_type
        self.multi_gpu = multi_gpu
        self.w_pix = w_pix
        self.w_vae = w_vae
        self.w_gan = w_gan
        self.pix_loss = w_pix != 0
        self.vae_loss = w_vae != 0
        self.gan_loss = w_gan != 0
        if self.vae_loss and self.pix_loss:
            print('Not possible to combine pix and vae loss')
        # Checkpointer
        if self.gan_loss is True:
            self.checkpoint_io.register_modules(
                model_g=self.model_g, model_d=self.model_d,
                model_g_ma=self.model_g_ma,
                optimizer_g=self.optimizer_g,
                optimizer_d=self.optimizer_d,
            )
        else:
            self.checkpoint_io.register_modules(
                model_g=self.model_g, model_d=self.model_d,
                model_g_ma=self.model_g_ma,
                optimizer_g=self.optimizer_g,
            )

        #print('w_pix: %f w_gan: %f w_vae: %f'
              #% (self.w_pix, self.w_gan, self.w_vae))

    def train_step(self, batch, epoch_it, it):
        '''
        A single training step for the conditional or generative experiment
        Output:
            Losses
        '''
        batch_model0, batch_model1 = batch
        if self.experiment == 'conditional':
            loss_con = self.train_step_cond(batch_model0)
            if self.gan_loss is True:
                loss_d = self.train_step_d(batch_model1)
            else:
                loss_d = 0

            losses = {
                'loss_con': loss_con,
                'loss_d': loss_d,
            }
        
        elif self.experiment == 'generative':
            loss_g = self.train_step_g(batch_model0)
            if self.gan_loss is True:
                loss_d = self.train_step_d(batch_model1)
            else:
                loss_d = 0
            losses = {
                'loss_g': loss_g,
                'loss_d': loss_d,
            }

        return losses

    def train_step_d(self, batch):
        '''
        A single train step of the discriminator
        '''
        model_d = self.model_d
        model_g = self.model_g

        model_d.train()
        model_g.train()

        if self.multi_gpu:
            model_d = nn.DataParallel(model_d)
            model_g = nn.DataParallel(model_g)

        self.optimizer_d.zero_grad()

        # Get data
        depth = batch['2d.depth'].to(self.device)
        img_real = batch['2d.img'].to(self.device)
        cam_K = batch['2d.camera_mat'].to(self.device)
        cam_W = batch['2d.world_mat'].to(self.device)
        mesh_repr = geometry.get_representation(batch, self.device)
        condition = batch['condition'].to(self.device)

        # Loss on real
        img_real.requires_grad_()
        d_real = model_d(img_real, depth, mesh_repr)
        
        dloss_real = self.compute_gan_loss(d_real, 1)
        dloss_real.backward(retain_graph=True)

        # R1 Regularizer
        reg = self.gp_reg * compute_grad2(d_real, img_real).mean()
        reg.backward()

        # Loss on fake
        with torch.no_grad():
            if self.vae_loss is True:
                loss_vae, re, kl, img_fake = model_g.elbo(img_real, depth,
                                                          cam_K, cam_W,
                                                          mesh_repr)
            elif self.gan_loss is True:
                img_fake = model_g(depth, cam_K, cam_W, mesh_repr, condition)

        d_fake = model_d(img_fake, depth, mesh_repr)
       
        dloss_fake = self.compute_gan_loss(d_fake, 0)
        dloss_fake.backward()

        # Gradient step
        self.optimizer_d.step()

        return self.w_gan * (dloss_fake.item() + dloss_real.item())

    def train_step_g(self, batch):
        '''
        A single train step of the generator part of generative model 
        (VAE: Encoder+Decoder and GAN: Generator)
        '''
        model_d = self.model_d
        model_g = self.model_g

        model_d.train()
        model_g.train()

        if self.multi_gpu:
            model_d = nn.DataParallel(model_d)
            model_g = nn.DataParallel(model_g)

        self.optimizer_g.zero_grad()

        # Get data
        depth = batch['2d.depth'].to(self.device)
        img_real = batch['2d.img'].to(self.device)
        cam_K = batch['2d.camera_mat'].to(self.device)
        cam_W = batch['2d.world_mat'].to(self.device)
        mesh_repr = geometry.get_representation(batch, self.device)

        # Loss on fake
        loss_vae = 0
        loss_gan = 0

        # Forward part and loss derivation for given experiment
        if self.vae_loss is True:          
            loss_vae, re, kl, img_fake = model_g.elbo(img_real, depth,
                                                      cam_K, cam_W, 
                                                      mesh_repr)
            if self.gan_loss is True:
                d_fake = model_d(img_fake, depth, mesh_repr)
                loss_gan = self.compute_gan_loss(d_fake, 1)
        elif self.gan_loss is True:
            img_fake = model_g(depth, cam_K, cam_W, mesh_repr)
            d_fake = model_d(img_fake, depth, mesh_repr)
            loss_gan = self.compute_gan_loss(d_fake, 1)

        # weighting of losses (128*128*3=49152)
        loss = self.w_vae * loss_vae + self.w_gan * loss_gan
        loss.backward()

        # Gradient step
        self.optimizer_g.step()

        # Update moving average
        # self.update_moving_average()

        return loss.item()

    def train_step_cond(self, batch):
        '''
        A single train step of the conditional model
        with or w/o generator part of adversarial loss
        '''
        model_d = self.model_d
        model_g = self.model_g

        model_d.train()
        model_g.train()

        if self.multi_gpu:
            model_d = nn.DataParallel(model_d)
            model_g = nn.DataParallel(model_g)

        self.optimizer_g.zero_grad()

        # Get data
        depth = batch['2d.depth'].to(self.device)
        img_real = batch['2d.img'].to(self.device)
        cam_K = batch['2d.camera_mat'].to(self.device)
        cam_W = batch['2d.world_mat'].to(self.device)
        mesh_repr = geometry.get_representation(batch, self.device)
        
        # Berk: I added these new fields.
        condition_image = batch['condition.img'].to(self.device)
        condition_depth = batch['condition.depth'].to(self.device)
        condition_cam_K = batch['condition.camera_mat'].to(self.device)
        condition_cam_W = batch['condition.world_mat'].to(self.device)

        # Loss on fake
        # Berk : Updated model_g
        img_fake = model_g(depth, cam_K, cam_W, mesh_repr, condition_image,
                           condition_depth, condition_cam_K , condition_cam_W)
        loss_pix = 0
        loss_gan = 0

        if self.pix_loss is True:
            loss_pix = self.compute_loss(img_fake, img_real)
        if self.gan_loss is True:
            d_fake = model_d(img_fake, depth, mesh_repr)
            loss_gan = self.compute_gan_loss(d_fake, 1)

        # weighting
        loss = self.w_pix * loss_pix + self.w_gan * loss_gan
        loss.backward()

        # Gradient step
        self.optimizer_g.step()

        # Update moving average
        # self.update_moving_average()

        return loss.item()

    def compute_loss(self, img_fake, img_real):
        '''
        Compute Pixelloss
        '''
        if self.loss_type == 'L2':
            loss = F.mse_loss(img_fake, img_real)
        elif self.loss_type == 'L1':
            loss = F.l1_loss(img_fake, img_real)
        else:
            raise NotImplementedError

        return loss

    def compute_gan_loss(self, d_out, target):
        '''
        Compute GAN loss (standart cross entropy or wasserstein distance)
        !!! Without Regularizer
        '''
        targets = d_out.new_full(size=d_out.size(), fill_value=target)

        if self.gan_type == 'standard':
            loss = F.binary_cross_entropy_with_logits(d_out, targets)
        elif self.gan_type == 'wgan':
            loss = (2*target - 1) * d_out.mean()
        else:
            raise NotImplementedError

        return loss

    def eval_step(self, batch):
        '''
        Evaluation step with L1, SSIM, featl1 metrics
        '''
        depth = batch['2d.depth'].to(self.device)
        img_real = batch['2d.img'].to(self.device)
        cam_K = batch['2d.camera_mat'].to(self.device)
        cam_W = batch['2d.world_mat'].to(self.device)
        mesh_repr = geometry.get_representation(batch, self.device)

        condition_image = batch['condition.img'].to(self.device)
        condition_depth = batch['condition.depth'].to(self.device)
        condition_cam_K = batch['condition.camera_mat'].to(self.device)
        condition_cam_W = batch['condition.world_mat'].to(self.device)
        
        # Get model
        model_g = self.model_g
        model_g.eval()

        if self.multi_gpu:
            model_g = nn.DataParallel(model_g)

        # Predict
        with torch.no_grad():
            img_fake = model_g(depth, cam_K, cam_W, mesh_repr, condition_image,
                           condition_depth, condition_cam_K , condition_cam_W)
        
        # Derive metrics
        loss_val = self.compute_loss(img_fake, img_real)
        ssim, l1 = SSIM.calculate_ssim_l1_given_tensor(img_fake, img_real)
        featl1 = feature_l1.calculate_feature_l1_given_tensors(
            img_fake, img_real, img_real.size(0), True, 2048)

        loss_val_dict = {'loss_val': loss_val.item(), 'SSIM': ssim, 
                         'featl1': featl1}
        return loss_val_dict

    def visualize(self, batch):
        '''
        Visualization step
        '''
        depth = batch['2d.depth'].to(self.device)
        img_real = batch['2d.img'].to(self.device)
        cam_K = batch['2d.camera_mat'].to(self.device)
        cam_W = batch['2d.world_mat'].to(self.device)
        mesh_repr = geometry.get_representation(batch, self.device)


        condition_image = batch['condition.img'].to(self.device)
        condition_depth = batch['condition.depth'].to(self.device)
        condition_cam_K = batch['condition.camera_mat'].to(self.device)
        condition_cam_W = batch['condition.world_mat'].to(self.device)
        
        # determine constants
        batch_size = depth.size(0)
        num_views = depth.size(1)
        assert(num_views == 5)

        mesh_points = mesh_repr['points']
        mesh_normals = mesh_repr['normals']

        # batch loop
        for j in range(batch_size):

            # Gather input information on shape, depth, camera and condition
            geom_repr = {
                'points': mesh_points[j].expand(num_views,
                                                mesh_points.size(1),
                                                mesh_points.size(2)),
                'normals': mesh_normals[j].expand(num_views,
                                                  mesh_normals.size(1),
                                                  mesh_normals.size(2)),
            }

            depth_ = depth[j]

            img_real_ = img_real[j]
            if len(condition_image.size()) == 1:
                condition_ = condition_image[j].expand(num_views)
            else:
                condition_ = condition_image[j].expand(num_views,
                                                 condition_image.size(1),
                                                 condition_image.size(2),
                                                 condition_image.size(3))
                
            if len(condition_depth.size()) == 1:
                condition_depth_ = condition_depth[j].expand(num_views)
            else:
                condition_depth_ = condition_depth[j].expand(num_views,
                                                 condition_depth.size(1),
                                                 condition_depth.size(2),
                                                 condition_depth.size(3))
                
            condition_cam_K_ = condition_cam_K[j]
            condition_cam_W_ = condition_cam_W[j]
                            
            if len(condition_cam_K.size()) == 1:
                condition_cam_K_ = condition_cam_K[j].expand(num_views)
            else:
                condition_cam_K_ = condition_cam_K[j].expand(num_views,
                                                 condition_cam_K.size(1),
                                                 condition_cam_K.size(2))
                
            if len(condition_cam_W.size()) == 1:
                condition_cam_W_ = condition_cam_W[j].expand(num_views)
            else:
                condition_cam_W_ = condition_cam_W[j].expand(num_views,
                                                 condition_cam_W.size(1),
                                                 condition_cam_W.size(2))
                
            cam_K_ = cam_K[j]
            cam_W_ = cam_W[j]


            # save real images
            save_image(img_real_,
                       os.path.join(self.vis_dir, 'real_%i.png' % j))

            # predict fake images and save
            self.model_g.eval()
            with torch.no_grad():
                img_fake = self.model_g(depth_, cam_K_, cam_W_,
                                        geom_repr, condition_, condition_depth_,  condition_cam_K_ , condition_cam_W_)
                
            save_image(img_fake.cpu(),
                       os.path.join(self.vis_dir, 'fake_%i.png' % j))

            # save condition images
            if len(condition_image.size()) != 1:
                save_image(condition_image[j].cpu(),
                           os.path.join(self.vis_dir, 'condition_%i.png' % j))
            else:
                np.savetxt(os.path.join(self.vis_dir, 'condition_%i.txt' % j), 
                           condition_.cpu())

    def update_moving_average(self):
        '''
        Update moving average
        '''
        param_dict_src = dict(self.model_g.named_parameters())
        beta = self.ma_beta
        for p_name, p_tgt in self.model_g_ma.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            with torch.no_grad():
                p_ma = beta * p_tgt + (1. - beta) * p_src
            p_tgt.copy_(p_ma)


def compute_grad2(d_out, x_in):
    '''
    Derive L2-Gradient penalty for regularizing the GAN
    '''
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg
