from pytorch_lightning.callbacks import Callback
import torchvision
import torch

import matplotlib.pyplot as plt

class MocoImageLogger(Callback):
    def __init__(self, num_images=12, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            (img1, img2), _ = batch
            
            max_num_image = min(img1.shape[0], self.num_images)
            grid_img1 = torchvision.utils.make_grid(img1[0:max_num_image])
            trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
            
            grid_img2 = torchvision.utils.make_grid(img2[0:max_num_image])
            trainer.logger.experiment.add_image('img2', grid_img2.cpu().numpy(), 0)
            with torch.no_grad():
                logits, labels, k = pl_module(img1, img2, pl_module.queue)

            for idx, logit in enumerate(logits[0:max_num_image]):
                trainer.logger.experiment.add_scalar('logits', torch.argmax(logit).cpu().numpy(), idx)


class SimCLRImageLogger(Callback):
    def __init__(self, num_images=12, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            (img1, img2), _ = batch
            
            max_num_image = min(img1.shape[0], self.num_images)
            grid_img1 = torchvision.utils.make_grid(img1[0:max_num_image])
            trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
            
            grid_img2 = torchvision.utils.make_grid(img2[0:max_num_image])
            trainer.logger.experiment.add_image('img2', grid_img2.cpu().numpy(), 0)

class SimImageLogger(Callback):
    def __init__(self, num_images=18, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            img1, img2 = batch
            
            with torch.no_grad():
                img2 = pl_module.noise_transform(img2)

            max_num_image = min(img1.shape[0], self.num_images)
            grid_img1 = torchvision.utils.make_grid(img1[0:max_num_image])
            trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
            
            grid_img2 = torchvision.utils.make_grid(img2[0:max_num_image])
            trainer.logger.experiment.add_image('img2', grid_img2.cpu().numpy(), 0)

class SimScoreImageLogger(Callback):
    def __init__(self, num_images=18, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            (img1, img2), scores = batch
            
            with torch.no_grad():
                img2 = pl_module.noise_transform(img2)

            max_num_image = min(img1.shape[0], self.num_images)
            grid_img1 = torchvision.utils.make_grid(img1[0:max_num_image])


            # Generate figure
            plt.clf()
            fig = plt.figure(figsize=(7, 9))
            ax = plt.imshow(grid_img1.permute(1, 2, 0).cpu().numpy())            
            # trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
            trainer.logger.experiment["images"].upload(fig)

            plt.close()
            
            # grid_img2 = torchvision.utils.make_grid(img2[0:max_num_image])
            # trainer.logger.experiment.add_image('img2', grid_img2.cpu().numpy(), 0)

            # for idx, s in enumerate(scores):
            #     trainer.logger.experiment.add_scalar('scores', s, idx)

class SimNorthImageLogger(Callback):
    def __init__(self, num_images=18, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            img1, img2 = batch
            
            with torch.no_grad():
                img2 = pl_module.noise_transform(img2)

            max_num_image = min(img1.shape[0], self.num_images)
            grid_img1 = torchvision.utils.make_grid(img1[0:max_num_image])


            # Generate figure
            plt.clf()
            fig = plt.figure(figsize=(7, 9))
            ax = plt.imshow(grid_img1.permute(1, 2, 0).cpu().numpy())

            # trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
            trainer.logger.experiment["images"].upload(fig)

            plt.close()
            
            # grid_img2 = torchvision.utils.make_grid(img2[0:max_num_image])
            # trainer.logger.experiment.add_image('img2', grid_img2.cpu().numpy(), 0)

            # for idx, s in enumerate(scores):
            #     trainer.logger.experiment.add_scalar('scores', s, idx)

class EffnetDecodeImageLogger(Callback):
    def __init__(self, num_images=12, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            img1, img2 = pl_module.train_transform(batch)
            
            max_num_image = min(img1.shape[0], self.num_images)
            grid_img1 = torchvision.utils.make_grid(img1[0:max_num_image])
            trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
            
            grid_img2 = torchvision.utils.make_grid(img2[0:max_num_image])
            trainer.logger.experiment.add_image('img2', grid_img2.cpu().numpy(), 0)

            # with torch.no_grad():
            #     x_hat, z = pl_module(img1)

            #     grid_x_hat = torchvision.utils.make_grid(x_hat[0:max_num_image])
            #     trainer.logger.experiment.add_image('x_hat', grid_x_hat, 0)

class AutoEncoderImageLogger(Callback):
    def __init__(self, num_images=16, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():
                img1, img2 = batch
                img2 = pl_module.noise_transform(img2)

                max_num_image = min(img1.shape[0], self.num_images)
                grid_img1 = torchvision.utils.make_grid(img1[0:max_num_image])
                trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
                
                grid_img2 = torchvision.utils.make_grid(img2[0:max_num_image])
                trainer.logger.experiment.add_image('img2', grid_img2.cpu().numpy(), 0)

                x_hat, z = pl_module(img2)

                grid_x_hat = torchvision.utils.make_grid(x_hat[0:max_num_image])
                trainer.logger.experiment.add_image('x_hat', torch.tensor(grid_x_hat), 0)

class BlindSweepImageLogger(Callback):
    def __init__(self, num_images=16, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            
            img, ga = batch                
            
            # grid_img1 = torchvision.utils.make_grid(img[0,:,:,:])
            # trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)


class DiffusionImageLogger(Callback):
    def __init__(self, num_images=16, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():
                x = batch

                max_num_image = min(x.shape[0], self.num_images)
                grid_img1 = torchvision.utils.make_grid(x[0:max_num_image])


                trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), pl_module.global_step)

                x_hat, z_mu, z_sigma = pl_module(x)

                grid_x_hat = torchvision.utils.make_grid(x_hat[0:max_num_image])


                trainer.logger.experiment.add_image('x_hat', torch.tensor(grid_x_hat), pl_module.global_step)
                

class DiffusionImageLoggerNeptune(Callback):
    def __init__(self, num_images=16, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():
                x = batch

                max_num_image = min(x.shape[0], self.num_images)


                x = x[0:max_num_image]

                grid_img1 = torchvision.utils.make_grid(x[0:max_num_image])
                x_ = pl_module(x)

                if isinstance(x_, tuple):
                    if len(x_) == 2:
                        x_hat, _ = x_
                    else:
                        x_hat, z_mu, z_sigma = x_
                else:
                    x_hat = x_


                x = x.clip(0, 1)
                x_hat = x_hat.clip(0,1)

                grid_x_hat = torchvision.utils.make_grid(x_hat[0:max_num_image])
                
                # Generate figure                
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_img1.permute(1, 2, 0).cpu().numpy())
                # trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
                trainer.logger.experiment["images/x"].upload(fig)
                plt.close()

                # Generate figure
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_hat.permute(1, 2, 0).cpu().numpy())
                # trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
                trainer.logger.experiment["images/x_hat"].upload(fig)
                plt.close()


class DiffusionImageLoggerPairedNeptune(Callback):
    def __init__(self, num_images=16, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():
                x = batch[0]
                y = batch[1]

                max_num_image = min(x.shape[0], self.num_images)


                x = x[0:max_num_image]

                grid_img1 = torchvision.utils.make_grid(x[0:max_num_image])
                x_ = pl_module(x)

                if isinstance(x_, tuple):
                    if len(x_) == 2:
                        x_hat, _ = x_
                    else:
                        x_hat, z_mu, z_sigma = x_
                else:
                    x_hat = x_


                x = x.clip(0, 1)
                x_hat = x_hat.clip(0,1)

                grid_x_hat = torchvision.utils.make_grid(x_hat[0:max_num_image])
                
                # Generate figure                
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_img1.permute(1, 2, 0).cpu().numpy())
                # trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
                trainer.logger.experiment["images/x"].upload(fig)
                plt.close()

                # Generate figure
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_hat.permute(1, 2, 0).cpu().numpy())
                # trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
                trainer.logger.experiment["images/x_hat"].upload(fig)
                plt.close()



                y = y[0:max_num_image]

                grid_img2 = torchvision.utils.make_grid(y[0:max_num_image])
                
                # Generate figure
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_img2.permute(1, 2, 0).cpu().numpy())                
                trainer.logger.experiment["images/y"].upload(fig)
                plt.close()

class GenerativeImageLoggerNeptune(Callback):
    def __init__(self, num_images=16, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():
                x = batch

                max_num_image = min(x.shape[0], self.num_images)

                x = x[0:max_num_image]

                x_hat = pl_module(x.shape[0])

                x = x - torch.min(x)
                x = x/torch.max(x)

                x_hat = x_hat - torch.min(x_hat)
                x_hat = x_hat/torch.max(x_hat)


                grid_img1 = torchvision.utils.make_grid(x)
                grid_x_hat = torchvision.utils.make_grid(x_hat)
                
                # Generate figure                
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_img1.permute(1, 2, 0).cpu().numpy())
                # trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
                trainer.logger.experiment["images/x"].upload(fig)
                plt.close()

                # Generate figure
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_hat.permute(1, 2, 0).cpu().numpy())
                # trainer.logger.experiment.add_image('img1', grid_img1.cpu().numpy(), 0)
                trainer.logger.experiment["images/x_hat"].upload(fig)
                plt.close()

class DiffusionImageLoggerMRUS(Callback):
    def __init__(self, num_images=16, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():
                x_mr, x_us = batch

                max_num_image = min(x_mr.shape[0], self.num_images)

                x_mr_us_hat, z_mu_mr_us, z_sigma_mr_us = pl_module.get_us(x_mr)

                x_mr = x_mr.clip(0, 1)
                x_mr_us_hat = x_mr_us_hat.clip(0,1)


                grid_x_mr = torchvision.utils.make_grid(x_mr[0:max_num_image])
                grid_x_mr_us_hat = torchvision.utils.make_grid(x_mr_us_hat[0:max_num_image])

                
                # add figure          
                trainer.logger.experiment.add_image('x_mr', grid_x_mr.cpu().numpy(), pl_module.global_step)
                trainer.logger.experiment.add_image('x_mr_us_hat', grid_x_mr_us_hat.cpu().numpy(), pl_module.global_step)


                x_us_mr_hat, z_mu_us_mr, z_sigma_us_mr = pl_module.get_mr(x_us)

                x_us = x_us.clip(0, 1)
                x_us_mr_hat = x_us_mr_hat.clip(0,1)


                grid_x_us = torchvision.utils.make_grid(x_us_mr_hat[0:max_num_image])
                grid_x_us_mr_hat = torchvision.utils.make_grid(x_us_mr_hat[0:max_num_image])

                
                # Generate figure                
                trainer.logger.experiment.add_image('x_us', grid_x_us.cpu().numpy(), pl_module.global_step)
                trainer.logger.experiment.add_image('x_us_mr_hat', grid_x_us_mr_hat.cpu().numpy(), pl_module.global_step)
                


class DiffusionImageLoggerMRUSNeptune(Callback):
    def __init__(self, num_images=16, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():
                x_mr, x_us = batch

                max_num_image = min(x_mr.shape[0], self.num_images)

                x_mr_us_hat, z_mu_mr_us, z_sigma_mr_us = pl_module.get_us(x_mr)

                x_mr = x_mr.clip(0, 1)
                x_mr_us_hat = x_mr_us_hat.clip(0,1)


                grid_x_mr = torchvision.utils.make_grid(x_mr[0:max_num_image])
                grid_x_mr_us_hat = torchvision.utils.make_grid(x_mr_us_hat[0:max_num_image])

                
                # Generate figure                
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_mr.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_mr"].upload(fig)
                plt.close()

                # Generate figure
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_mr_us_hat.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_mr_us_hat"].upload(fig)
                plt.close()





                x_us_mr_hat, z_mu_us_mr, z_sigma_us_mr = pl_module.get_mr(x_us)

                x_us = x_us.clip(0, 1)
                x_us_mr_hat = x_us_mr_hat.clip(0,1)


                grid_x_us = torchvision.utils.make_grid(x_us[0:max_num_image])
                grid_x_us_mr_hat = torchvision.utils.make_grid(x_us_mr_hat[0:max_num_image])

                
                # Generate figure                
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_us.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_us"].upload(fig)
                plt.close()

                # Generate figure
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_x_us_mr_hat.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_us_mr_hat"].upload(fig)
                plt.close()