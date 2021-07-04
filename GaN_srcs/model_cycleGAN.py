import torch
import itertools
from torch import nn, optim
from generator_unet import Unet
from discriminator import Discriminator
from GAN_loss import GANLoss
from image_pool import ImagePool

def init_weights(net, init='norm', gain=0.02):
    
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
            
    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model

def abs_criterion(in_, target):
    return torch.mean(torch.abs(in_ - target))


# def mae_criterion(in_, target):
#     return torch.mean((in_-target)**2)

# def sce_criterion(logits, labels):
#     return torch.mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

class MainModel_cycleGAN(nn.Module):
    def __init__(self, net_G_a2b=None, net_G_b2a=None, lr_G=2e-4, lr_D=2e-4, 
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        self.lr_G = lr_G
        self.lr_D = lr_D
        if net_G_a2b or net_G_b2a is None:
            self.net_G_a2b = init_model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device)
            self.net_G_b2a = init_model(Unet(input_c=2, output_c=1, n_down=8, num_filters=64), self.device)
            
        else:
            self.net_G_a2b = net_G_a2b.to(self.device)
            self.net_G_b2a = net_G_b2a.to(self.device)

        self.net_D_b = init_model(Discriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.net_D_a = init_model(Discriminator(input_c=1, n_down=3, num_filters=64), self.device)

        self.criterionGAN = GANLoss(gan_mode='vanilla').to(self.device)
        self.criterionCycle = nn.L1Loss()
        #self.criterionIdt = nn.L1loss()
        self.opt_G = optim.Adam(itertools.chain(self.net_G_a2b.parameters(), self.net_G_b2a.parameters()), lr=self.lr_G, betas=(beta1, beta2))
        torch.optim.lr_scheduler.StepLR(self.opt_G, 30, gamma=0.1, last_epoch=-1)
        self.opt_D_b = optim.Adam(self.net_D_b.parameters(), lr=self.lr_D, betas=(beta1, beta2))
        self.opt_D_a = optim.Adam(self.net_D_a.parameters(), lr=self.lr_D, betas=(beta1, beta2))

        torch.optim.lr_scheduler.StepLR(self.opt_D_b, 30, gamma=0.1, last_epoch=-1)
        torch.optim.lr_scheduler.StepLR(self.opt_D_a, 30, gamma=0.1, last_epoch=-1)
        self.fake_A_pool = ImagePool(100)  # create image buffer to store previously generated images
        self.fake_B_pool = ImagePool(100)  # create image buffer to store previously generated images

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)
        
    def forward(self):
        self.fake_B = self.net_G_a2b(self.L)
        self.rec_A = self.net_G_b2a(self.fake_B)
        self.fake_A = self.net_G_b2a(self.ab)
        self.rec_B = self.net_G_a2b(self.fake_A)

    def backward_D_standard(self, net_D, real, fake):
        pred_real = net_D(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = net_D(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_b(self):
        real_image = torch.cat([self.L, self.ab], dim=1)
        fake_temp = self.fake_B_pool.query(self.fake_B)
        fake_image = torch.cat([self.L, fake_temp], dim=1)
        self.loss_D_b = self.backward_D_standard(self.net_D_b, real_image, fake_image)
    
    def backward_D_a(self):
        fake_image = self.fake_A_pool.query(self.fake_A)
        self.loss_D_a = self.backward_D_standard(self.net_D_a, self.L, fake_image)

    def backward_G(self):
        fake_image_B = torch.cat([self.L, self.fake_B], dim=1)
        self.loss_G_a2b = self.criterionGAN(self.net_D_b(fake_image_B), True)
        self.loss_G_b2a = self.criterionGAN(self.net_D_a(self.fake_A), True)
        self.loss_cycle_a = self.criterionCycle(self.rec_A, self.L) * self.lambda_L1
        self.loss_cycle_b = self.criterionCycle(self.rec_B, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_a2b + self.loss_G_b2a + self.loss_cycle_a + self.loss_cycle_b
        self.loss_G.backward()

        # fake_image = torch.cat([self.L, self.fake_color_1], dim=1)
        # fake_preds = self.net_D_2(fake_image)
        # self.loss_G_GAN_1 = self.GANcriterion(fake_preds, True)
        # self.loss_G_L1_color = self.L1criterion(self.fake_color_2, self.ab) * self.lambda_L1
        # self.loss_G_L1_gray = self.L1criterion(self.fake_gray_1, self.L) * self.lambda_L1
        # self.loss_G_1 = self.loss_G_GAN_1 + self.loss_G_L1_color + self.loss_G_L1_gray
        # #self.loss_G_1.backward()
        # fake_preds_1 = self.net_D_1(self.fake_gray_2)
        # self.loss_G_GAN_2 = self.GANcriterion(fake_preds_1, True)
        # #self.loss_G_L1_color = self.L1criterion(self.fake_color_2, self.ab) * self.lambda_L1
        # #self.loss_G_L1_gray = self.L1criterion(self.fake_gray_1, self.L) * self.lambda_L1
        # self.loss_G_2 = self.loss_G_GAN_2 + self.loss_G_L1_color + self.loss_G_L1_gray
        # #self.loss_G_2.backward()
        # self.loss_G = (self.loss_G_GAN_2 + self.loss_G_GAN_1 + self.loss_G_L1_color + self.loss_G_L1_gray) / 2
        # #self.loss_G.backward()

    # def backward_D(self):
    #     real_image = torch.cat([self.L, self.ab], dim=1)
    #     self.DB_real = self.net_D_2(real_image.detach())
    #     self.DA_real = self.net_D_1(self.L)
    #     fake_image = torch.cat([self.L, self.fake_color_1], dim=1)
    #     self.DB_fake_sample = self.net_D_2(fake_image.detach())
    #     self.DA_fake_sample = self.net_D_1(self.fake_gray_2)
    #     self.db_loss_real = self.GANcriterion(self.DB_real, False)
    #     self.db_loss_fake = self.GANcriterion(self.DB_fake_sample, False)
    #     self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
    #     self.da_loss_real = self.GANcriterion(self.DA_real, False)
    #     self.da_loss_fake = self.GANcriterion(self.DA_fake_sample, False)
    #     self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
    #     self.loss_D = (self.da_loss + self.db_loss) / 2
    #     self.loss_D.backward(retain_graph=True)
    #     #self.da_loss.backward()
    #     #self.db_loss.backward()
    
    def optimize(self):
        self.forward()

        self.set_requires_grad(self.net_D_b, True)
        self.set_requires_grad(self.net_D_a, True)
        self.opt_D＿b.zero_grad()
        self.opt_D_a.zero_grad()
        self.backward_D_b()
        self.backward_D_a()
        # self.lr_D = set_lr(epochs,e,lr_D)
        self.opt_D_b.step()
        self.opt_D_a.step()
        
        self.set_requires_grad(self.net_D_b, False)
        self.set_requires_grad(self.net_D_a, False)
        self.opt_G.zero_grad()
        self.backward_G()
        # self.lr_G = set_lr(epochs,e,lr_G)
        self.opt_G.step()




