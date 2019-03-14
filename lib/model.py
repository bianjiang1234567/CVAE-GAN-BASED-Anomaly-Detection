"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from lib.networks import NetG, NetD, weights_init, Net_Auto_D, Net_W_D, weights_init_WD, sampling, Gaussian, y_classfier, latent_loss, GlobalDiscriminator, LocalDiscriminator, DeepInfoMaxLoss, D_net_gauss
from lib.visualizer import Visualizer
from lib.loss import l2_loss
from lib.evaluate import evaluate

import lib.pytorch_ssim as ssim_package
from torch.optim import lr_scheduler
import torch.nn.functional as F
import math
# from PIL import Image
# from scipy.signal import convolve2d
#from SSIM_PIL import compare_ssim as ssim

##
class Ganomaly(object):
    """GANomaly Class
    """

    @staticmethod
    def name():
        """Return name of the class.
        """
        return 'Ganomaly'

    def __init__(self, opt, dataloader=None):
        super(Ganomaly, self).__init__()
        ##
        # Initalize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train') #输出路径
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test') #输出路径

        self.D_count=0
        self.CRITIC_ITERS = self.opt.CRITIC_ITERS
        self.CRITIC_ITERS2 = self.opt.CRITIC_ITERS2
        self.CRITIC_ITERS3 = self.opt.CRITIC_ITERS3
        #self.device = torch.device("cuda:2" if self.opt.device != 'cpu' else "cpu")
        self.device = torch.device("cuda:{}".format(self.opt.gpu_ids[0]) if self.opt.device != 'cpu' else "cpu")

        # -- Discriminator attributes.
        self.out_d_real = None#d输出 对应真正的样本
        self.feat_real = None#真正的real样本 图片
        self.err_d_real = None
        self.fake = None
        self.latent_i = None#z_mean
        self.latent_o = None#z_log_var
        self.out_d_fake = None#d输出 对应真正的假样本
        self.feat_fake = None#真正的假样本
        self.err_d_fake = None#
        self.err_d = None
        self.ssim_loss = None#SSIM loss 最小为-1

        self.err_d_bce = None#判别器的交叉熵
        self.D_w_cost = None
        self.Wasserstein_D = None
        self.G_w_cost = None

        self.kl_loss = None
        self.cat_loss = None
        self.z_loss = None

        self.vae_loss= None

        self.aae_d_loss = None
        self.aae_g_loss = None

        self.info_local = None
        self.info_global = None
        self.prior_loss = None
        self.local_fool = None
        self.global_fool = None
        self.prior_fool = None
        self.kl_loss = None
        self.cat_loss = None
        self.info_local_decoder = None

        #negative learning
        self.ssim_loss_neg = None
        self.err_g_l1l_neg = None
        self.err_g_neg = None


        # -- Generator attributes.
        self.out_g = None
        self.err_g_bce = None
        self.err_g_l1l = None
        self.err_g_enc = None
        self.err_g = None
        self.err_auto_d = None
        self.err_g_d_l1l = None

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0
        self.best_auc = 0

        ##
        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device)
        self.netd = NetD(self.opt).to(self.device)
        self.net_auto_d = Net_Auto_D(self.opt).to(self.device)#不用
        self.net_w_d = Net_W_D(self.opt).to(self.device)
        self.net_Gaussian = Gaussian(self.opt).to(self.device)#不用
        self.net_y_classfier = y_classfier(self.opt).to(self.device)#不用
        #GlobalDiscriminator, LocalDiscriminator, DeepInfoMaxLoss
        self.net_d_aae = D_net_gauss(self.opt).to(self.device)  # 不用初始化
        self.net_dim_loss_fn = DeepInfoMaxLoss(self.opt).to(self.device)


        # 高斯那个已经初始化了
        #self.net_y_classfier.apply(weights_init)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)
        self.net_auto_d.apply(weights_init)#不用
        #self.net_w_d.apply(weights_init)
        self.net_w_d.apply(weights_init_WD)
        self.ssim = ssim_package.SSIM().to(self.device)

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            print("\tDone.\n")

        # print(self.netg)
        # print(self.netd)

        ##
        # Loss Functions
        self.bce_criterion = nn.BCELoss()
        self.l1l_criterion = nn.L1Loss()
        self.l2l_criterion = l2_loss

        ##
        # Initialize input tensors.#创建初始的输入
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.real_label = 1
        self.fake_label = 0

        ##mnist 8 8 10 cifar 2 2 4
        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.net_auto_d.train()
            self.net_w_d.train()
            self.net_Gaussian.train()
            self.net_y_classfier.train()
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))#D2
            self.net_d_aae.train()


            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr / 2  , betas=(self.opt.beta1, 0.999))#E D1 E /2222222222222222222

            self.optimizer_w_d = optim.Adam(self.net_w_d.parameters(), lr=self.opt.lr / 1  , betas=(self.opt.beta1, 0.999))#/2
            self.optimizer_w_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr / 1, betas=(self.opt.beta1, 0.999))#/4444444444444444
            self.optimizer_vae_g = optim.Adam([{'params':self.netg.encoder1_part1.parameters()},
                                               {'params':self.netg.encoder1_part2.parameters()},
                                               {'params':self.netg.z_mean.parameters()},
                                               {'params':self.netg.z_log_var.parameters()}], lr=self.opt.lr / 2, betas=(self.opt.beta1, 0.999))#/44444444444444
            #self.optimizer_vae_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr / 2, betas=(self.opt.beta1, 0.999))  # /4
            self.optimizer_neg_g = optim.Adam([{'params':self.netg.decoder_part1.parameters()},
                                               {'params':self.netg.decoder_part2.parameters()}], lr=self.opt.lr / 2, betas=(self.opt.beta1, 0.999))#/444444444444
            self.optimizer_net_d_aae = optim.Adam(self.net_d_aae.parameters(),
                                                  lr=self.opt.lr / 4)  # ,betas=(self.opt.beta1, 0.999))
            self.optimizer_aae_g = optim.Adam([{'params':self.netg.encoder1_part1.parameters()},
                                               {'params':self.netg.encoder1_part2.parameters()},
                                               {'params':self.netg.z_mean.parameters()}], lr=self.opt.lr / 4, betas=(self.opt.beta1, 0.999))  # , betas=(self.opt.beta1, 0.999))

            self.optimizer_dim_loss_fn = optim.Adam([#######原始代码
                                                     #{'params':self.netg.encoder1_part1.parameters()},
                                                     #{'params':self.netg.encoder1_part2.parameters()},
                                                     #{'params': self.netg.z_mean.parameters()},
                                                     #{'params': self.netg.z_log_var.parameters()},
                                                     #######
                                                     {'params': self.netg.parameters()},
                                                     {'params':self.net_dim_loss_fn.parameters()}], lr=self.opt.lr / 1)#/2会去寻找loss从哪里来2222222222222222


            self.optimizer_y_classfier = optim.Adam([{'params': self.net_Gaussian.parameters()},
                                                     {'params': self.net_y_classfier.parameters()},
                                                  {'params': self.netg.encoder1_part1.parameters()},
                                                  {'params': self.netg.encoder1_part2.parameters()},
                                                  {'params': self.netg.z_mean.parameters()},
                                                  {'params': self.netg.z_log_var.parameters()}], lr=self.opt.lr / 4,#/4
                                                 betas=(self.opt.beta1, 0.999))

            #self.optimizer_w_d = optim.RMSprop(self.net_w_d.parameters(), lr=self.opt.lr/4)
            #self.optimizer_w_g = optim.RMSprop(self.netg.parameters(), lr=self.opt.lr/4)


            #wgan-gp的学习率改下来 交替更新次数 更新下降方法 更新哪些参数
            #self.optimizer_ssim = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

            def lambda_rule(epoch):#epoach自带的变量
                lr_l = 1.0 - max(0, epoch + 1  - self.opt.iter - 10 ) / float(self.opt.niter_decay)#从30开始以1/200斜率线性缩减
                return lr_l
            ####从30开始的参数是针对cifar和mnist的 学习率衰减
            ####caltch 不需要学习率衰减

            self.scheduler_optimizer_g = lr_scheduler.LambdaLR(self.optimizer_g, lr_lambda=lambda_rule)

            self.scheduler_optimizer_d = lr_scheduler.LambdaLR(self.optimizer_d, lr_lambda=lambda_rule)
            self.scheduler_optimizer_w_d = lr_scheduler.LambdaLR(self.optimizer_w_d, lr_lambda=lambda_rule)
            self.scheduler_optimizer_w_g = lr_scheduler.LambdaLR(self.optimizer_w_g, lr_lambda=lambda_rule)
            self.scheduler_optimizer_vae_g = lr_scheduler.LambdaLR(self.optimizer_vae_g, lr_lambda=lambda_rule)
            self.scheduler_optimizer_neg_g = lr_scheduler.LambdaLR(self.optimizer_neg_g, lr_lambda=lambda_rule)
            self.scheduler_optimizer_dim_loss_fn = lr_scheduler.LambdaLR(self.optimizer_dim_loss_fn, lr_lambda=lambda_rule)
            self.scheduler_optimizer_net_d_aae = lr_scheduler.LambdaLR(self.optimizer_net_d_aae, lr_lambda=lambda_rule)
            self.scheduler_optimizer_aae_g = lr_scheduler.LambdaLR(self.optimizer_aae_g, lr_lambda=lambda_rule)

            self.scheduler_optimizer_y_classfier = lr_scheduler.LambdaLR(self.optimizer_y_classfier, lr_lambda=lambda_rule)#不用


            #self.scheduler_optimizer_g = lr_scheduler.StepLR(self.optimizer_g, step_size=15, gamma=0.5)
            #self.scheduler_optimizer_auto_d = lr_scheduler.StepLR(self.optimizer_auto_d, step_size=15, gamma=0.5)
            #self.scheduler_optimizer_d = lr_scheduler.StepLR(self.optimizer_d, step_size=15, gamma=0.5)
            #self.scheduler_optimizer_w_d = lr_scheduler.StepLR(self.optimizer_w_d, step_size=15, gamma=0.5)
            #self.scheduler_optimizer_w_g = lr_scheduler.StepLR(self.optimizer_w_g, step_size=15, gamma=0.5)

    ##
    def set_input(self, input):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        self.input.data.resize_(input[0].size()).copy_(input[0])#0 1 表示数据和标签
        self.gt.data.resize_(input[1].size()).copy_(input[1])#0 1 表示数据和标签
        #print("self.gt=",self.gt)
        #print("self.total_steps4=",self.total_steps)
        #print("self.opt.batchsize=",self.opt.batchsize)
        #print("input[0].size()=",input[0].size())
        #print("input[1].size()=", input[1].size())
        # Copy the first batch as the fixed input.
        if self.total_steps == self.opt.batchsize:#无论是第几个epoach都是first batch
            #print("Writing")
            #print("self.total_steps=",self.total_steps)
            #print("self.opt.batchsize=",self.opt.batchsize)
            self.fixed_input.data.resize_(input[0].size()).copy_(input[0])#

    ##


    def update_w_netd(self):
        self.net_w_d.zero_grad()
        one = torch.FloatTensor([1])
        mone = one * -1
        one = one.cuda(self.device)
        mone = mone.cuda(self.device)


        out_d_real, _ = self.net_w_d(self.input)  # sigomid    和   特征层LeakyReLU(negative_slope=0.2, inplace)
        out_d_real = out_d_real.mean()
        #out_d_real.backward(one)

        #####原始

        #####自编吗器加噪声
        #####
        #self.fake, self.latent_i, self.latent_o,_,_,_ = self.netg(self.input + (torch.rand(size=(self.opt.batchsize, self.opt.nc, self.opt.isize, self.opt.isize))/5 - 0.1).to(self.device))
        self.fake, self.latent_i, self.latent_o, _, _, _ = self.netg(self.input)
        #####

        #####限制wgan拉近哪些样本距离 避免将negative拉近
        ##self.fake, self.latent_i, self.latent_o = self.netg(self.input)
        ##fake = self.netg.decoder(self.latent_i.view(-1,self.opt.nz,1,1))
        ##out_d_fake, _ = self.net_w_d(fake.detach())
        ##out_d_fake = out_d_fake.mean()
        #####

        #####原始
        out_d_fake, _ = self.net_w_d(self.fake.detach())
        out_d_fake = out_d_fake.mean()
        #####
        #out_d_fake.backward(mone)

        #####原始
        gradient_penalty = self.calc_gradient_penalty(self.net_w_d, self.input, self.fake)
        #####

        #####限制wgan拉近哪些样本距离 避免将negative拉近
        ##gradient_penalty = self.calc_gradient_penalty(self.net_w_d, self.input, fake)
        #####


        #gradient_penalty.backward()

        #self.D_w_cost = - out_d_fake + out_d_real - gradient_penalty
        self.D_w_cost = out_d_fake - out_d_real + gradient_penalty
        self.D_w_cost.backward()
        self.Wasserstein_D = out_d_real - out_d_fake

        #print("Wasserstein_D=",self.Wasserstein_D)
        self.optimizer_w_d.step()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        # print "real_data: ", real_data.size(), fake_data.size()
        BATCH_SIZE = self.opt.batchsize
        #alpha = torch.rand(BATCH_SIZE, 1)
        #alpha = torch.ones(BATCH_SIZE, 1)-0.5###收敛太快，太像和不像都不好 因此取0.5  1最快
        alpha = torch.ones(BATCH_SIZE, 1)  ###收敛太快，太像和不像都不好 因此取0.5  1最快
        alpha = alpha.expand(BATCH_SIZE, real_data.nelement() // BATCH_SIZE).contiguous().view(BATCH_SIZE, self.opt.nc, self.opt.isize, self.opt.isize)
        alpha = alpha.cuda(self.device) if 1 else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        ###############惩罚生成数据
        #interpolates_fake =  alpha * fake_data
        ###############

        if 1:
            interpolates = interpolates.cuda(self.device)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates,_ = netD(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(
                                      self.device) if 1 else torch.ones( disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

        ######惩罚生成数据
        ##interpolates_fake = torch.ones(BATCH_SIZE, 1).expand(BATCH_SIZE, fake_data.nelement() // BATCH_SIZE).contiguous().view(BATCH_SIZE, self.opt.nc, self.opt.isize, self.opt.isize) * fake_data
        #interpolates_fake = interpolates_fake.cuda(self.device)
        #interpolates_fake = torch.autograd.Variable(interpolates_fake, requires_grad=True)
        #disc_interpolates_fake, _ = netD(interpolates_fake)
        #gradients_fake = torch.autograd.grad(outputs=disc_interpolates_fake, inputs=interpolates_fake,
        #                                grad_outputs=torch.ones(disc_interpolates_fake.size()).cuda(
        #                                    self.device) if 1 else torch.ones(disc_interpolates.size()),
        #                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        #gradients_fake = gradients_fake.view(gradients_fake.size(0), -1)#只取batchsize
        #gradient_penalty_fake = ((gradients_fake.norm(2, dim=1) - 1) ** 2).mean() * 10#消除dim1

        #gradient_penalty = 0.5* (gradient_penalty_fake + gradient_penalty)
        ########
        return gradient_penalty


    def update_w_netg_auto(self):
        self.netg.zero_grad()

        ####加噪声
        #self.fake, self.latent_i, self.latent_o, _, _,_ = self.netg(self.input + (torch.rand(size=(self.opt.batchsize, self.opt.nc, self.opt.isize, self.opt.isize))/5 - 0.1).to(self.device))
        self.fake, self.latent_i, self.latent_o, _, _, _ = self.netg(self.input)
        #####compute ssmi
        #self.ssim_loss = 1 - self.ssim(self.input, self.fake)  # 在 0到1
        self.ssim_loss = - torch.log(self.ssim(self.input, self.fake) + 1e-15)###在update_w_netd里面被算过
        #####交叉熵 按照batchsize来比较  欺骗 D网络的输出
        #self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)
        #self.err_g_bce = self.bce_criterion(self.out_g, self.label)  # 交叉熵 按照batchsize来比较  欺骗 D网络的输出
        # print("err_g_bce=",self.err_g_bce)
        #####
        self.err_g_l1l = self.l1l_criterion(self.fake, self.input)  # constrain x' to look like x
        #self.err_g_enc = self.l2l_criterion(self.latent_o, self.latent_i)  # 编码误差
        # self.err_g = self.err_g_bce * self.opt.w_bce + self.err_g_l1l * self.opt.w_rec + self.err_g_enc * self.opt.w_enc
        # self.err_g = self.err_g_bce * self.opt.w_bce + self.err_g_l1l * self.opt.w_rec
        # self.err_g = self.err_g_bce * self.opt.w_bce * (self.opt.w_rec/2) + self.ssim_loss * (self.opt.w_rec/2) + self.err_g_l1l * self.opt.w_rec
        # self.err_g =  self.err_g_bce * (self.opt.w_rec / 10) + self.err_d * (self.opt.w_rec / 2) + self.ssim_loss * (self.opt.w_rec / 2) + self.err_g_l1l * self.opt.w_rec
        #self.err_g = self.ssim_loss * self.opt.w_rec + self.err_g_l1l * self.opt.w_rec  # 仅仅让特征越来越近 使得分不开

        #latent loss 计算 不加这个loss就会退化成普通的编码器
        #self.vae_loss = latent_loss(self.latent_i, torch.exp(self.latent_o / 2))

        #self.vae_negative_loss = latent_loss(self.latent_i, 2*torch.exp(self.latent_o / 2))

        #cifar#减少权重增加gan的探索
        #self.err_g = 50 * self.ssim_loss + 50 * self.err_g_l1l
        #self.err_g = 65 * self.ssim_loss + 35 * self.err_g_l1l
        #self.err_g = 10 * self.ssim_loss + 50 * self.err_g_l1l##caltech的权重   打分也按照权重分配
        #self.err_g = 50 * self.ssim_loss + 10 * self.err_g_l1l##cifar10的权重   打分也按照权重分配5比1 ssim l1 #\

        #####caltech256 想要像参数要大
        self.err_g = 50 * self.ssim_loss + 50 * self.err_g_l1l  ##cifar10的权重   打分也按照权重分配5比1 ssim l1 #\
        ###self.err_g = 10 * self.ssim_loss + 50 * self.err_g_l1l
        ####ssim 下降过快容易不稳定，ssimloss对于重建困难的 不好分辨的可以加大ssim loss

        #self.err_g = 50 * self.err_g_l1l
        #self.err_g = 1 * self.err_g_l1l
        #####

        #mnist
        #self.err_g = 50 * self.ssim_loss + 50 * self.err_g_l1l

        self.err_g.backward(retain_graph=True)
        #self.vae_loss.backward()
        self.optimizer_g.step()


    def update_vae_netg(self):#方差不能过小或者过大
        #latent loss 计算 不加这个loss就会退化成普通的编码器
        ##self.netg.zero_grad()

        self.netg.encoder1_part1.zero_grad()
        self.netg.encoder1_part2.zero_grad()
        self.netg.z_mean.zero_grad()
        self.netg.z_log_var.zero_grad()

        ###self.net_Gaussian.zero_grad()
        ###self.net_y_classfier.zero_grad()

        ##### 0均值的高斯分布
        _, self.latent_i, self.latent_o, _,_,_ = self.netg(self.input)
        self.vae_loss = latent_loss(self.latent_i, torch.exp(self.latent_o / 2))#没有网络 只是计算
        if self.epoch >= 0:
            #vae_loss = 5 * self.vae_loss#10是对于64的isize  对于32的isize用5
            self.vae_loss = 1 * torch.log(self.vae_loss)  # 10是对于64的isize  对于32的isize用5
            ###self.vae_loss.backward(retain_graph=True)
            ##self.optimizer_vae_g.step()
        #####

        #####自动聚类的高斯分布
        self.fake, self.latent_i, self.latent_o, _, _,_ = self.netg(self.input)#laten——i去对齐均值，均值为真值，还是从latenti采样

        z_prior_mean = self.net_Gaussian(self.netg.z_added_noise)#(None, self.num_classes, nz)
        y = self.net_y_classfier(self.netg.z_added_noise)#(-1, 9)
        #print("z_prior_mean=", z_prior_mean)
        #print("y=", y)

        M = torch.zeros(1, self.opt.nz)
        self.kl_loss = - 0.5 * (self.latent_o.unsqueeze(1) - torch.pow(z_prior_mean,2))  # (batch_size, num_classes, latent_dim)  隐空间num_classes,其实只有一个数字表示类别，这么多latent_dim对应一个numclass 实际的均值还是latentdim 独立同分布的组合
        #print("self.kl_loss1=", self.kl_loss)
        #print("y.unsqueeze(1).size()=", y.unsqueeze(1).size())
        #print("self.kl_loss.size()=", self.kl_loss.size())
        self.kl_loss = torch.mean(torch.addbmm(M, y.unsqueeze(1).cpu(), self.kl_loss.cpu())/self.opt.batchsize#(-1,1, 9)#(None, self.num_classes, input_shape[-1])
                         ).to(self.device) #mean0 对numclasse 所有类别求和   #softmax的结果给(batch_size, num_classes_probility）里的numclasses家了括号 和  klloss乘积  沿着非batchsize第一个维度行求平均  扩大一个维度  softmax=K.expand_dims(y, 1)
        #addbmm batchsize已经求和了  1*300 已经对y进行了求和
        #print("self.kl_loss2=", self.kl_loss)
        #self.kl_loss = torch.mean(self.kl_loss)
        #print("self.kl_loss3=", self.kl_loss)
        self.cat_loss = torch.mean(torch.mean(y * torch.log(y + 1e-15),0))#(-1,9)调整重合度 softmax输出接近均匀分布  交叉上 p(y|z)是网络输出,q(y)是标签
        #-2.197就已经是最小数值了
        #batchsize 也要球平均
        #print("self.cat_loss1=", self.cat_loss)
        #px标签是y qx输出是y 最大商
        #self.cat_loss = torch.mean(self.cat_loss)#全部求和  对应还会求导
        ###loss = self.cat_loss + 10 * self.kl_loss###(64,300)  (64,9)
        ###loss = self.cat_loss + 5 * self.kl_loss  ###(64,300)  (64,9)###10是对于64的isize 对于32的isize用5

        loss = 1 * self.cat_loss + 1 * self.kl_loss
        #loss = 1 * self.kl_loss

        #print("self.cat_loss2=",self.cat_loss)
        #print("self.kl_loss4=",self.kl_loss)
        ######cvae KLloss 优化
        loss.backward()
        self.optimizer_y_classfier.step()
        ######

        ####通过调节参数权重  改成aae后更加连续  抽样后scheduler_optimizer_dim_loss_fn的数值强行拉到正太分布上去

    def update_aae_netg(self):  # 方差不能过小或者过大
        # latent loss 计算 不加这个loss就会退化成普通的编码器
        #self.netg.zero_grad()
        self.net_d_aae.zero_grad()

        self.netg.encoder1_part1.zero_grad()
        self.netg.encoder1_part2.zero_grad()
        self.netg.z_mean.zero_grad()

        TINY = 1e-15

        ##z_real_gauss = Variable(torch.randn(self.opt.batchsize, self.opt.nz) * 5.).to(self.device)
        z_real_gauss = Variable(torch.randn(self.opt.batchsize, self.opt.nz) * 1.).to(self.device)

        D_real_gauss = self.net_d_aae(z_real_gauss)
        D_fake_gauss = self.net_d_aae(self.latent_i.detach())

        self.aae_d_loss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))

        self.aae_d_loss.backward()
        self.optimizer_net_d_aae.step()

        D_fake_gauss = self.net_d_aae(self.latent_i)
        self.aae_g_loss = -0.25 * torch.mean(torch.log(D_fake_gauss + TINY))

        self.aae_g_loss.backward(retain_graph=True)
        self.optimizer_aae_g.step()



    def update_vae_negative_netg(self):
        # latent loss 计算 不加这个loss就会退化成普通的编码器
        self.netg.zero_grad()
        #        self.fake, self.latent_i, self.latent_o,_ = self.netg(self.input)
        # sigma = torch.exp(self.latent_o / 2)
        #        latent_i = self.latent_i
        # sigma = torch.exp(self.latent_o / 2) + 0.5 * torch.exp(self.latent_o / 2)

        ###网络已经改变 应该重新计算
        self.fake, self.latent_i, self.latent_o, _, _,_ = self.netg(self.input)
        sigma = torch.exp(self.latent_o / 2)
        # print('sigma.size()=', sigma.size())
        ##sigma = torch.exp(self.latent_o / 2) + (torch.rand(size=(self.latent_o .view(-1,self.opt.nz).shape[0], self.opt.nz))).to(self.device) * torch.exp(self.latent_o / 2)
        ##latent_i = self.latent_i + (torch.rand((self.latent_i.view(-1,self.opt.nz).shape[0], self.opt.nz))).to(self.device) * self.latent_i
        # sigma.to(self.device)
        #        z_added_noise_negative = sampling(latent_i, torch.log(sigma * sigma), self.opt)
        # print("sigma=",sigma)
        # print("self.latent_o=", self.latent_o)
        # print("torch.abs(z_added_noise_negative)=", )
        # z_added_noise_negative = z_added_noise_negative + (z_added_noise_negative>0).to(torch.float32).mul(sigma).mul(0.5)\
        # + (z_added_noise_negative<0).to(torch.float32).mul(sigma).mul(-0.5)
        #        tmp=(z_added_noise_negative > 0).to(torch.float32). \
        #            mul(sigma).mul(abs(torch.rand(size=(z_added_noise_negative.shape[0], self.opt.nz)).to(self.device))) \
        #        + (z_added_noise_negative < 0).to(torch.float32). \
        #            mul(sigma).mul(abs(torch.rand(size=(z_added_noise_negative.shape[0], self.opt.nz)).to(self.device)))

        tmp4 = (self.netg.z_added_noise > 0).to(torch.float32). \
                  mul(2.5*sigma).mul(abs((torch.rand(size=(self.netg.z_added_noise.shape[0], self.opt.nz))/2 + 0.5).to(self.device))) \
              + (self.netg.z_added_noise < 0).to(torch.float32). \
                  mul(-2.5*sigma).mul(abs((torch.rand(size=(self.netg.z_added_noise.shape[0], self.opt.nz))/2 + 0.5).to(self.device)))

        tmp5 = (self.netg.z_added_noise > 0).to(torch.float32). \
                   mul(3 * sigma).mul(
            abs((torch.rand(size=(self.netg.z_added_noise.shape[0], self.opt.nz))/2 + 0.5).to(self.device))) \
               + (self.netg.z_added_noise < 0).to(torch.float32). \
                   mul(-3 * sigma).mul(
            abs((torch.rand(size=(self.netg.z_added_noise.shape[0], self.opt.nz))/2 + 0.5).to(self.device)))

        #tmp = 1.5 * sigma.mul((torch.rand(size=(self.netg.z_added_noise.shape[0], self.opt.nz)).to(self.device)))
        #tmp2 = 2 * sigma.mul((torch.rand(size=(self.netg.z_added_noise.shape[0], self.opt.nz)).to(self.device)))
        #tmp3 = 3 * sigma.mul((torch.rand(size=(self.netg.z_added_noise.shape[0], self.opt.nz)).to(self.device)))


        #z_added_noise_negative = self.netg.z_added_noise + tmp.to(self.device)
        #z_added_noise_negative2 = self.netg.z_added_noise + tmp2.to(self.device)
        #z_added_noise_negative3 = self.netg.z_added_noise + tmp3.to(self.device)
        z_added_noise_negative4 = self.netg.z_added_noise + tmp4.to(self.device)
        z_added_noise_negative5 = self.netg.z_added_noise + tmp5.to(self.device)

        #        z_added_noise_negative = latent_i + (latent_i > 0).to(torch.float32).mul(
        #            torch.abs(torch.randn(self.opt.batchsize, self.opt.nz, 1, 1).to(self.device))).mul(0.5) \
        #                                 + (latent_i < 0).to(torch.float32).mul(
        #            torch.abs(torch.randn(self.opt.batchsize, self.opt.nz, 1, 1).to(self.device))).mul(-0.5)

        # print('z_added_noise_negative.size=',z_added_noise_negative.size())
        # print('z_added_noise_negative[0,0]=', z_added_noise_negative[0,0])
        ##for i in range(0,self.opt.batchsize):
        ##    for j in range(0,self.opt.nz):
        ##        if z_added_noise_negative[i,j]>=0:
        # print('i={},j={}'.format(i,j))
        # print('z_added_noise_negative[i,j]=', z_added_noise_negative[i, j])
        # z_added_noise_negative[i, j] = 1 + z_added_noise_negative[i, j]
        # print('z_added_noise_negative[i,j]=', z_added_noise_negative[i, j])
        ##            z_added_noise_negative[i, j] = z_added_noise_negative[i,j] + (sigma[i,j])
        # print('z_added_noise_negative[i,j]=', z_added_noise_negative[i, j])
        # z_added_noise_negative[i,j] = z_added_noise_negative[i,j] + (sigma).to(self.device)
        ##        else:
        ##            z_added_noise_negative[i,j] = z_added_noise_negative[i,j] + ( -1 * sigma[i,j])
        # while(any(z_added_noise_negative < latent_i))
        # z_added_noise_negative = sampling(latent_i, torch.log(sigma * sigma), self.opt)
        ##z_added_noise_negative = z_added_noise_negative.to(self.device)

        #gen_imag = self.netg.decoder(z_added_noise_negative.view(-1, self.opt.nz, 1, 1))
        #gen_imag2 = self.netg.decoder(z_added_noise_negative.view(-1, self.opt.nz, 1, 1))
        #gen_imag3 = self.netg.decoder(z_added_noise_negative.view(-1, self.opt.nz, 1, 1))
        gen_imag4 = self.netg.decoder(z_added_noise_negative4.view(-1, self.opt.nz, 1, 1))
        gen_imag5 = self.netg.decoder(z_added_noise_negative5.view(-1, self.opt.nz, 1, 1))

        #self.ssim_loss_neg = torch.log(self.ssim(self.input, gen_imag))  # 最小化两者之间的ssim
        #self.err_g_l1l_neg = -torch.log(self.l1l_criterion(gen_imag, self.input))
        #self.ssim_loss_neg2 = torch.log(self.ssim(self.input, gen_imag2))  # 最小化两者之间的ssim
        #self.err_g_l1l_neg2 = -torch.log(self.l1l_criterion(gen_imag2, self.input))
        #self.ssim_loss_neg3 = torch.log(self.ssim(self.input, gen_imag3))  # 最小化两者之间的ssim
        #self.err_g_l1l_neg3 = -torch.log(self.l1l_criterion(gen_imag3, self.input))
        self.ssim_loss_neg4 = torch.log(self.ssim(self.input, gen_imag4))  # 最小化两者之间的ssim
        self.err_g_l1l_neg4 = -self.l1l_criterion(gen_imag4, self.input)
        self.ssim_loss_neg5 = torch.log(self.ssim(self.input, gen_imag5))  # 最小化两者之间的ssim
        self.err_g_l1l_neg5 = -self.l1l_criterion(gen_imag5, self.input)


        #self.err_g_neg = 1 * self.ssim_loss_neg + 1 * self.err_g_l1l_neg \
        #                 + 1 * self.ssim_loss_neg2 + 1 * self.err_g_l1l_neg2 + 1 * self.ssim_loss_neg3 + 1 * self.err_g_l1l_neg3# ssim 尽快提升相似度 后面靠gan

        #self.err_g_neg = 10 * self.ssim_loss_neg4 + 10 * self.err_g_l1l_neg4
        #self.err_g_neg = self.ssim_loss_neg4 + self.err_g_l1l_neg4
        err_g_neg = 1 * (self.ssim_loss_neg4 + self.err_g_l1l_neg4 + self.ssim_loss_neg5 +  self.err_g_l1l_neg5)

        self.err_g_neg = -0.25 *( err_g_neg / 1 )


        # print(self.epoch)
        #if self.epoch <= 5:
        self.ssim_loss_neg = -0.5 * (self.ssim_loss_neg4 + self.ssim_loss_neg5)
        self.err_g_l1l_neg = -0.5 * (self.err_g_l1l_neg4 + self.err_g_l1l_neg5)
        #    self.err_g_neg = 0 * self.err_g_neg

        if self.epoch >= 0 :
            err_g_neg.backward()
            self.optimizer_neg_g.step()



    def update_dim_netg(self):
        self.net_dim_loss_fn.zero_grad()

        self.netg.zero_grad()

        ###加噪声
        #self.fake, self.latent_i, self.latent_o, M, global_features, local_features_decoder = self.netg(self.input + (torch.rand(size=(self.opt.batchsize, self.opt.nc, self.opt.isize, self.opt.isize))/5 - 0.1).to(self.device))
        self.fake, self.latent_i, self.latent_o, M, global_features, local_features_decoder = self.netg(self.input)
        ######vae编码器
        y = self.netg.z_added_noise#M local_features_encoder
        #####普通编码器
        #y = self.latent_i  # M local_features_encoder

        orders1 = np.array(range(0, M.shape[0]))
        np.random.shuffle(orders1)
        #M_prime = M[orders]
        M_prime = M[orders1[0]].unsqueeze(0)
        for i in orders1[1:]:
            M_prime = torch.cat((M_prime, M[i].unsqueeze(0)), dim=0)  # batchsize内部打乱
        #M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)#batchsize内部打乱

        #####global features
        ###orders2 = np.array(range(0, global_features.shape[0]))
        ###np.random.shuffle(orders2)
        # M_prime = M[orders]
        ###global_features_prime = global_features[orders2[0]].unsqueeze(0)
        ###for i in orders2[1:]:
        ###    global_features_prime = torch.cat((global_features_prime, global_features[i].unsqueeze(0)), dim=0)  # batchsize内部打乱
        # M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)#batchsize内部打乱
        #####

        ########利用y和y采样
        orders3 = np.array(range(0, y.shape[0]))
        np.random.shuffle(orders3)
        Y_prime = y[orders3[0]].unsqueeze(0)
        for i in orders3[1:]:
            Y_prime = torch.cat((Y_prime, y[i].unsqueeze(0)), dim=0)
        #print("y.size()=",y.size())
        #print("Y_prime.size()=", Y_prime.size())
        y_resample = sampling(self.latent_i, self.latent_o, self.opt)
        ###self.info_local, self.info_global = self.net_dim_loss_fn(y, M, M_prime, y_resample, Y_prime)
        ########

        #self.info_local, self.info_global, self.prior_loss, self.local_fool, self.global_fool, self.prior_fool = self.net_dim_loss_fn(y, M, M_prime)


        #####global features
        ###self.info_local, self.info_global = self.net_dim_loss_fn(y, M, M_prime, global_features, global_features_prime)
        #####

        #####local features decoder
        orders4 = np.array(range(0, local_features_decoder.shape[0]))
        np.random.shuffle(orders4)
        # M_prime = M[orders]
        local_features_decoder_prime = local_features_decoder[orders4[0]].unsqueeze(0)
        for i in orders4[1:]:
            local_features_decoder_prime = torch.cat((local_features_decoder_prime, local_features_decoder[i].unsqueeze(0)), dim=0)  #
        #####

        #####原始代码
        ###self.info_local, self.info_global = self.net_dim_loss_fn(y, M, M_prime, y_resample, Y_prime)
        #####
        self.info_local, self.info_global, self.info_local_decoder = self.net_dim_loss_fn(y, M, M_prime, y_resample, Y_prime, local_features_decoder, local_features_decoder_prime)

        #loss = self.info_local + self.info_global + self.prior_loss + self.prior_fool
        ###loss = self.info_local + self.info_global
        loss = 15 * self.info_local
        ###loss = self.info_local + self.info_local_decoder

        #if self.epoch <= 3:
        #    self.info_local = 0 * self.info_local
        #    self.info_global = 0 * self.info_global

        #if self.epoch >= 0 :
        loss.backward()
        self.optimizer_dim_loss_fn.step()

        #self.net_dim_loss_fn.zero_grad()
        #self.netg.zero_grad()
        #self.prior_fool.backward()
        #self.optimizer_dim_loss_fn.step()




    def update_w_netg(self, only_w_g=False):
        self.netg.zero_grad()

        # one = torch.empty(size=(self.opt.batchsize,1,1,1)).fill_(1.0)
        one = torch.FloatTensor([1])
        # one = one.cuda(self.device)
        mone = -1 * one.cuda(self.device)

        ######compute feature matching loss
        # _, self.feat_real = self.netd(self.input)  # sigomid    和   特征层LeakyReLU(negative_slope=0.2, inplace)

        #####原始
        self.fake, _, _, _, _, _ = self.netg(self.input)
        #####加噪声
        #self.fake, _, _, _,_,_ = self.netg(self.input + (torch.rand(size=(self.opt.batchsize, self.opt.nc, self.opt.isize, self.opt.isize))/5 - 0.1).to(self.device))
        out_g, _ = self.net_w_d(self.fake)
        #####

        #####限制wgan拉近哪些样本距离 避免将negative拉近
        ##self.fake, self.latent_i, _ = self.netg(self.input)
        ##fake = self.netg.decoder(self.latent_i.view(-1,self.opt.nz,1,1))
        ##out_g, _ = self.net_w_d(fake)
        #####

        # out_g = torch.squeeze(out_g).mean()
        out_g = -torch.squeeze(out_g).mean()
        # out_g.backward(one, retain_graph=True)
        # out_g.backward(mone, retain_graph=True)
        out_g.backward(retain_graph=True)
        self.optimizer_w_g.step()

        # out_g = torch.squeeze(out_g)
        # self.G_w_cost = out_g
        self.G_w_cost = out_g  # 不应该加两次负号 在这里再加一次就成了两次负号了
        # self.err_d = l2_loss(self.feat_real, self.feat_fake)  # 特征层的误差  [64, 256, 4, 4] 变成一个数作为误差
        # print(self.err_d) 在0.5左右
        ######


    ##
    def reinitialize_netd(self):
        """ Initialize the weights of netD
        """
        self.netd.apply(weights_init)
        print('Reloading d net')

    ##
    def optimize(self):
        """ Optimize netD and netG  networks.
        """

        #self.update_netd()
        #self.update_netg()

        #if self.D_count<self.CRITIC_ITERS:
        #    self.update_w_netd()
        #    self.update_w_netg(only_w_g=True)全部要更新ssim l1reconstuction cheat欺骗
        #else:
        #   self.update_w_netd()
        #   self.update_w_netg(only_w_g=False)只欺骗

        #与上面那一块等价
        #if self.D_count<self.CRITIC_ITERS:
        #    self.update_w_netd()
        #    self.update_w_netg_auto()
        #    self.update_w_netg()
        #else:
        #    self.update_w_netd()
        #    self.update_w_netg()

        ##wgan和自编码器
        ##self.update_dim_netg()
        ##self.update_vae_netg()
        if self.D_count<self.CRITIC_ITERS:#一起训练5步3
            self.update_w_netd()
            self.update_w_netg_auto()

            ###self.update_aae_netg()
            self.update_vae_netg()

            ####caltech 不需要
            self.update_vae_negative_netg()
            ####


            #self.update_w_netg_auto()
            self.update_dim_netg()
            #self.update_cvae() 暂时不用
            #self.update_w_netg()
            #print("step11111")
            self.update_w_netg_auto()

        elif self.D_count<self.CRITIC_ITERS2:#gan_d训练4步3
            self.update_w_netd()
            #print("step22222")
        elif self.D_count<self.CRITIC_ITERS3:#gan_g训练1步1
            self.update_w_netg()
            #print("step33333")


        #self.update_w_netd()
        #self.update_w_netg_auto()
        #self.update_w_netg()


        #self.update_net_auto_d()
        #if self.epoch % 5 == 0:
        #    self.update_net_auto_g()

        # If D loss is zero, then re-initialize netD
        #if self.err_d_real.item() < 1e-5 or self.err_d_fake.item() < 1e-5:
        #    self.reinitialize_netd()

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([#('err_d_fm', self.err_d.item()),
                              #('err_g', self.err_g.item()),
                              #('err_d_real', self.err_d_real.item()),
                              #('err_d_fake', self.err_d_fake.item()),
                              #('err_g_bce', self.err_g_bce.item()),
                              #('err_g_d_l1l', self.err_g_d_l1l.item()),
                              ('discriminator_cost', self.D_w_cost.item()),
                              #('err_g_l1l', self.err_g_l1l.item()),
                              ('generator_cost', self.G_w_cost.item()),
                              ('Wasserstein_distance', self.Wasserstein_D.item())])
                              #('err_g_enc', self.err_g_enc.item()),
                              #('ssim_loss', self.ssim_loss.item())])
                              #('err_d_bce', self.err_d_bce.item()),])
                              #('err_auto_d', self.err_auto_d.item())])
                              #('err_d_dis', self.err_d_dis.item())])

        errors2 = OrderedDict([  # ('err_d_fm', self.err_d.item()),
            # ('err_g', self.err_g.item()),
            # ('err_d_real', self.err_d_real.item()),
            # ('err_d_fake', self.err_d_fake.item()),
            # ('err_g_bce', self.err_g_bce.item()),
            # ('err_g_d_l1l', self.err_g_d_l1l.item()),
            #('D_w_cost', self.D_w_cost.item()),
            #('Wasserstein_D', self.Wasserstein_D.item()),
            ('kl_loss', self.kl_loss.item()),
            ('cat_loss',self.cat_loss.item()),
            #('z_loss',self.z_loss.item()),
            #err_g_neg
            #('err_g_neg', self.err_g_neg.item()),
            ('err_g_l1l_neg', self.err_g_l1l_neg.item()),
            ('ssim_loss_neg', self.ssim_loss_neg.item()),
            #('local_fool', self.local_fool.item()),
            #('global_fool', self.global_fool.item()),
            #('prior_fool', self.prior_fool.item()),
            #('prior_loss', self.prior_loss.item()),
            #('info_max', self.info_local.item()),
            ######('info_local_decoder', self.info_local_decoder),
            ######('kl_loss', self.kl_loss.item()),
            ######('category_loss', self.cat_loss.item()),
            ('info_max', self.info_local.item()),
            #('info_global', self.info_global.item()),
            ######('vae_loss',self.vae_loss.item()),
            #('aae_d_loss', self.aae_d_loss.item()),
            #('aae_g_loss', self.aae_g_loss.item()),

            ('l1_loss', self.err_g_l1l.item()),
            #('G_w_cost', self.G_w_cost.item()),
            # ('err_g_enc', self.err_g_enc.item()),
            ('ssim_loss', self.ssim_loss.item())])

            # ('err_d_bce', self.err_d_bce.item()),])
            # ('err_auto_d', self.err_auto_d.item())])
            # ('err_d_dis', self.err_d_dis.item())])

        return errors, errors2

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input)[0].data#netg返回的第0个参数 生成的图像

        return reals, fakes, fixed

    ##
    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
                   '%s/netG.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
                   '%s/netD.pth' % (weight_dir))

    ##
    def train_epoch(self):
        """ Train the model for one epoch.
        """

        self.netg.train()#进入训练状态
        epoch_iter = 0
        self.D_count = 0

        for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):# data和label  一个epoach
            self.total_steps += self.opt.batchsize # 取多少个样本
            epoch_iter += self.opt.batchsize # 取多少个样本
            #print("data.size()=", len(data))
            #print("self.total_steps=", self.total_steps)
            self.set_input(data) # data已经是batchsize了
            self.optimize()
            if self.D_count == self.CRITIC_ITERS3 - 1:
                self.D_count = 0
            else:
                self.D_count += 1
                #print("self.D_count=",self.D_count)
            #print("data=",data) dataloader 里面把data全部变成了-1 到1 之间的数字所以能跟tanh
            if self.total_steps % self.opt.print_freq == 0:
                errors, errors_2 = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)
                    self.visualizer.plot_current_errors2(self.epoch, counter_ratio, errors_2)

            if self.total_steps % self.opt.save_image_freq == 0:
                reals, fakes, fixed = self.get_current_images()
                self.visualizer.save_current_images(self.epoch, reals, fakes, fixed, self.fixed_input)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes, fixed)


            ###if self.opt.dataset == "cifar10":
            ###if self.opt.dataset == "cifar10" and self.epoch >= 8:
            ###if self.opt.dataset == "mnist":##注释掉相当与每个采样 测试一下
            if 0:
                res = self.test()
                if res['AUC'] > self.best_auc:
                    self.best_auc = res['AUC']
                    self.save_weights(self.epoch)
                self.visualizer.print_current_performance(res, self.best_auc)  # 当前的auc打印出来

                if res['AUC_PR'] > self.best_auc_pr:
                    self.best_auc_pr = res['AUC_PR']
                    # self.save_weights(self.epoch)
                self.visualizer.print_current_performance(res, self.best_auc_pr)  # 当前的auc打印出来

                if res['f1_score'] > self.best_f1_score:
                    self.best_f1_score = res['f1_score']
                    # self.save_weights(self.epoch)
                self.visualizer.print_current_performance(res, self.best_f1_score)  # 当前的auc打印出来


        print(">> Training model %s. Epoch %d/%d" % (self.name(), self.epoch+1, self.opt.niter))


        # self.visualizer.print_current_errors(self.epoch, errors)
    def update_lr(self):
        #self.scheduler_optimizer_g.step()
        #self.scheduler_optimizer_auto_d.step()
        #self.scheduler_optimizer_d.step()
        self.scheduler_optimizer_w_d.step()#更新wgand的权重衰减
        self.scheduler_optimizer_w_g.step()#更新wgang的权重衰减
        self.scheduler_optimizer_g.step()#更新自编码器的权重衰减

        self.scheduler_optimizer_vae_g.step()
        self.scheduler_optimizer_neg_g.step()
        self.scheduler_optimizer_dim_loss_fn.step()

        self.scheduler_optimizer_net_d_aae.step()
        self.scheduler_optimizer_aae_g.step()

        self.scheduler_optimizer_y_classfier.step()#暂时不用

        print("optimizer_w_d.param_groups=", self.optimizer_w_d.param_groups[0]['lr'])
        #print("optimizer_auto_d.param_groups=", self.optimizer_auto_d.param_groups[0]['lr'])
        print("optimizer_w_g.param_groups=", self.optimizer_w_g.param_groups[0]['lr'])
        print("optimizer_g.param_groups=", self.optimizer_g.param_groups[0]['lr'])
        print("optimizer_vae_g.param_groups=", self.optimizer_vae_g.param_groups[0]['lr'])
        print("optimizer_neg_g.param_groups=", self.optimizer_neg_g.param_groups[0]['lr'])
        print("optimizer_dim_loss_fn.param_groups=", self.optimizer_dim_loss_fn.param_groups[0]['lr'])
        print("optimizer_y_classfier.param_groups=", self.optimizer_y_classfier.param_groups[0]['lr'])

        print("optimizer_net_d_aae.param_groups=", self.optimizer_net_d_aae.param_groups[0]['lr'])
        print("optimizer_aae_g.param_groups=", self.optimizer_aae_g.param_groups[0]['lr'])


    ##
    def train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0
        self.best_auc = 0
        self.best_auc_pr = 0
        self.best_f1_score = 0

        # Train for niter epochs.
        print(">> Training model %s." % self.name())
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            #print("Train for one epoch")
            #print("total_steps1=",self.total_steps)
            self.train_epoch()
            print("Wasserstein_D=", self.Wasserstein_D)
            print("ssim_loss=", self.ssim_loss)
            print("err_g_l1l=", self.err_g_l1l)
            print("vae_loss=", self.vae_loss)
            print("err_g_neg=", self.err_g_neg)
            print("kl_loss=", self.kl_loss.item())
            print("cat_loss=", self.cat_loss.item())
            #print("info_local_decoder=", self.info_local_decoder.item()),
            #print("err_g_l1l_neg=", self.err_g_l1l_neg)
            #print("ssim_loss_neg=", self.ssim_loss_neg)
            #print("prior_loss=", self.prior_loss)
            ##print("prior_fool=", self.prior_fool)
            print("info_max=", self.info_local)
            #print("info_local=", self.info_local)
            print("aae_d_loss=", self.aae_d_loss)
            print("aae_g_loss=", self.aae_g_loss)
            #print("info_global=", self.info_global)

            self.update_lr()
            #print("total_steps2=", self.total_steps)
            res = self.test()#计算AUC
            #res_auc_roc, res_auc_pr, res_f1_score = self.test()
            #print("total_steps3=", self.total_steps)
            if res['AUC'] > self.best_auc:
                self.best_auc = res['AUC']
                self.save_weights(self.epoch)
            self.visualizer.print_current_performance(res, self.best_auc)#当前的auc打印出来

            if res['AUC_PR'] > self.best_auc_pr:
                self.best_auc_pr = res['AUC_PR']
                #self.save_weights(self.epoch)
            self.visualizer.print_current_performance(res, self.best_auc_pr)  # 当前的auc打印出来

            if res['f1_score'] > self.best_f1_score:
                self.best_f1_score = res['f1_score']
                #self.save_weights(self.epoch)
            self.visualizer.print_current_performance(res, self.best_f1_score)  # 当前的auc打印出来

        print(">> Training model %s.[Done]" % self.name())

    ##
    def test(self):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                path = "./output/{}/{}/train/weights/netG.pth".format(self.name().lower(), self.opt.dataset)
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            self.opt.phase = 'test'

            # Create big error tensor for the test set.
            #全部的数据作为大误差 按照batchsize来算
            self.an_scores_recon = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
            self.an_scores_local_info = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
            self.an_scores_local_info_decoder = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32,
                                                    device=self.device)
            self.an_scores_global_info = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32,
                                                    device=self.device)
            #self.an_scores_fm = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
            self.an_scores_ssim = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
            self.an_scores_wdis = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
            self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long,    device=self.device)
            self.latent_i  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.latent_o  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)#按照batchsize算好

            # print("   Testing model %s." % self.name())
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            for i, data in enumerate(self.dataloader['test'], 0):#0表示从0开始 已经是batchsize了  shuffle=shuffle[x],#每个epoach都要改组
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                self.fake, self.latent_i, self.latent_o, local_features, global_features, local_features_decoder = self.netg(self.input)# batchsize 100 1 1     batchsize 1 4 4

                #reconstructio loss
                if 1:
                    ###########重建误差算2范数
                    #error_reconstruction_flatten=torch.flatten((self.input - self.fake), start_dim=1, end_dim=-1)
                    #error_reconstruction = torch.norm(error_reconstruction_flatten,dim=1)#按列算矩阵的2范数
                    ###########
                    ###########info
                    y = self.netg.z_added_noise
                    M = local_features
                    #M_prime = torch.cat((features[1:], features[0].unsqueeze(0)), dim=0)
                    #self.info_local, self.info_global = self.net_dim_loss_fn(latent_i, features, M_prime)

                    y_exp = y.unsqueeze(-1).unsqueeze(-1)  ##在后面加一个维度[64 64 ]
                    ###y_exp = y_exp.expand(-1, -1, 8, 8)  # 我们的featuremap是16 复制
                    y_exp = y_exp.expand(-1, -1, 16, 16)  # 我们的featuremap是16 复制
                    ###y_exp = y_exp.expand(-1, -1, 32, 32)  # 我们的featuremap是16 复制
                    y_M = torch.cat((M, y_exp), dim=1)  # encoded features 拼接
                    #y_M_prime = torch.cat((M_prime, y_exp), dim=1)  # encoded 旋转图片 拼接
                    #print("local_loss.size()=",local_loss.size())
                    local_loss = -F.softplus(-self.net_dim_loss_fn.local_d(y_M)).mean(dim=1).mean(dim=1).mean(dim=1)  # 局部 直接分类  用1×1卷积 用的是论文中的第一种结构

                    #### local_feature_decoder
                    y_local_features_decoder = torch.cat((local_features_decoder, y_exp), dim=1)
                    local_loss_decoder = -F.softplus(-self.net_dim_loss_fn.local_d_decoder(y_local_features_decoder)).mean(dim=1).mean(dim=1).mean(dim=1)
                    ####

                    #####global features
                    ###h = torch.cat((global_features.view(-1, 512), y), dim=1)
                    ###global_loss = -F.softplus(-self.net_dim_loss_fn.global_d(h)).squeeze(1)  # 全局 对features和旋转图片再提特征 相当于用能够进行分类的图片特征
                    #global_loss = local_loss
                    #####

                    # 因为是无监督 因此保留了图片是哪一类的信息   再拼接去分类
                    #print("local_loss.size()=", local_loss.size()) 64
                    #print("global_loss.size()=", global_loss.size()) 64

                    #####利用 z 和 z采样
                    y_resample = sampling(self.latent_i, self.latent_o, self.opt)
                    h = torch.cat((y_resample.view(-1, self.opt.nz), y), dim=1)
                    global_loss = -F.softplus(-self.net_dim_loss_fn.global_d(h)).squeeze(1)
                    #####


                    ###########
                    ###########重建l1 loss
                    error_reconstruction_l1 = torch.flatten(abs(self.input - self.fake), start_dim=1, end_dim=-1)
                    #print("error_reconstruction_l1=", error_reconstruction_l1)#64
                    #print("error_reconstruction_l1.shape=", error_reconstruction_l1.shape)
                    error_reconstruction_l1 = torch.mean(error_reconstruction_l1,dim=1)#把第二个维度消除
                    ###########ssim loss
                    #error_ssim = 1 - ssim_package.ssim(self.input, self.fake, size_average = False)
                    error_ssim = - torch.log(ssim_package.ssim(self.input, self.fake, size_average=False) + 1e-15)
                    #error_ssim = - ssim_package.ssim(self.input, self.fake, size_average=False)
                    #print("error_ssim=",error_ssim)
                    ###########
                   # print("error_ssim.size()=", error_ssim.size())
                   # print("error_reconstruction_l1.size()=", error_reconstruction_l1.size())
                    out_d_real, _ = self.net_w_d(self.input)  # sigomid    和   特征层LeakyReLU(negative_slope=0.2, inplace)
                    out_d_fake, _ = self.net_w_d(self.fake)
                    Wasserstein_D = out_d_real - out_d_fake
                    #print("Wasserstein_D1=",Wasserstein_D)
                    Wasserstein_D = torch.flatten(Wasserstein_D, start_dim=1, end_dim=-1)
                    Wasserstein_D = torch.mean(Wasserstein_D, dim=1)  # 把第二个维度消除
                    #print("Wasserstein_D2=", Wasserstein_D)
                    ###########ssim loss
                    #error_ssim = 1 - ssim_package.ssim(self.input, self.fake, size_average=False)



                    ###########
                #elif self.opt.dataset == "cifar10":
                    ###########重建l1 loss
                #    error_reconstruction_l1 = torch.flatten(abs(self.input - self.fake), start_dim=1, end_dim=-1)
                    # print("error_reconstruction_l1=", error_reconstruction_l1)#64
                    # print("error_reconstruction_l1.shape=", error_reconstruction_l1.shape)
                #    error_reconstruction_l1 = torch.mean(error_reconstruction_l1, dim=1)  # 把第二个维度消除
                    ###########

                #fm loss
                ###if self.opt.dataset == "mnist" or "cifar10":
                    #############特征层2范数
                    #_, feat_input_test = self.netd(self.input)# 256层 batch_norm
                    #_, feat_fake_test = self.netd(self.fake)
                    #error_fm_flatten=torch.flatten((feat_input_test - feat_fake_test), start_dim=1, end_dim=-1)
                    #error_fm = torch.norm(error_fm_flatten, dim=1)  # 按列算矩阵的2范数
                    ############

                    ############feature loss l2
                    ###_, feat_input_test = self.netd(self.input)  # 256层 batch_norm
                    ###_, feat_fake_test = self.netd(self.fake)
                    ###error_fm_l2 = torch.pow((feat_input_test - feat_fake_test), 2)#对应元素
                    #print("error_fm_l2.shape=", error_fm_l2.shape)
                    ###error_fm_l2 = torch.flatten(error_fm_l2,start_dim=1,end_dim=-1)
                    ###error_fm_l2 = torch.mean(error_fm_l2, dim=1)
                    #error_fm_l2 = l2_loss(feat_input_test , feat_fake_test)
                    #print("error_fm_l2.shape=",error_fm_l2.shape)
                    ############

                    ############feature loss l2 auto dis
                    #_, feat_input_test = self.net_auto_d(self.input)  # 256层 batch_norm
                    #_, feat_fake_test = self.net_auto_d(self.fake)
                    #error_fm_l2 = torch.pow((feat_input_test - feat_fake_test), 2)  # 对应元素
                    # print("error_fm_l2.shape=", error_fm_l2.shape)
                    #error_fm_l2 = torch.flatten(error_fm_l2, start_dim=1, end_dim=-1)
                    #error_fm_l2 = torch.mean(error_fm_l2, dim=1)
                    # error_fm_l2 = l2_loss(feat_input_test , feat_fake_test)
                    # print("error_fm_l2.shape=",error_fm_l2.shape)
                    ############


                # total loss
                #if self.opt.dataset == "mnist" or self.opt.dataset == "cifar10":
                    #####平衡权重 重建误差2范数 特征层2范数
                    #error_reconstruction_sum=torch.sum(error_reconstruction)
                    #error_fm_sum = torch.sum(error_fm)
                    #weight_error_reconstruction_sum = 1 - error_reconstruction_sum / (error_fm_sum + error_reconstruction_sum);
                    #weight_error_fm_sum = 1 - error_fm_sum / (error_fm_sum + error_reconstruction_sum);
                    #error = weight_error_reconstruction_sum * error_reconstruction + weight_error_fm_sum * error_fm
                    #####

                    #####平衡权重 loss
                    #error_reconstruction_sum_l1 = torch.sum(error_reconstruction_l1)#这个不合理 只比较像素灰度
                    #error_fm_sum_l2 = torch.sum(error_fm_l2)#生成器扩大 判别器缩小，最后没有变化
                    #error_ssim_sum = torch.sum(error_ssim)
                    #weight_error_reconstruction_sum_l1 = 1 - error_reconstruction_sum_l1 / (
                    #        error_fm_sum_l2 + error_reconstruction_sum_l1 + error_ssim_sum)
                    #weight_error_fm_sum_l2 = 1 - error_fm_sum_l2 / (error_fm_sum_l2 + error_reconstruction_sum_l1 + error_ssim_sum)
                    #weight_error_ssim = 1 - error_ssim_sum / ( error_fm_sum_l2 + error_reconstruction_sum_l1 + error_ssim_sum)
                    #error = weight_error_reconstruction_sum_l1 * error_reconstruction_l1 + weight_error_fm_sum_l2 * error_fm_l2 + weight_error_ssim * error_ssim_sum
                    #print("error_ssim=",error_ssim)
                    #print("error_ssim_sum=", error_ssim_sum)
                    #weight_error_reconstruction_sum_l1_2 = 1 - error_reconstruction_sum_l1 / (error_reconstruction_sum_l1 + error_ssim_sum)
                    #weight_error_ssim_2 = 1 - error_ssim_sum / (error_reconstruction_sum_l1 + error_ssim_sum)
                    #error = weight_error_ssim_2 * error_ssim + weight_error_reconstruction_sum_l1_2 * error_reconstruction_l1
                    #####

                ########original loss
                #error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)  # 沿着列算  沿着第二个维度球平均
                ######

                time_o = time.time()

                self.an_scores_recon[i*self.opt.batchsize : i*self.opt.batchsize+error_reconstruction_l1.size(0)] = error_reconstruction_l1
                #self.an_scores_fm[i*self.opt.batchsize : i*self.opt.batchsize+error_fm_l2.size(0)] = error_fm_l2
                self.an_scores_ssim[i*self.opt.batchsize : i*self.opt.batchsize+error_ssim.size(0)] = error_ssim
                self.an_scores_wdis[i*self.opt.batchsize : i*self.opt.batchsize+Wasserstein_D.size(0)] = Wasserstein_D

                self.an_scores_local_info[i*self.opt.batchsize : i*self.opt.batchsize+local_loss.size(0)] = -local_loss
                #print("global_loss.size()=",global_loss.size())
                #print("self.an_scores_global_info[i * self.opt.batchsize: i * self.opt.batchsize + global_loss.size(0)].size()=", self.an_scores_global_info[i * self.opt.batchsize: i * self.opt.batchsize + global_loss.size(0)].size())
                self.an_scores_global_info[i*self.opt.batchsize : i*self.opt.batchsize+global_loss.size(0)] = -global_loss
                self.an_scores_local_info_decoder[i * self.opt.batchsize: i * self.opt.batchsize + local_loss_decoder.size(0)] = -local_loss_decoder
                #self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0)) #变成单一维度 batchsize
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error_ssim.size(0)] = self.gt.reshape(error_ssim.size(0))
                #print("self.gt=",self.gt)

                #self.latent_i [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)
                #self.latent_o [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_o.reshape(error.size(0), self.opt.nz)#batchsize 100

                self.times.append(time_o - time_i)
                #print(self.opt.save_test_images)
                #print("noSAVE!!!!!")
                # Save test images.
                if self.opt.save_test_images:
                    #print(self.opt.save_test_images)
                    #print("SAVE!!!!!")
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    #储存图像
                    vutils.save_image(real, '%s/real_%03d.png' % (dst, i+1), normalize=True)#i表示地几个batchsize
                    vutils.save_image(fake, '%s/fake_%03d.png' % (dst, i+1), normalize=True)

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]

            self.an_scores_recon = (self.an_scores_recon - torch.min(self.an_scores_recon)) / (torch.max(self.an_scores_recon) - torch.min(self.an_scores_recon))
            #self.an_scores_fm = (self.an_scores_fm - torch.min(self.an_scores_fm)) / (torch.max(self.an_scores_fm) - torch.min(self.an_scores_fm))
            self.an_scores_ssim = (self.an_scores_ssim - torch.min(self.an_scores_ssim)) / (torch.max(self.an_scores_ssim) - torch.min(self.an_scores_ssim))
            self.an_scores_wdis = (self.an_scores_wdis - torch.min(self.an_scores_wdis)) / (torch.max(self.an_scores_wdis) - torch.min(self.an_scores_wdis))
            self.an_scores_local_info = (self.an_scores_local_info - torch.min(self.an_scores_local_info)) / (torch.max(self.an_scores_local_info) - torch.min(self.an_scores_local_info))
            self.an_scores_global_info = (self.an_scores_global_info - torch.min(self.an_scores_global_info)) / (torch.max(self.an_scores_global_info) - torch.min(self.an_scores_global_info))
            self.an_scores_local_info_decoder = (self.an_scores_local_info_decoder - torch.min(self.an_scores_local_info_decoder)) / (
                    torch.max(self.an_scores_local_info_decoder) - torch.min(self.an_scores_local_info_decoder))

            #self.an_scores = self.an_scores_recon + self.an_scores_ssim + self.an_scores_wdis
            #self.an_scores = self.an_scores_ssim + self.an_scores_wdis
            #self.an_scores = self.an_scores_ssim
            #self.an_scores = self.an_scores_recon#只用住像素误差 caltech打分
            #self.an_scores = self.an_scores_wdis #只用w距离
            #print("self.an_scores=",self.an_scores)
            #print("self.gt_labels=", self.gt_labels)
            #self.an_scores = 0.1 * self.an_scores_recon + 0.5 * self.an_scores_ssim
            #self.an_scores = self.an_scores_local_info
            #self.an_scores = 0.2 * self.an_scores_recon + 0.8 * self.an_scores_ssim + 0.2 * self.an_scores_local_info + 0.2 * self.an_scores_local_info_decoder
            #self.an_scores = 0.2 * self.an_scores_recon + 0.8 * self.an_scores_ssim + 0.5 * self.an_scores_local_info

            print("self.an_scores_recon=",self.an_scores_recon)
            print("self.an_scores_ssim=",self.an_scores_ssim)
            #print("self.an_scores_ssim=", self.an_scores_ssim)
            print("self.an_scores_local_info=",self.an_scores_local_info)

            if all(np.isnan(self.an_scores_ssim.cpu().numpy())) == False:
                ######对于ssim过于相似的物体 采用recon误差好 或者降低训练中ssim的权重
                #print(282)
                print("ssim")
                self.an_scores = self.an_scores_ssim + self.an_scores_recon +  self.an_scores_local_info
                                 # self.an_scores_ssim + 0.5 * self.an_scores_recon +  0.5 * self.an_scores_local_info
                ####self.an_scores = 0.2 * self.an_scores_recon +  0.2 * self.an_scores_local_info
                ####self.an_scores = 0.2 * self.an_scores_recon + 0.8 * self.an_scores_ssim
            ###    self.an_scores = 0.2 * self.an_scores_recon + 0.8 * self.an_scores_ssim + 0.2 * self.an_scores_local_info  ##W距离抖动太大先拿掉 cifar10 打分
            else:
                self.an_scores = self.an_scores_recon + self.an_scores_local_info
                #self.an_scores_local_info + self.an_scores_recon

                print("recon")

            #self.an_scores = 0.2 * self.an_scores_recon + 0.8 * self.an_scores_ssim + 0.2 * self.an_scores_local_info + 0.2 * self.an_scores_global_info ##W距离抖动太大先拿掉 cifar10 打分
            #self.an_scores = 0.2 * self.an_scores_recon + 0.8 * self.an_scores_ssim ##W距离抖动太大先拿掉 cifar10 打分
            #self.an_scores = 0.2 * self.an_scores_recon + 0.8 * self.an_scores_ssim + 0.05 * self.an_scores_wdis ##W距离抖动太大先拿掉 cifar10 打分
            #self.an_scores = self.ssim_loss  ##W距离抖动太大先拿掉 cifar10 打分
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
            #auc, eer = roc(self.gt_labels, self.an_scores)

            #print("self.gt_labels=", self.gt_labels)
            #print("self.an_scores=", self.an_scores)
            auc = evaluate(self.gt_labels, self.an_scores, metric = self.opt.metric)
            auc_pr = evaluate(self.gt_labels, self.an_scores, metric = 'auprc')
            f1_score = evaluate(self.gt_labels, self.an_scores, metric = 'f1_score')
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc), ('AUC_PR', auc_pr), ('f1_score',f1_score)])
            #performance_auc_roc = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])
            #performance_auc_pr = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])
            #performance_f1_score = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.dataloader['test'].dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return performance
