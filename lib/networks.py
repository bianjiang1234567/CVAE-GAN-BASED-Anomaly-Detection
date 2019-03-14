""" Network architectures.
"""
# pylint: disable=W0221,W0622,C0103,R0913

##
import torch
import torch.nn as nn
import torch.nn.parallel
import sys
import torch.nn.functional as F
sys.path.append("..")
from options import Options
##
def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    #print("mod=", mod)
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        #print('BatchNorm initial')
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
    ###elif classname.find('Linear') != -1:###名字就是Linear  nn.Linear函数名
        #params = list(mod.parameters())
        #print("mod=",mod)
    ##    mod.weight.data.normal_(0.0, 0.02)
    ###    nn.init.xavier_normal_(mod.weight.data, gain=1)
        #print('Linear initial')

def weights_init_WD(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    #print("mod=", mod)
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        #mod.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(mod.weight.data, gain=1)
    #elif classname.find('BatchNorm') != -1:
        #print('BatchNorm initial')
        #mod.weight.data.normal_(1.0, 0.02)
        #mod.bias.data.fill_(0)
    ###elif classname.find('Linear') != -1:###名字就是Linear  nn.Linear函数名
        #params = list(mod.parameters())
        #print("mod=",mod)
        #mod.weight.data.normal_(0.0, 0.02)
    ###    nn.init.xavier_normal_(mod.weight.data, gain=1)
        #print('Linear initial')

def weights_init_info(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        # mod.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(mod.weight.data, gain=1)
    # elif classname.find('BatchNorm') != -1:
    # print('BatchNorm initial')
    # mod.weight.data.normal_(1.0, 0.02)
    # mod.bias.data.fill_(0)
    ##elif classname.find('Linear') != -1:###名字就是Linear  nn.Linear函数名
    # params = list(mod.parameters())
    # print("mod=",mod)
    # mod.weight.data.normal_(0.0, 0.02)
    ##    nn.init.xavier_normal_(mod.weight.data, gain=1)
    # print('Linear initial')


###
class Encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """
    #ndf是输出的channel个数
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0, add_final_conv=True):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()# model模型
        # input is nc x isize x isize

        ##main.add_module('initial-conv',
        ##                nn.Conv2d(nc, ndf, 1, 1, 0, bias=False))  # （32+2×0-1）/1+1=32 #wgan-gp kernel是3
        ##main.add_module('initial-relu',
        ##                nn.LeakyReLU(0.2, inplace=True))

        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))# （32+2×1-4）/2+1=16 #wgan-gp kernel是3###第一个ndf是nc
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf# 图像的大小缩小两倍  channel数量不变 16对应64

        #self.netg.main.initial-relu-64

        # Extra layers
        for t in range(n_extra_layers):#没有额外的卷积层
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4: # 图像大于4的话就继续 16 8 4 一共新加两层卷积层
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2# channel 变为2倍
            csize = csize / 2 # 图像缩小两倍

        # state size. K x 4 x 4 #最后一层卷积  一共四层卷积
        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                            nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))# 图像大小现在已经小于4了 (（3）+2×0-4）/2+1=1  nz=100

        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

##
class Decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    """
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4# ngf=64  图像大小      32个channel对应4的图像大小
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial-{0}-{1}-convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf),
                        nn.ReLU(True))

        csize, _ = 4, cngf
        while csize < isize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2 # 配合前面

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(nc),
                        nn.Tanh())#逐元素
        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


##
class NetD(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self, opt):
        super(NetD, self).__init__()
        model = Encoder(opt.isize, 1, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)###nz 改成了1
        layers = list(model.main.children())
        #print("(model.main.children())=",model.main.children())
        #print("layers=",layers)
        #print("*layers[:-1]=", *layers[:-1])
        #print("layers[-1]=", layers[-1])

        self.features = nn.Sequential(*layers[:-1]) # 利用特征层。  *用来去掉list的中括号
        #Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False) BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) LeakyReLU(negative_slope=0.2, inplace

        # 左闭右开所以又加入了 layers[-1]
        self.classifier = nn.Sequential(layers[-1])#nz = 100
        #print("self.classifier", self.classifier)
        self.classifier.add_module('Sigmoid', nn.Sigmoid())
        #print("features", self.features)

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1) #压缩最后一个维度  只剩下batchsize这个维度

        return classifier, features

def sampling(z_mean_tmp, z_log_var_tmp, opt_tmp):
    z_mean, z_log_var, opt = z_mean_tmp, z_log_var_tmp, opt_tmp
    device = torch.device("cuda:{}".format(opt_tmp.gpu_ids[0]) if opt_tmp.device != 'cpu' else "cpu")
    #print("z_mean=",z_mean.view(-1,opt.nz))
    epsilon=torch.randn(size=(z_mean.view(-1,opt.nz).shape[0], opt.nz)).to(device)
    #epsilon = K.random_normal(shape=(K.shape(z_mean)[0], opt.nz))  # 采样  返回形状 tensor z 空间采样  batchsize  w×h×c   两个latentdim
    return z_mean + torch.exp(z_log_var / 2) * epsilon  ## 输出的是logsigma平方

class Gaussian(nn.Module):
    def __init__(self, opt):
        super(Gaussian, self).__init__()
        self.num_classes = opt.num_classes
        #tensor = torch.ones((2,), dtype=torch.float64)
        self.mean = torch.nn.Parameter(torch.zeros(size=(self.num_classes, opt.nz), requires_grad=True))#不用在定义成为variable
    def forward(self, z):
        z = torch.unsqueeze(z,1)
        mean1 = torch.unsqueeze(self.mean,0)
        return z - mean1

class y_classfier(nn.Module):
    def __init__(self, opt):
        super(y_classfier, self).__init__()

        #classfier = nn.Sequential()
        #classfier.add_module('y1', nn.Linear(opt.nz, 2 * opt.nz))
        #classfier.add_module('y1_leakyrelu', nn.LeakyReLU(0.2, inplace=True))
        #classfier.add_module('y2', nn.Linear(2 * opt.nz, opt.nz))
        #classfier.add_module('y2_leakyrelu', nn.LeakyReLU(0.2, inplace=True))
        #classfier.add_module('y3', nn.Linear(opt.nz, opt.num_classes))
        #classfier.add_module('softmax', nn.Softmax())

        self.linear1 = nn.Linear(opt.nz, opt.nz)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.linear2 = nn.Linear(opt.nz, int(opt.nz/2.0))
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.linear3 = nn.Linear(int(opt.nz/2.0), opt.num_classes)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.Softmax = nn.Softmax(dim=1)

        self.opt = opt
        #classfier.add_module(nn.Linear(opt.nz, 2 * opt.nz))
        #classfier.add_module(nn.LeakyReLU(0.2, inplace=True))
        #classfier.add_module(nn.Linear(2 * opt.nz, opt.nz))
        #classfier.add_module(nn.LeakyReLU(0.2, inplace=True))
        #classfier.add_module(nn.Linear(opt.nz, opt.num_classes))
        #classfier.add_module(nn.Softmax())

        #self.classfier = classfier

    def forward(self, input):

        y1=self.linear1(input.view(-1,self.opt.nz))
        y1=self.leaky_relu1(y1)
        y2=self.linear2(y1.view(-1,self.opt.nz))
        y2=self.leaky_relu2(y2)
        y3=self.linear3(y2.view(-1,int(self.opt.nz/2.0)))
        y3=self.leaky_relu3(y3)
        h = self.Softmax(y3)

        return h
        #if self.ngpu > 1:
        #    output = nn.parallel.data_parallel(self.classfier, input, range(self.ngpu))
        #else:
        #    output = self.classfier(input)
        #return output

##
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self, opt):
        super(NetG, self).__init__()
        self.encoder1 = Encoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)#nz是z的维度 ngf是channel数量
        self.decoder = Decoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)
        self.encoder2 = Encoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)
        self.z_mean = nn.Linear(opt.nz, opt.nz)  # 多加一层表示每个独立z的均值 可以不初始化
        self.z_log_var = nn.Linear(opt.nz, opt.nz)  # 多加一层表示每个独立z的方差 可以不初始化
        self.opt = opt
        #self.gaussian = self.Gaussian(self.opt)
        self.layers = list(self.encoder1.main.children())
        self.encoder1_part1 = nn.Sequential(*self.layers[0:2])
        #print("local_features=", self.encoder1_part1)
        self.encoder1_part2 = nn.Sequential(*self.layers[2:-1])
        self.encoder1_part3 = nn.Sequential(self.layers[-1])
        #self.encoder1_part2.add_module('final-conv',self.layers[-1])
        self.z_added_noise = None
        self.max_pooling = nn.MaxPool2d((4,4))

        self.layers2 = list(self.decoder.main.children())
        self.decoder_part1 = nn.Sequential(*self.layers2[0:-2])
        #print("self.decoder_part1=", self.decoder_part1)
        self.decoder_part2 = nn.Sequential(*self.layers2[-2:])
        #print("self.decoder_part2=", self.decoder_part2)

        #####self.Gaussian() = Gaussian()

    def forward(self, x):#u sigma 的训练 写在forward里面
        #latent_i = self.encoder1(x)
        local_features = self.encoder1_part1(x)

        features1 = self.encoder1_part2(local_features.view(-1, self.opt.ndf, int(self.opt.isize/2), int(self.opt.isize/2)))
        #print("global_features1.size()=", global_features1.size())
        global_features = self.max_pooling(features1)
        #print("global_features2.size()=", global_features2.size())
        latent_i = self.encoder1_part3(features1)
        #latent_i = self.encoder1_part2(features.view(-1, self.opt.ndf, int(self.opt.isize), int(self.opt.isize )))
        #print("global_features1=", global_features1)

        #print("layers=",self.layers)
        #print("layers1=", self.layers[0:2])
        #print("layers2=", self.layers[2:-1])

        #print("latent_i=",latent_i)
        #print("latent_i.shape=", latent_i.shape)
        #z_mean = self.z_mean(latent_i.view(-1,self.opt.nz))#隐变量编码是z_mean


        z_mean = self.z_mean(latent_i.view(-1, self.opt.nz)) # 隐变量编码是z_mean
        z_log_var = self.z_log_var(latent_i.view(-1,self.opt.nz))

        ############vae
        self.z_added_noise = sampling(z_mean, z_log_var, self.opt)
        ############

        ############kl cat
        ####z_mean = self.Gaussian(z_mean)
        ####self.z_added_noise = sampling(self.Gaussian.mean, 1, self.opt)
        ############

        #z_y = self.gaussian(z_added_noise)
        #y = self.main(z_added_noise)

        #####原始代码
        ###gen_imag = self.decoder(self.z_added_noise.view(-1,self.opt.nz,1,1))
        #####

        #####加了vae噪声
        local_features_decoder = self.decoder_part1(self.z_added_noise.view(-1, self.opt.nz, 1, 1))
        #####

        #####普通自编吗器
        #local_features_decoder = self.decoder_part1(latent_i.view(-1, self.opt.nz, 1, 1))
        #####


        gen_imag = self.decoder_part2(local_features_decoder.view(-1, self.opt.ngf, self.opt.isize // 2, self.opt.isize // 2))


        ##gen_imag = self.decoder(z_mean.view(-1, self.opt.nz, 1, 1))

        #latent_o = self.encoder2(gen_imag)
        #return gen_imag, latent_i, latent_o
        return gen_imag, z_mean, z_log_var, local_features, global_features, local_features_decoder #z_mean 用于逼近分类  z_mean和z_log_var

def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

class Net_Auto_D(nn.Module):


    def __init__(self, opt):
        super(Net_Auto_D, self).__init__()
        self.encoder = Encoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)#nz是z的维度 ngf是channel数量
        self.model = Decoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)

        self.layers = list(self.model.main.children())

        self.features = nn.Sequential(*self.layers[:-1])

        self.features.add_module('Sigmoid', nn.Sigmoid())


    def forward(self, x):
        latent_i = self.encoder(x)
        #gen_imag = self.decoder(latent_i)
        gen_imag_sigmoid = self.features(latent_i)

        return gen_imag_sigmoid, latent_i



class Encoder_without_batchnorm(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """
    #ndf是输出的channel个数
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0, add_final_conv=True):
        super(Encoder_without_batchnorm, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()# model模型
        # input is nc x isize x isize

        ##main.add_module('initial-conv',
        ##                nn.Conv2d(nc, ndf, 1, 1, 0, bias=False))  # （32+2×0-1）/1+1=32 #wgan-gp kernel是3
        ##main.add_module('initial-relu',
        ##                nn.LeakyReLU(0.2, inplace=True))

        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))# （32+2×1-4）/2+1=16 #wgan-gp kernel是3###第一个ndf是nc
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf# 图像的大小缩小两倍  channel数量不变 16对应64

        # Extra layers
        for t in range(n_extra_layers):#没有额外的卷积层
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            #main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
            #                nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4: # 图像大于4的话就继续
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            #main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
            #                nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2# channel 变为2倍
            csize = csize / 2 # 图像缩小两倍

        # state size. K x 4 x 4
        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                            nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))# 图像大小现在已经小于4了 (（3）+2×0-4）/2+1=1  nz=100

        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output


class Net_W_D(nn.Module):

    def __init__(self, opt):
        super(Net_W_D, self).__init__()
        self.model = Encoder_without_batchnorm(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)#nz是z的维度 ngf是channel数量
        #self.model = Encoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)  # nz是z的维度 ngf是channel数量
        self.layers = list(self.model.main.children())
        self.features = nn.Sequential(*self.layers[:-1])

        #self.linear = nn.Linear()
        self.classifier = nn.Sequential(self.layers[-1])  # nz = 1 (0): Conv2d(256, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
        #print("self.classifier=",self.classifier)
        self.linear1 = nn.Linear(opt.nz, opt.nz)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.linear2 = nn.Linear(opt.nz, opt.nz)
        self.leakt_relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.linear3 = nn.Linear(opt.nz, 1)
        ##self.linear = nn.Linear(opt.nz, 1)

        self.nz = opt.nz

    def forward(self, x):
        latent_i = self.features(x)
        #w_dis = self.classifier(latent_i)
        w_dis_tmp1 = self.classifier(latent_i)
        w_dis_tmp2 = self.linear1(w_dis_tmp1.view(-1, self.nz))
        w_dis_tmp2 = self.leaky_relu1(w_dis_tmp2)
        w_dis_tmp3 = self.linear2(w_dis_tmp2.view(-1, self.nz))
        w_dis_tmp3 = self.leakt_relu2(w_dis_tmp3)
        w_dis = self.linear3(w_dis_tmp3.view(-1, self.nz))

        ##w_dis_tmp = self.classifier(latent_i)
        ##w_dis = self.linear(w_dis_tmp.view(-1, self.nz))

        return w_dis, latent_i

class GlobalDiscriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        #self.c0 = nn.Conv2d(128, 64, kernel_size=3)
        ##self.c0 = nn.Conv2d(opt.ndf, 64, kernel_size=11)
        #self.c0 = nn.Conv2d(opt.ndf, 64, kernel_size=3)
        #self.c1 = nn.Conv2d(64, 32, kernel_size=3)

        #self.

        #self.l0 = nn.Linear(32 * 22 * 22 + 64, 512)
        #self.l0 = nn.Linear(25388, 512)
        #self.l0 = nn.Linear(4908, 512)
        ###self.l0 = nn.Linear(512 + opt.nz, 512)
        #####self.l0 = nn.Linear(2 * opt.nz, 512)
        #####self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)
        #self.l0 = nn.Linear(4808, 512)
        ##self.l0 = nn.Linear(2604, 512)
        #####self.l1 = nn.Linear(512, 512)
        #####self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)
        #####self.l2 = nn.Linear(512, 1)
        #####self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.l0 = nn.Linear(2 * opt.nz, 512)
        #self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)
        # self.l0 = nn.Linear(4808, 512)
        ##self.l0 = nn.Linear(2604, 512)
        self.l1 = nn.Linear(512, 512)
        #self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.l2 = nn.Linear(512, 1)
        #self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)


        self.opt = opt

    def forward(self, h):
        ##h = F.relu(self.c0(M))#进来的M是64层,不是128层  需要修改cndf和cngf为128层
        ##h = self.c1(h)
        ##h = h.view(y.shape[0], -1)
        ##h = torch.cat((y, h), dim=1)
        #print("y.size()=", y.size())
        #print("y_prime.size()=", y_prime.size())


        #h = F.relu(self.l0(h.view(-1, 512 + self.opt.nz)))
        h = F.relu(self.l0(h.view(-1, 2 * self.opt.nz)))
        h = F.relu(self.l1(h.view(-1, 512)))
        #h = self.leaky_relu1(self.l0(h.view(-1, 2 * self.opt.nz)))
        #h = self.leaky_relu2(self.l1(h.view(-1, 512)))
        #h = self.leaky_relu3(self.l2(h.view(-1, 512)))
        #return h
        return self.l2(h.view(-1,512))


class LocalDiscriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()#--ndf 128 --ngf 128
        #self.c0 = nn.Conv2d(192, 512, kernel_size=1)#修改ndf 128 channels + 300 channels nz =428
        self.c0 = nn.Conv2d(opt.ndf + opt.nz, 256, kernel_size=1)###原来是512
        self.c1 = nn.Conv2d(256, 256, kernel_size=1)
        self.c2 = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return self.c2(h)


class LocalDiscriminator_decoder(nn.Module):
    def __init__(self, opt):
        super().__init__()#--ndf 128 --ngf 128
        #self.c0 = nn.Conv2d(192, 512, kernel_size=1)#修改ndf 128 channels + 300 channels nz =428
        self.c0 = nn.Conv2d(opt.ngf + opt.nz, 512, kernel_size=1)
        self.c1 = nn.Conv2d(512, 512, kernel_size=1)
        self.c2 = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return self.c2(h)


class DeepInfoMaxLoss(nn.Module):
    def __init__(self, opt, alpha=0.5, beta=1.0, gamma=0.1):
        super().__init__()
        self.global_d = GlobalDiscriminator(opt)
        self.local_d = LocalDiscriminator(opt)
        self.prior_d = PriorDiscriminator(opt)

        self.local_d_decoder = LocalDiscriminator_decoder(opt)

        ###初始化 不一定要有！！！
        #self.global_d.apply(weights_init_info)
        #self.local_d.apply(weights_init_info)
        #self.prior_d.apply(weights_init_info)
        ###
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.opt=opt

    def forward(self, y, M, M_prime, global_features, global_features_prime, local_features_decoder, local_features_decoder_prime):#encoded features 旋转图片

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf

        y_exp = y.unsqueeze(-1).unsqueeze(-1)##在后面加一个维度[64 64 ]
        #y_exp = y_exp.expand(-1, -1, 26, 26)#为了拼接  26×26 featuremap 是 26×26
        ###y_exp = y_exp.expand(-1, -1, 32, 32) #我们的featuremap是16 复制
        y_exp = y_exp.expand(-1, -1, 16, 16)  # 我们的featuremap是16 复制
        ###y_exp = y_exp.expand(-1, -1, 8, 8)  # 我们的featuremap是16 复制
        #y_exp = y_exp.expand(-1, -1, 32, 32)  # 我们的featuremap是16 复制

        #print("M.size()=", M.size())
        #print("y_exp.size()=", y_exp.size())
        y_M = torch.cat((M, y_exp), dim=1)#encoded features 拼接
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)#encoded 旋转图片 拼接

        Ej = -F.softplus(-self.local_d(y_M)).mean()#局部 直接分类  用1×1卷积 用的是论文中的第一种结构
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        #LOCAL_FOOL = -F.softplus(self.local_d(y_M_prime)).mean()


        #####local features decoder
        y_local_features_decoder = torch.cat((local_features_decoder, y_exp), dim=1)  # encoded features 拼接
        y_local_features_decoder_prime = torch.cat((local_features_decoder_prime, y_exp), dim=1)  # encoded 旋转图片 拼接

        Ej_decoder = -F.softplus(-self.local_d_decoder(y_local_features_decoder)).mean()  # 局部 直接分类  用1×1卷积 用的是论文中的第一种结构
        Em_decoder = F.softplus(self.local_d_decoder(y_local_features_decoder_prime)).mean()
        LOCAL_decoder = (Em_decoder - Ej_decoder) * self.beta
        #####



        #Ej = -F.softplus(-self.global_d(y, M)).mean()#全局 对features和旋转图片再提特征 相当于用能够进行分类的图片特征
        # 因为是无监督 因此保留了图片是哪一类的信息   再拼接去分类
        #Em = F.softplus(self.global_d(y, M_prime)).mean()
        #GLOBAL = (Em - Ej) * self.alpha
        #print("global_features.view(-1, 512).size()=",global_features.view(-1, 512).size())
        #print("y.size()=", y.size())

        #####global features
        ###h1 = torch.cat((global_features.view(-1, 512), y), dim=1)
        ###h2 = torch.cat((global_features_prime.view(-1, 512), y), dim=1)
        #####

        ##### z features 利用y和y采样
        h1 = torch.cat((global_features.view(-1, self.opt.nz), y), dim=1)
        h2 = torch.cat((global_features_prime.view(-1, self.opt.nz), y), dim=1)
        #####

        #####global features
        Ej = -F.softplus(-self.global_d(h1)).mean()#全局 对features和旋转图片再提特征 相当于用能够进行分类的图片特征\
        #print("Ej.size()=",Ej.size())
        #print("-F.softplus(-self.global_d(y, y)).size()=", (-F.softplus(-self.global_d(y, y))).size())
        # 因为是无监督 因此保留了图片是哪一类的信息   再拼接去分类
        #h2 = torch.cat((Y_prime, y), dim=0)
        Em = F.softplus(self.global_d(h2)).mean()
        GLOBAL = (Em - Ej) * self.alpha
        #GLOBAL = 0
        #####

        #GLOBAL_FOOL = -F.softplus(self.global_d(y, M_prime)).mean()
        #GLOBAL_FOOL = 0


        #prior = torch.rand_like(y)
        #term_a = torch.log(self.prior_d(prior.view(-1,self.opt.nz))).mean()
        #term_b = torch.log(1.0 - self.prior_d(y.view(-1,self.opt.nz))).mean()

        #PRIOR = - (term_a + term_b) * self.gamma
        #PRIOR_FOOL =  torch.log(1 - self.prior_d(y.view(-1,self.opt.nz))).mean()

        #return LOCAL, GLOBAL, PRIOR, LOCAL_FOOL, GLOBAL_FOOL, PRIOR_FOOL

        return LOCAL, GLOBAL, LOCAL_decoder


#####原始
##        prior = torch.rand_like(y)
##
##        term_a = torch.log(self.prior_d(prior)).mean()
##        term_b = torch.log(1.0 - self.prior_d(y)).mean()
##        PRIOR = - (term_a + term_b) * self.gamma

##        return LOCAL + GLOBAL + PRIOR
#####

class PriorDiscriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        #self.l0 = nn.Linear(opt.nz, 1000)
        #self.l1 = nn.Linear(1000, 200)
        self.l0 = nn.Linear(opt.nz, 200)
        self.l1 = nn.Linear(200, 200)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


class D_net_gauss(nn.Module):
    def __init__(self,opt):
        super(D_net_gauss, self).__init__()
        self.N = 1000
        self.lin1 = nn.Linear(opt.nz, self.N)
        self.lin2 = nn.Linear(self.N, self.N)
        self.lin3 = nn.Linear(self.N, 1)
        self.opt = opt

    def forward(self, x):
        x = F.dropout(self.lin1(x.view(-1, self.opt.nz)), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x.view(-1, self.N)), p=0.2, training=self.training)
        x = F.relu(x)

        return F.sigmoid(self.lin3(x.view(-1, self.N)))