# CVAE-GAN-BASED Anomaly Detection

This repository contains PyTorch implementation of the following paper: A Novel and Efficient CVAE-GAN-BASED Approach with Informative Manifold for Semi-Supervised Anomaly Detection

## Prerequisites
1. Linux
2. Python 2 or 3
3. CPU or GPU + CUDA & CUDNN

## Installation
1. First clone the repository
   ```
   git clone git@github.com:bianjiang1234567/CVAE-GAN-BASED-Anomaly-Detection.git
   ```
2. Install PyTorch and torchvision from [https://pytorch.org](https://pytorch.org/)
3. Install the dependencies.
   ```
   pip install -r requirements.txt
   ```
**UPDATE**: This repository now supports PyTorch v0.4. If you still would like to work with v0.3, you could use the branch named PyTorch.v0.3, which contains the previous version of the repo.

## Experiment

To replicate the results in the paper, run the following commands:

For CIFAR experiments:
```  
python ../train.py --dataset cifar10 --isize 32 --nc 3 --nz=300 --niter 100 --anomaly_class "plane" --manualseed 0 --display --save_test_images --ndf=128 --ngf=128 --gpu_ids 1
```
plane is the anomaly class, can be replaced by: car, bird, cat, deer, dog, frog, horse, ship, truck.
To obtain better results, the weights of anomaly score can be adjusted.

## Training
To list the arguments, run the following command:
```
python train.py -h
```

### Train on Custom Dataset
To train the model on a custom dataset, the dataset should be copied into `./data` directory, and should have the following directory & file structure:

```
Custom Dataset
├── test
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_n.png
│   ├── 1.abnormal
│   │   └── abnormal_tst_img_0.png
│   │   └── abnormal_tst_img_1.png
│   │   ...
│   │   └── abnormal_tst_img_m.png
├── train
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_t.png

```

For more training options, run `python train.py -h` as shown below:
```
usage: train.py [-h] [--dataset DATASET] [--dataroot DATAROOT]
                [--batchsize BATCHSIZE] [--workers WORKERS] [--droplast]
                [--isize ISIZE] [--nc NC] [--nz NZ] [--ngf NGF] [--ndf NDF]
                [--extralayers EXTRALAYERS] [--gpu_ids GPU_IDS] [--ngpu NGPU]
                [--name NAME] [--model MODEL]
                [--display_server DISPLAY_SERVER]
                [--display_port DISPLAY_PORT] [--display_id DISPLAY_ID]
                [--display] [--outf OUTF] [--manualseed MANUALSEED]
                [--anomaly_class ANOMALY_CLASS] [--print_freq PRINT_FREQ]
                [--save_image_freq SAVE_IMAGE_FREQ] [--save_test_images]
                [--load_weights] [--resume RESUME] [--phase PHASE]
                [--iter ITER] [--niter NITER] [--beta1 BETA1] [--lr LR]
                [--alpha ALPHA]

optional arguments:
  -h, --help            show this help message and exit
  --dataset             folder | cifar10 | mnist (default: cifar10)
  --dataroot            path to dataset (default: '')
  --batchsize           input batch size (default: 64)
  --workers             number of data loading workers (default: 8)
  --droplast            Drop last batch size. (default: True)
  --isize               input image size. (default: 32)
  --nc                  input image channels (default: 3)
  --nz                  size of the latent z vector (default: 100)
  --ngf                 Number of features of the generator network
  --ndf                 Number of features of the discriminator network.
  --extralayers         Number of extra layers on gen and disc (default: 0)
  --gpu_ids             gpu ids: e.g. 0 0,1,2, 0,2. use -1 for CPU (default: 0)
  --ngpu                number of GPUs to use (default: 1)
  --name                name of the experiment (default: experiment_name)
  --model               chooses which model to use. (default:ganomaly)
  --display_server      visdom server of the web display (default: http://localhost)
  --display_port        visdom port of the web display (default: 8097)
  --display_id          window id of the web display (default: 0)
  --display             Use visdom. (default: False)
  --outf                folder to output images and model checkpoints (default: ./output)
  --manualseed          manual seed (default: None)
  --anomaly_class       Anomaly class idx for mnist and cifar datasets (default: car)
  --print_freq          frequency of showing training results on console (default: 100)
  --save_image_freq     frequency of saving real and fake images (default:100)
  --save_test_images    Save test images for demo. (default: False)
  --load_weights        Load the pretrained weights (default: False)
  --resume              path to checkpoints (to continue training) (default: '')
  --phase               train, val, test, etc (default: train)
  --iter                Start from iteration i (default: 0)
  --niter               number of epochs to train for (default: 15)
  --beta1               momentum term of adam (default: 0.5)
  --lr                  initial learning rate for adam (default: 0.0002)
  --alpha               alpha to weight l1 loss. default=500 (default: 50)

```

![Experimental results in CIFAR10 dataset. (a) Input samples of normal classes in testing set. Deers are designated as anomalous class. (b) Reconstruction results of (a). (c) Input samples of abnormal class deer in testing set. (d) Reconstruction results of (c).](https://github.com/bianjiang1234567/CVAE-GAN-BASED-Anomaly-Detection/blob/master/cifarreconstruction.png)

## Reference
ArXiv
