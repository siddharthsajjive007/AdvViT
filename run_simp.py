import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
import utils
import math
import random
import argparse
import os,logging
from os import listdir
from os.path import isfile,join
from simp import SimP
import torch.nn as nn
from models.DeiT import deit_base_patch16_224, deit_tiny_patch16_224, deit_small_patch16_224
from models.resnet import ResNet50, ResNet152, ResNet101
import cv2
from torch.nn import functional as F
import torchvision.transforms as T
import numpy as np
from utils import clamp, get_loaders
import timm
from config import get_config
from models import build_model
from data import build_loader
# from swin_transformer import SwinTransformer
#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda',0)
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#a = torch.cuda.is_available()
devices = torch.cuda.current_device()

trf = T.Compose([T.ToPILImage(),
				 T.ToTensor(),
				 T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs SimBA on a set of images')
    #parser.add_argument('--cfg', default='D:\\zc\\simple-patch-master2\\configs\\swin\\swin_base_patch4_window7_224.yaml',type=str, required=False, metavar="FILE", help='path to config file', )
    '''parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    ) '''   
    parser.add_argument('--data_root', type=str, default='D:\\zc\\simple-patch-master2', required=False, help='root directory of imagenet data')
    parser.add_argument('--result_dir', type=str, default='save', help='directory for saving results')
    parser.add_argument('--sampled_image_dir', type=str, default='save', help='directory to cache sampled images')
    parser.add_argument('--model', type=str, default='DeiT_T', help='type of base model to use')
    parser.add_argument('--num_runs', type=int, default=100, help='number of image samples')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for parallel runs')
    parser.add_argument('--max_iters', type=int, default=4000, help='maximum number of iterations, 0 for unlimited')
    parser.add_argument('--log_every', type=int, default=10, help='log every n iterations')
    parser.add_argument('--epsilon', type=float, default=0.2, help='step size per iteration')
    parser.add_argument('--linf_bound', type=float, default=0.0, help='L_inf bound for frequency space attack')
    parser.add_argument('--patch_size', type=int, default=16, help='size of patch')
    parser.add_argument('--order', type=str, default='rand', help='(random) order of coordinate selection')
    parser.add_argument('--stride', type=int, default=7, help='stride for block order')
    parser.add_argument('--targeted', action='store_true', help='perform targeted attack')
    parser.add_argument('--pixel_attack', action='store_true', help='attack in pixel space')
    parser.add_argument('--save_suffix', type=str, default='', help='suffi  BZcvxzVVVVVVVVVV ZXCx appended to save file')
    parser.add_argument('--workers', default=1, type=int)
    args = parser.parse_args()

    # config = get_config(args)          ####################

    # if not os.path.exists(args.result_dir):
    #     os.mkdir(args.result_dir)
    # if not os.path.exists(args.sampled_image_dir):
    #     os.mkdir(args.sampled_image_dir)

    # load model and dataset
    #model = getattr(models, args.model)(pretrained=True).cuda()
    if args.model == 'ResNet152':
        model = ResNet152(pretrained=True)
    elif args.model == 'ResNet50':
        model = ResNet50(pretrained=True)
    elif args.model == 'ResNet18':
        model = torchvision.models.resnet18(pretrained=True)  
    elif args.model == 'VGG16':
        model = torchvision.models.vgg16(pretrained=True)
    elif args.model == 'DeiT_T':
        model = deit_tiny_patch16_224(pretrained=True)
    elif args.model == 'DeiT_S':
        model = deit_small_patch16_224(pretrained=True)
    elif args.model == 'DeiT_B':
        model = deit_base_patch16_224(pretrained=True)
    
    elif args.model == 'Swin_B':
        pretrained_cfg = timm.models.create_model('swin_base_patch4_window7_224').default_cfg
        pretrained_cfg['file'] = 'D:\\zc\\simple-patch-master-plus\\swin_base_patch4_window7_224.pth'
        model = timm.models.create_model('swin_base_patch4_window7_224', pretrained=True, pretrained_cfg=pretrained_cfg)
    elif args.model == 'Swin_S':
        pretrained_cfg = timm.models.create_model('swin_small_patch4_window7_224').default_cfg
        pretrained_cfg['file'] = 'D:\\zc\\simple-patch-master-plus\\swin_small_patch4_window7_224.pth'
        model = timm.models.create_model('swin_small_patch4_window7_224', pretrained=True, pretrained_cfg=pretrained_cfg)
    elif args.model == 'Swin_T':
        pretrained_cfg = timm.models.create_model('swin_tiny_patch4_window7_224').default_cfg
        pretrained_cfg['file'] = 'D:\\zc\\simple-patch-master-plus\\swin_tiny_patch4_window7_224.pth'
        model = timm.models.create_model('swin_tiny_patch4_window7_224', pretrained=True, pretrained_cfg=pretrained_cfg)

    ###############################################################       
    elif args.model == 'ViT_B':
        model = timm.create_model('vit_base_patch32_224.augreg_in1k', pretrained=True)
        model = model.eval()

        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

    ##############################################################
    else:
        print('Wrong Network')
        raise

    model = model.to(device)
    #model = torch.nn.DataParallel(model)
    model.eval()

    if args.model.startswith('inception'):
        image_size = 299
        testset = dset.ImageFolder(args.data_root + '/val', utils.INCEPTION_TRANSFORM)
    else:
        image_size = 224
        #testset = dset.ImageFolder(args.data_root + '/val', utils.IMAGENET_TRANSFORM)
    attacker = SimP(model, 'imagenet', image_size)

    #datasetfile = os.path.join(args.data_root,'dataset')

    #image_list =  [f for f in listdir(datasetfile) if isfile(join(datasetfile,f))]
    patch_num = int(image_size / args.patch_size)
    patch_num = patch_num * patch_num

    #loader = get_loaders(args)
    #############################################################
    #batchfile = '%s/images_%s_%d.pth' % (args.sampled_image_dir, args.model, args.num_runs)
    batchfile = 'save\\images_DeiT_T_1000.pth'
    if os.path.isfile(batchfile):
        checkpoint = torch.load(batchfile)
        images = checkpoint['images']
        labels = checkpoint['labels']
    else:
        images = torch.zeros(args.num_runs, 3, image_size, image_size)
        labels = torch.zeros(args.num_runs).long()
        preds = labels + 1
        while preds.ne(labels).sum() > 0:
            idx = torch.arange(0, images.size(0)).long()[preds.ne(labels)]
            for i in list(idx):
                images[i], labels[i] = testset[random.randint(0, len(testset) - 1)]
            preds[idx], _ = utils.get_preds(model, images[idx], 'imagenet', batch_size=args.batch_size)
        torch.save({'images': images, 'labels': labels}, batchfile)
    #############################################################

    N = int(math.floor(float(args.num_runs) / float(args.batch_size)))

    total_r_count = 0
    total_clean_count = 0
    total_distance = 0
    rays_successes = []
    successes = []
    stop_queries = [] # wrc added to match RayS
    distances_list = []
    success_thold = 5.0
    success_thold3 = 3.0
    success_thold8 = 8.0        
    attack_num = 0
    success_num = 0

    under_thold = 0
    under_thold3 = 0
    under_thold8 = 0
    #for k, (X, Y) in enumerate(loader):
    for i in range(N):
        upper = min((i + 1) * args.batch_size, args.num_runs)
        images_batch = images[(i * args.batch_size):upper].to(device)
        labels_batch = labels[(i * args.batch_size):upper].to(device)

        adv, distortion, is_success, nqueries, prub = attacker.simp_batch(
            images_batch, labels_batch, args.patch_size, args.max_iters, attack_num, epsilon=args.epsilon, linf_bound=args.linf_bound,
            order=args.order, targeted=args.targeted, pixel_attack=args.pixel_attack, log_every=args.log_every)
        attack_num = attack_num+1
        if is_success :  #'''and nqueries !=0'''
            stop_queries.append(nqueries)
            distances_list.append(distortion)
            print('agv distorn{}'.format(np.mean(distances_list)))
            print('median distorn{}'.format(np.median(distances_list))) 
            success_num +=1
            if distortion <= success_thold:
                under_thold +=1
            if distortion <= success_thold3:
                under_thold3 +=1
            if distortion <= success_thold8:
                under_thold8 +=1                                
            print('success{}'.format(under_thold /success_num))
            print('success3{}'.format(under_thold3 /success_num))
            print('success8{}'.format(under_thold8 /success_num))

    avg_success = success_num / attack_num
    logging.info(f"clean count:{total_clean_count}")
    logging.info(f"acc under attack count:{total_r_count}")
    logging.info(f"avg distortion:{np.mean(distances_list)}")
    logging.info(f"L2 under_thold rate:{under_thold / success_num}")
    logging.info(f"avg stop queries used:{np.mean(stop_queries)}")
    logging.info(f"avg success:{avg_success}")
    savefile = '%s.pth' % (args.model)
    torch.save({'succs': under_thold /success_num, 'queries': stop_queries,'l2_norms': distances_list}, savefile)
    print(distances_list)
    print(np.median(distances_list))
    print('success num{}'.format(success_num))    
