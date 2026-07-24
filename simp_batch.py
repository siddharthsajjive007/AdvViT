import torch
import torch.nn.functional as F
from skimage import io, color,data,exposure,img_as_float
import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from numpy import linalg as LA


DATASET = "IMAGENET_3599"          # "CIFAR" | "IMAGENET" | "GTSRB" | "IMAGENET_3599"

# mean and std for different datasets
IMAGENET_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()])

INCEPTION_SIZE = 299
INCEPTION_TRANSFORM = T.Compose([
    T.Resize(342),
    T.CenterCrop(299),
    T.ToTensor()])

CIFAR_SIZE = 32
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
CIFAR_TRANSFORM = T.Compose([
    T.ToTensor()])

MNIST_SIZE = 28
MNIST_MEAN = [0.5]
MNIST_STD = [1.0]
MNIST_TRANSFORM = T.Compose([
    T.ToTensor()])

GTSRB_SIZE = 32
GTSRB_MEAN = [0.3403, 0.3121, 0.3214]
GTSRB_STD  = [0.2724, 0.2608, 0.2669]
GTSRB_TRANSFORM = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
])

# Single source of truth for everything dataset-dependent: native resolution,
# normalization stats, and DCT-attack geometry (patch_size, dimen_size).
# patch_size: kept at 16 for ImageNet (matches ViT tokenization, paper-tuned).
#   For smaller-resolution datasets, scaled down so patch COUNT stays
#   reasonably fine-grained (e.g. 32/4=8x8=64 patches, vs 224/16=14x14=196).
# dimen_size: the low-frequency corner size (r). Scaled to preserve the same
#   ratio rho=r/d=4/16=0.25 used elsewhere in this file, rounded to at least 1.
#   This matters: dimen_size must be < patch_size, or the "low-frequency
#   restriction" covers the whole patch and dimension reduction does nothing.
DATASET_CONFIGS = {
    'IMAGENET':      {'size': IMAGENET_SIZE, 'mean': IMAGENET_MEAN, 'std': IMAGENET_STD, 'patch_size': 16, 'dimen_size': 4},
    'IMAGENET_3599': {'size': IMAGENET_SIZE, 'mean': IMAGENET_MEAN, 'std': IMAGENET_STD, 'patch_size': 16, 'dimen_size': 4},
    'CIFAR':         {'size': CIFAR_SIZE,    'mean': CIFAR_MEAN,    'std': CIFAR_STD,    'patch_size': 4,  'dimen_size': 1},
    'MNIST':         {'size': MNIST_SIZE,    'mean': MNIST_MEAN,    'std': MNIST_STD,    'patch_size': 4,  'dimen_size': 1},
    'GTSRB':         {'size': GTSRB_SIZE,    'mean': GTSRB_MEAN,    'std': GTSRB_STD,    'patch_size': 4,  'dimen_size': 1},
}


# trf = T.Compose([T.ToPILImage(),
# 				 T.ToTensor(),
# 				 T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
unloader = T.ToPILImage()

device = torch.device('cuda',0)

# Base directory for all saved artifacts (ori/adv/prub images).
# Fixed from the original hardcoded 'D:\\zc\\simple-patch-master-plus\\save\\...'
# which only existed on the authors' Windows machine.
SAVE_BASE = '/home/siddarth/AdvViT/save'

# applies the normalization transformations



class SimP:
    
    def __init__(self, model, dataset, image_size=None, k=200):
        if dataset not in DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Expected one of: {list(DATASET_CONFIGS.keys())}."
            )
        expected_size = DATASET_CONFIGS[dataset]['size']
        if image_size is None:
            image_size = expected_size
        elif image_size != expected_size:
            print(
                f"[SimP] Warning: image_size={image_size} passed but dataset '{dataset}' "
                f"natively expects {expected_size}. Proceeding with image_size={image_size}, "
                f"but the mean/std normalization stats for '{dataset}' were computed at "
                f"{expected_size} -- results may be slightly off if your checkpoint expects "
                f"a genuinely different resolution than what it was normalized for."
            )
        self.model = model
        self.k = k
        self.dataset = dataset
        self.image_size = image_size
        self.model.eval()
    
    def expand_vector(self, x, size):
        batch_size = x.size(0)
        x = x.view(-1, 3, size, size)
        z = torch.zeros(batch_size, 3, self.image_size, self.image_size)
        z[:, :, :size, :size] = x
        return z
        
    def apply_normalization(self, imgs, dataset):
        if dataset not in DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset '{dataset}' -- no config defined. "
                f"Expected one of: {list(DATASET_CONFIGS.keys())}."
            )
        mean = DATASET_CONFIGS[dataset]['mean']
        std = DATASET_CONFIGS[dataset]['std']
        imgs_tensor = imgs.clone()
        if dataset == 'MNIST':
            imgs_tensor = (imgs_tensor - mean[0]) / std[0]
        else:
            if imgs.dim() == 3:
                for i in range(imgs_tensor.size(0)):
                    imgs_tensor[i, :, :] = (imgs_tensor[i, :, :] - mean[i]) / std[i]
            else:
                for i in range(imgs_tensor.size(1)):
                    imgs_tensor[:, i, :, :] = (imgs_tensor[:, i, :, :] - mean[i]) / std[i]
        return imgs_tensor
        
    def normalize(self, x):
        return self.apply_normalization(x, self.dataset)


    def _forward_logits(self, x):
        """
        Runs the model and returns just the classification logits.

        Handles both model families used in this repo:
          - ViT-style models (DeiT via models/vision_transformer.py) whose
            forward() returns a (logits, attention_list) tuple.
          - CNN-style models (models/resnet.py) whose forward() returns a
            plain logits Tensor.

        The original code assumed every model returns a 2-tuple
        (`class_prob, atten = self.model(x)`), which breaks for ResNet
        (forward() returns a single Tensor there) either with a
        "too many values to unpack" error, or silently mis-splitting a
        batch if batch size happens to equal 2. This makes the unpacking
        conditional on what the model actually returns.
        """
        output = self.model(x)
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def get_results(self, x):
        x = self.normalize(x.to(device))
        class_prob = self._forward_logits(x)
        img_class = class_prob.max(1)[1]
        prob = torch.max(torch.softmax(class_prob[0], dim=0))
        
        return img_class, prob

    def get_label(self, x):
        x = self.normalize(x.to(device))
        class_prob = self._forward_logits(x)
        img_class = class_prob.max(1)[1]
        return img_class


    def generate_adv(self, x0, patch_num,dct_theta):
        x = np.array(x0.clone())                        #[3,image_size,image_size]
        x_dct = self.DCT_trans(x, patch_num)
        adv = self.IDCT_trans(x_dct + dct_theta,patch_num)     #IDCT of x_0(dct) + variance, alpha scaled gaussian direction [3,image_size,image_size]
        patch_size = int(self.image_size/patch_num)                         #pixel length of each patch
        if self.image_size % patch_num !=0:
            adv[:,0:self.image_size,patch_num*patch_size:self.image_size] = x0[:,0:self.image_size,patch_num*patch_size:self.image_size]
            adv[:,patch_num*patch_size:self.image_size,0:self.image_size] = x0[:,patch_num*patch_size:self.image_size,0:self.image_size]     #if ori length is not divisible exactly by patch length , the remaining length or height values are populated from the original image
        #adv = self.IDCT_trans(x_dct,patch_num)
        adv = torch.tensor(adv, dtype=torch.float)
        adv = adv.clamp(0, 1)
        #adv = adv.unsqueeze(0)
        prub = adv - x0                                           #[3,image_size,image_size]
        return adv,prub                                      # returns the adversarial image and perturbation made


    def simp_batch(self, images_batch, labels_batch, patch_size, max_iters, num, epsilon=0.0, linf_bound=0.0,
                    order='rand', targeted=False, pixel_attack=False, log_every=1):
        batch_size = images_batch.size(0)
        image_size = images_batch.size(2)

        patch_num = int(image_size / patch_size)
        #patch_num = patch_num * patch_num

        # Fixed: these used to be hardcoded to the authors' Windows machine
        # ('D:\\zc\\simple-patch-master-plus\\save\\...'), which don't exist
        # on WSL and would crash on the first .save() call. Now they live
        # under the repo's own save/ directory and are created if missing.
        ori_path = os.path.join(SAVE_BASE, 'ori')
        adv_path = os.path.join(SAVE_BASE, 'adv')
        prub_path = os.path.join(SAVE_BASE, 'prub')
        os.makedirs(ori_path, exist_ok=True)
        os.makedirs(adv_path, exist_ok=True)
        os.makedirs(prub_path, exist_ok=True)

        for n in range(batch_size):
            ##################save original image#######################
            image_ori = images_batch[n].cpu().clone()
            image_ori = image_ori.squeeze(0)
            image_ori = unloader(image_ori)
            #plt.imshow(image_ori)
            image_ori.save(os.path.join(ori_path,'ori{}.jpg'.format(num)))
            ############################################################
            #mask = np.zeros([image_size,image_size])
            ori_class, ori_prob = self.get_results(images_batch[n].unsqueeze(0))
            adv, distortion, is_success, nqueries, prub = self.attack_untargeted(images_batch[n],labels_batch[n], ori_prob, patch_num, query_limit=max_iters)
            
            adv = adv.clamp(0, 1)
            adv = adv.squeeze(0)
            prub = prub.squeeze(0)

            distortion = torch.norm(adv - images_batch[n])
            distortion = distortion.cpu().numpy()
            adv = unloader(adv)
            prub = unloader(prub)            
            if is_success == True:
                adv.save(os.path.join(adv_path,'adv{}_{}.jpg'.format(num,nqueries)))
                prub.save(os.path.join(prub_path,'prub{}_{}.jpg'.format(num,nqueries)))

        return adv, distortion, is_success, nqueries, prub
    
    def attack_untargeted(self, x0, y0, ori_probal, patch_num, alpha = 0.2, beta = 0.001, iterations = 1000, query_limit=4000,
                          distortion=None, svm=False, momentum=0.0, stopping=0.0001, use_sign_opt_plus=False):
        """
        use_sign_opt_plus=False (default): plain Sign-OPT line search (AdvViT / "AD").
        use_sign_opt_plus=True: adds the Sign-OPT+ gate (Ran & Wang 2022) before each
                                binary search in the line-search loop -- one cheap query at the current
                                best DCT-scale (gg_dct) checks whether a candidate direction even succeeds
                                before paying for the full fine_grained_binary_search_local call. This is
                                the AdvViT+ / "AD+" variant.

        """
    
        model = self.model
        query_count = 0
        ls_total = 0

        num_directions = 100
        dimen_size = DATASET_CONFIGS[self.dataset]['dimen_size']   # scaled per-dataset -- see DATASET_CONFIGS comment;
                                                                     # must stay < patch_size or dimension reduction does nothing
        alp = 4  #alpha (scaling factor) -- not resolution-dependent, same across datasets
        success_flag = False
        best_theta_dct, g_theta, g_dct  = None, float('inf'), float('inf') 
        # g_dct is the lambda value

        print("Searching for the initial direction on %d random directions: " % (num_directions))
         
        #####################################################################
        patch_size = int(self.image_size/patch_num)                                 #int
        dct_mask = self.Mask_weight(x0, dimen_size,alp,patch_size)  
        print(f"Dct mask:{dct_mask.shape}")                             #[1, 3, image_size, image_size]
        dct_mask = dct_mask.squeeze(0)                                  #[3, image_size, image_size]
        #####################################################################
        for i in range(num_directions):
            query_count += 1
            x0 = x0.cpu().clone()                                       #[3, image_size, image_size]
            dct_theta = np.random.randn(*np.array(x0).shape)            # gaussian distortion [3, image_size, image_size]
            dct_theta = dct_theta * dct_mask                            # (theta' = theta * mask) gaussian direction mask scaled with variance and alpha and allows only top left corner for each patch 
            
            adv, prub= self.generate_adv(x0,patch_num,dct_theta)                   #returns adversarial image by doing idct (x_0(dct) + dct theta)            # register adv directions
            adv_class, adv_prob = self.get_results(adv.unsqueeze(0).to(device))    #class number and softmax probability
            if adv_class != y0:                                                    # if adversarial (NOT ORIGINAL LABEL)
                success_flag = True
                initial_lbd_dct = LA.norm(dct_theta)                    #magnitude of the random direction (DCT space)
                initial_l2 = LA.norm(prub)                              #magnitude of actual distance between ori and adv

                prub /= initial_l2                                      # normalized distance vector b/w ori and adv
                dct_theta /= initial_lbd_dct                            # (theta' / ||theta'||)  dct l2 normalized
                lbd_dct, lbd_l2, count = self.fine_grained_binary_search(model, x0, y0, patch_num, dct_theta, initial_l2,  initial_lbd_dct, g_theta, g_dct)
                #RETURNS CONVERGED BEST LBD_DCT (LAMBDA) AND L2 OF THE DIRECTION THAT HAS HAD BETTER RAW NORM THAN ALL THE PREVIOUS SELECTED DIRECTION
                """
                CONVERGED HERE MEANS THE DIRECTION WHICH IS OPTIMIZED ON LAMBDA BY BINARY SEARCH (LOWEST LAMBDA VALUE)
                """
                query_count += count   
                if lbd_l2 < g_theta:                                   #CHECKS IF THE NEW CONVERGED POINT HAS LOWER RAW NORM THAN EARLIER BEST CONVERGED DIRECTION
                    g_theta, best_theta_dct, g_dct  = lbd_l2, dct_theta, lbd_dct     #IF YES, THIS BECOMES THE BEST DIRECTION WITH BEST CONVERGED LAMBDA VALUE
                    print("--------> Found l2distortion %.4f,dctdistortion%.4f" % (g_theta, g_dct))
                    if round(g_theta, 3) == 0.000:
                        print("not need to be attacked")
                        return x0.to(device), 0, True, query_count, torch.zeros(3,self.image_size,self.image_size)    
            
            
        ###############################################################
        if success_flag == False:                            #(FALLBACK) IF NONE OF THE GENERATED DIRECTION FOOLS THE MODEL
            for j in range(patch_num):
                for k in range(patch_num):
                    dct_mask[:,j*patch_size:j*patch_size+dimen_size,k*patch_size:k*patch_size+dimen_size] = 1      #SCRAPS THE ORIGINAL VARIANCE WEIGHTED MASK AND GIVES A SIMPLE 1 MASK
            for i in range(query_limit - query_count):
                query_count += 1
                dct_theta = np.random.randn(*np.array(x0).shape) # gaussian distortion
                dct_theta = dct_theta * dct_mask                                         

                adv, prub= self.generate_adv(x0,patch_num,dct_theta)
                adv_class, adv_prob = self.get_results(adv.unsqueeze(0).to(device))    #DOES THE SAME PROCESS AGAIN THIS TIME WITH A SIMPLE MASK 
                if adv_class != y0:
                    success_flag = True
                    initial_lbd_dct = LA.norm(dct_theta)
                    initial_l2 = LA.norm(prub)

                    prub /= initial_l2 # l2 normalize                            
                    dct_theta /= initial_lbd_dct                                        #SAME AS ABOVE
                    lbd_dct, lbd_l2, count = self.fine_grained_binary_search(model, x0, y0, patch_num, dct_theta, initial_l2,  initial_lbd_dct, g_theta, g_dct)

                    query_count += count
                    if lbd_l2 < g_theta:
                        g_theta, best_theta_dct, g_dct  = lbd_l2, dct_theta, lbd_dct
                        print("--------> Found l2distortion %.4f,dctdistortion%.4f" % (g_theta, g_dct))
                    break                   
            ##########################################################

        # fail if cannot find a adv direction within 200 Gaussian
        if g_theta == float('inf'):
            print("Couldn't find valid initial, failed")
            return x0.to(device), 0, False, query_count, torch.zeros(3,self.image_size,self.image_size)
        if round(g_theta, 3) == 0.000:
            print("not need to be attacked")
            return x0.to(device), 0, True, query_count, torch.zeros(3,self.image_size,self.image_size)
        print("==========> Found best distortion %.4f"
              "using %d queries" % (g_theta,query_count))

        #### Begin Gradient Descent.
        xg, gg, gg_dct = best_theta_dct, g_theta, g_dct#    #xg IS THE BEST (THETA'/||THETA'||), gg IS THE BEST RAW NORM (PRUB DISTANCE) AND gg_dct IS THE BEST LAMBDA FOR THE xg (LOWEST)
        distortions = [gg]
        for i in range(iterations):                                                                                     # EACH ITERATION CREATES DIFFERENT SIGN GRADIENT
            sign_gradient, grad_queries = self.sign_grad_v1(x0, y0, patch_num, xg, gg_dct, dct_mask, h=beta)            # RETURNS AVERAGE OF ALL mu(j) THAT NUDGED THETA' AND WAS ADVERSARIAL
            ls_count = 0
            min_theta_dct = xg             ## NEXT THETA'
            min_g2 = gg                    ## CURRENT RAW NORM
            min_lbd_dct = gg_dct           ## LAMBDA
            for _ in range(5):
                new_theta_dct = xg - alpha * sign_gradient            #UPDATED THETA' IN EVERY INNER LOOP BY CHANGING ALPHA BY ADDING ALPHA SCALED mu(j) average (NEW THETA')
                new_theta_dct /= LA.norm(new_theta_dct)               # NEW THETA' /|| NEW THETA'||  

                if use_sign_opt_plus:
                    # Sign-OPT+ gate: one cheap query at the CURRENT best-in-this-attempt-sequence
                    # DCT-scale (min_lbd_dct, which tightens as better candidates are found within
                    # this inner loop) before paying for fine_grained_binary_search_local's full
                    # binary search. Using gg_dct here (fixed for the whole outer iteration) was a
                    # bug -- it never tightened, weakening the efficiency gain within each iteration.
                    adv_check, _ = self.generate_adv(x0, patch_num, min_lbd_dct * new_theta_dct)      # GENERATES ADV IMAGE WITH NEW THETA' (LAMBDA OF THE PREVIOUS SELECTED THETA DCT IS TRIED WITH NEW THETA DCT (NEW ALPHA)... SINCE THE NEW ALPHA IS JUST SCALING THE BASE DIRECTION, IT SHOULD WORK, SO THIS GATE IS A CHEAP FILTER AGAIN)
                    check_class, _ = self.get_results(adv_check.unsqueeze(0).to(device))              # PREDICTS CLASS
                    ls_count += 1
                    if check_class != y0:                                                             # IF ADVERSARIAL        
                        new_lbd_dct, count = self.fine_grained_binary_search_local(                   # DO FINE GRAINED BINARY SEARCH HERE WITH THE INITIAL LAMBDA VALUE, NOT PREVIOUS LIKE ABOVE AS WE WANT TO FIND THE BEST CONVERGED LAMBDA TO THIS NEW THETA DCT, SO INITIAL LAMBDA VALUE
                            x0, y0, patch_num, new_theta_dct,  initial_lbd_dct = gg_dct, tol=beta/500)
                        ls_count += count
                        adv, prub= self.generate_adv(x0,patch_num,new_lbd_dct*new_theta_dct)          # GENERATE ADV IMAGE WITH NEW VALUES OF LAMBDA AND THETA DCT
                        new_g2 = LA.norm(prub)                                                        # NEW RAW NORM
                    else:
                        new_lbd_dct, new_g2, count = min_lbd_dct, float('inf'), 0                     #IF NOT ADV, RAW NORM IS SET TO INFINITY
                else:
                    new_lbd_dct, count = self.fine_grained_binary_search_local(
                        x0, y0, patch_num, new_theta_dct,  initial_lbd_dct = gg_dct,tol=beta/500)     # THIS ELSE BLOCK IS FOR SIGN-OPT INSTEAD OF PLUS
                    ls_count += count
                    adv, prub= self.generate_adv(x0,patch_num,new_lbd_dct*new_theta_dct)
                    new_g2 = LA.norm(prub)

                alpha = min(alpha * 2, 1e6) # CORRECTION MADE FROM ORIGINAL FILE                      # ALPHA IS INCREASED BY 2 BUT NOT BEYOND 1E6 (1 MILLION)
                                            # gradually increasing step size, capped to prevent
                                            # overflow to inf on long runs of consecutive successes
                                            # (unbounded doubling eventually overflows float precision,
                                            # producing nan candidates and permanently disabling the
                                            # alpha<1e-4 reset since inf is never < 1e-4)
                if new_g2 < min_g2:
                    min_theta_dct = new_theta_dct
                    min_lbd_dct = new_lbd_dct
                    min_g2 = new_g2                                    #UPDATES THE MIN LBD DCT TO CONVERGED LAMBDA FOR THE NEW THETA DCT THAT PRODUCES AN ADV IMAGE HAVING LESS RAW NORM THAN PREV SELECTED THETA DCT
                else:
                    break

            if min_g2 >= gg:                                            ## if the above code failed for the init alpha, we then try to decrease alpha
                for _ in range(5):                                      #  THIS BLOCK DOES THE SAME THING AS ABOVE , BUT ONLY ACTIVATES IF THE FIRST ADV CHECK FOR FIRST ALPHA FAILS AND THE NEW G2 IS SET TO INFINITY (IN THAT CASE NEW MIN G2 IS UNCHANGED, THAT IS WHY >= IS CHECKED HERE)                                    
                    alpha = alpha * 0.25
                    new_theta_dct = xg - alpha * sign_gradient
                    new_theta_dct /= LA.norm(new_theta_dct)

                    if use_sign_opt_plus:
                        adv_check, _ = self.generate_adv(x0, patch_num, min_lbd_dct * new_theta_dct)
                        check_class, _ = self.get_results(adv_check.unsqueeze(0).to(device))
                        ls_count += 1
                        if check_class != y0:
                            new_lbd_dct, count = self.fine_grained_binary_search_local(
                                x0, y0, patch_num, new_theta_dct,  initial_lbd_dct = gg_dct,tol=beta/500)
                            ls_count += count
                            adv, prub= self.generate_adv(x0,patch_num,new_lbd_dct*new_theta_dct)
                            new_g2 = LA.norm(prub)
                        else:
                            new_lbd_dct, new_g2, count = min_lbd_dct, float('inf'), 0
                    else:
                        new_lbd_dct, count = self.fine_grained_binary_search_local(
                            x0, y0, patch_num, new_theta_dct,  initial_lbd_dct = gg_dct,tol=beta/500)
                        ls_count += count
                        adv, prub= self.generate_adv(x0,patch_num,new_lbd_dct*new_theta_dct)
                        new_g2 = LA.norm(prub)

                    if new_g2 < gg:
                        min_theta_dct = new_theta_dct
                        min_lbd_dct = new_lbd_dct
                        min_g2 = new_g2
                        break

            if alpha < 1e-4:  ## if the above two blocks of code failed
                alpha = 1.0
                print("Warning: not moving")
                beta = beta*0.1
                if (beta < 1e-8):
                    break
            
            ## if all attemps failed, min_theta, min_g2 will be the current theta (i.e. not moving)
            xg, gg, gg_dct = min_theta_dct, min_g2, min_lbd_dct
            query_count += (grad_queries + ls_count)
            ls_total += ls_count
            distortions.append(gg)
            if query_count > query_limit:
                break

            if (i + 1) % 10 == 0:
                print("Iteration %3d distortion %.4f num_queries %d" % (i+1, gg, query_count))
        print("--------> sign opt attack  %.4f" % gg)
        adv, prub= self.generate_adv(x0,patch_num,gg_dct*xg)

        return adv.unsqueeze(0).to(device), gg, True, query_count, adv-x0              # RETURNS FINAL ADV IMAGE, RAW NORM, QUERY COUNT AND PRUB

    def fine_grained_binary_search_local(self, x0, y0,patch_num, theta_dct, initial_lbd_dct = 1.0, tol=5e-3):
        nquery = 0
    
        lbd_dct = initial_lbd_dct
        adv, _ = self.generate_adv(x0,patch_num,lbd_dct*theta_dct)
        adv_class = self.get_label(adv.unsqueeze(0).to(device))  

        if adv_class == y0:
            #lbd_dct_lo = lbd_dct
            #lbd_dct_hi = lbd_dct*1.01
            nquery += 1
            return float('inf'), nquery
            '''adv, prub = self.generate_adv(x0,patch_num,lbd_dct_hi*theta_dct)
            while self.get_label(adv.unsqueeze(0).to(device)) == y0:
                lbd_dct_hi = lbd_dct_hi*1.01
                nquery += 1
                adv, prub = self.generate_adv(x0,patch_num,lbd_dct_hi*theta_dct)'''
        else:
            lbd_dct_hi = lbd_dct
            lbd_dct_lo = lbd_dct*0.99
            nquery += 1
            adv, prub = self.generate_adv(x0,patch_num,lbd_dct_lo*theta_dct)
            while self.get_label(adv.unsqueeze(0).to(device)) != y0 :                      #THIS WHILE LOOP FINDS THE BEST BRACKET TP START FOR THE ACTUAL BINARY SEARCH (LOWEST LBD_DCT_LOW THAT JUST FAILS THE ADV CHECK)
                lbd_dct_lo = lbd_dct_lo*0.99
                nquery += 1
                adv, prub = self.generate_adv(x0,patch_num,lbd_dct_lo*theta_dct)
                if lbd_dct_lo < 1e-4:
                    break

        while (lbd_dct_hi - lbd_dct_lo) > tol:
            lbd_dct_mid = (lbd_dct_lo + lbd_dct_hi)/2.0                                    #ACTUAL BINARY SEARCH TI FIND BEST LAMBDA
            nquery += 1
            adv, _ = self.generate_adv(x0,patch_num,lbd_dct_mid*theta_dct)
            adv_class = self.get_label(adv.unsqueeze(0).to(device))
            if adv_class != y0:
                lbd_dct_hi = lbd_dct_mid
            else:
                lbd_dct_lo = lbd_dct_mid
        return lbd_dct_hi, nquery

                                 

    def sign_grad_v1(self, x0, y0, patch_num, theta_dct, initial_lbd_dct, dct_mask, h=0.001, target=None):
                                                          #(best lambda)
        K = self.k #200
        sign_grad = np.zeros(theta_dct.shape)
        queries = 0

        for _ in range(K): # for each u
            u = np.random.randn(*theta_dct.shape)   # mu(j)
            u = u * dct_mask                        # mu(j) = mu(j)* M(mask)
            u /= LA.norm(u)                         # normalized
            new_theta_dct = theta_dct + h*u         # theta' + epsilon*mu(j)
            new_theta_dct /= LA.norm(new_theta_dct) # normalized
            sign = 1
            #####################################################################            
            adv, prub = self.generate_adv(x0,patch_num,initial_lbd_dct*new_theta_dct)
            adv_class, adv_prob = self.get_results(adv.unsqueeze(0).to(device))  
            #####################################################################
            if (target is not None and adv_class == target):            #TARGETED ATTACK CHECK
                sign = -1

            if (target is None and adv_class != y0): # success
                sign = -1                             #if adversarial, sign = -1

            queries += 1                               
            sign_grad += u*sign                        #add all the directions that nudges our best theta and makes image adversarial scaled by sign
        
        sign_grad /= K                                # (1/J) normalize the resultant nudge direction
        
        return sign_grad, queries

    def Mask_weight(self, big_image, dimen_size, alp, patch_size, use_variance_weight=True):
        """
        Builds the weight mask matrix M (Eq. 4-7 in the AdvViT paper).

        use_variance_weight=True (default, matches the paper / AdvViT+):
            Each patch's low-frequency r x r corner is scaled by
            alpha * normalized_variance(patch), concentrating perturbation
            budget on high-texture patches where it's visually less
            noticeable, per Eq. 6-7 of the paper.

        use_variance_weight=False (matches this repo's checked-out state,
        i.e. paper's ablation "method B" in Table 5):
            Every patch's low-frequency corner gets a flat value of 1,
            with no variance weighting at all. This was the actual
            behavior found in the uploaded snapshot (the variance line
            was present but commented out).

        Flip this flag if you specifically want to reproduce the
        ablation numbers instead of the full AdvViT+ numbers.
        """
        big_image = big_image.unsqueeze(0)                     #[1,3,image_size,image_size]
        num_patches_per_row = int(self.image_size // patch_size)
        num_patches_per_col = num_patches_per_row

        variances = []
        for i in range(num_patches_per_col):
            for j in range(num_patches_per_row):
                patch = big_image[:, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                variance = torch.var(patch)
                variances.append(variance.cpu())            
        variances = np.array(variances)                        #len(variances) = 196
        max_var = np.max(variances)
        # Eq. 6: q'_i = q_i / max(Q) -- normalized per-patch variance
        variances = variances / max_var

        dct_mask = np.zeros_like(big_image.cpu())
        for r in range(num_patches_per_row):
            for c in range(num_patches_per_col):
                idx = r * num_patches_per_row + c
                if use_variance_weight:
                    # Eq. 7: M = alpha * q'_i * (0/1 low-freq mask)
                    dct_mask[:, :, r*patch_size:r*patch_size+dimen_size, c*patch_size:c*patch_size+dimen_size] = variances[idx] * alp
                else:
                    dct_mask[:, :, r*patch_size:r*patch_size+dimen_size, c*patch_size:c*patch_size+dimen_size] = 1
        
        return dct_mask



    def DCT_trans(self, image, patch_num)  :
        #x_dct = np.zeros(*image.shape())
        x_dct = np.zeros_like(image)
        patch_size = int(self.image_size/patch_num)
        for r in range(patch_num):
            for c in range(patch_num):
                row = r * patch_size
                col = c * patch_size
                x_0 = np.array(image[0])  #RGB
                x_1 = np.array(image[1])
                x_2 = np.array(image[2])

                patch0 = x_0[row:row+patch_size,col:col+patch_size]
                patch1 = x_1[row:row+patch_size,col:col+patch_size]
                patch2 = x_2[row:row+patch_size,col:col+patch_size]

                patch0_dct = cv2.dct(np.array(patch0))
                patch1_dct = cv2.dct(np.array(patch1))
                patch2_dct = cv2.dct(np.array(patch2))

                x_dct[0,row:row+patch_size,col:col+patch_size] = patch0_dct
                x_dct[1,row:row+patch_size,col:col+patch_size] = patch1_dct
                x_dct[2,row:row+patch_size,col:col+patch_size] = patch2_dct
        return x_dct

    def IDCT_trans(self, image_dct, patch_num)  :
        x_idct = np.zeros_like(image_dct)
        patch_size = int(self.image_size/patch_num)
        for r in range(patch_num):
            for c in range(patch_num): 
                row = r * patch_size
                col = c * patch_size
                x_0 = np.array(image_dct[0])  #rgb三通道数据
                x_1 = np.array(image_dct[1])
                x_2 = np.array(image_dct[2])

                patch0 = x_0[row:row+patch_size,col:col+patch_size]
                patch1 = x_1[row:row+patch_size,col:col+patch_size]
                patch2 = x_2[row:row+patch_size,col:col+patch_size]

                patch0_idct = cv2.idct(np.array(patch0))
                patch1_idct = cv2.idct(np.array(patch1))
                patch2_idct = cv2.idct(np.array(patch2))

                x_idct[0,row:row+patch_size,col:col+patch_size] = patch0_idct
                x_idct[1,row:row+patch_size,col:col+patch_size] = patch1_idct
                x_idct[2,row:row+patch_size,col:col+patch_size] = patch2_idct
        return x_idct
    
                                         
    def fine_grained_binary_search(self, model, x0, y0, patch_num, dct_theta, initial_lbd, initial_lbd_dct,current_best,current_best_dct):
        nquery = 0
        if initial_lbd > current_best:
            adv,prub = self.generate_adv(x0,patch_num,initial_lbd_dct*dct_theta)    
            adv_class, adv_prob = self.get_results(adv.unsqueeze(0).to(device))              
            if adv_class == y0:
                nquery += 1                                                          
                return initial_lbd_dct, float('inf'), nquery
            lbd_dct = current_best_dct                                        #THE CURRENT BEST IS THE INITIAL LAMBDA VALUE OF A RAW DIRECTION THAT IS LOWER THAN ALL THE INITIAL LAMBDA VALUE OF THE PREVIOUS DIRECTIONS... 
                                                                              #..THEN ONLY IT PASSES THE DIRECTION FOR BINARY SEARCH
        else:
            lbd_dct = initial_lbd_dct
        lbd = initial_lbd
        lbd_dct_hi = lbd_dct
        lbd_dct_lo = 0.0
        
        while (lbd_dct_hi - lbd_dct_lo) > 1e-3: # was 1e-5
            lbd_dct_mid = (lbd_dct_lo + lbd_dct_hi)/2.0
            nquery += 1
            adv,prub = self.generate_adv(x0,patch_num,lbd_dct_mid*dct_theta)
            lbd_new = LA.norm(prub)
            adv_class, adv_prob = self.get_results(adv.unsqueeze(0).to(device))            
            if adv_class != y0 :  #and lbd_new < lbd
                lbd_dct_hi = lbd_dct_mid                                              
                lbd = lbd_new
            else:                                                            #AND FROM THERE THE BINARY SEARCH CONVERGES TO THE LOWEST LAMBDA VALUE OF CURRENT BEST DIRECTION
                lbd_dct_lo = lbd_dct_mid                                     # (OUTSIDE FUNC) AFTER THIS, THE NEW RAW NORM (lbd) IS COMPARED TO THAT OF GTHETA(LAST BEST PRUB NORM), IF LOWER THAN GTHETA(MEANS UP UNTIL ALL DIRECTIONS THAT PASSED THROUGH BINARY SEARCH, IF THE DISTANCE BETWEEN THIS DIRECTION AND ORIGINAL IMAGE IS LOWER THAN EARLIER CONVERGED BEST), THIS IS THE NEW BEST
        return lbd_dct_hi, lbd, nquery

    # ==================================================================
    # BATCHED ATTACK ENGINE -- SAME ALGORITHM AS ABOVE, BUT RUNS B IMAGES AT ONCE
    #
    # KEY IDEA: every tensor that used to be ONE image's worth of data ([3,H,W]
    # or a single scalar) now has an extra leading batch dim B ([B,3,H,W] or [B]).
    # Every operation (dct, mask, query, compare) runs on the WHOLE batch in one
    # shot instead of looping image-by-image -- that's where the speedup comes from.
    #
    # WHY CV2 COULDN'T BE REUSED: cv2.dct only ever takes ONE 2D array at a time,
    # there is no way to hand it a batch. So the DCT/IDCT had to be rebuilt using
    # torch matrix multiplication instead (D @ patch @ D.T), which DOES support
    # a batch dimension for free via torch's broadcasting. Verified this produces
    # IDENTICAL numbers to cv2.dct (checked to float32 precision) before trusting it.
    #
    # WHAT CAN'T BE DIRECTLY COPIED FROM THE SEQUENTIAL VERSION, AND WHY:
    #  1) THE SEQUENTIAL LINE SEARCH "TRIES UP TO 5 STEPS, STOPS EARLY IF ONE FAILS"
    #     -- can't do a per-image early stop inside a batched tensor op (some images
    #     would want to stop at step 2, others at step 5 -- a tensor can't have rows
    #     that did different numbers of loop passes). FIX: always run all 5 candidate
    #     steps for EVERY image, and just keep whichever one was best per image
    #     (the ones that "should" have stopped early just keep testing candidates
    #     that don't improve on what they already found -- wasted compute, not a
    #     correctness problem).
    #  2) THE SEQUENTIAL BINARY SEARCH RUNS "UNTIL THE GAP < tol" (VARIABLE LENGTH)
    #     -- same problem, different images converge after a different number of
    #     halvings. FIX: just run a FIXED number of halvings (n_iters) for
    #     everybody. After enough halvings the gap is tiny regardless of where it
    #     started, so this reaches the same answer, just sometimes with a few
    #     "wasted" extra halvings on images that would've converged sooner.
    #  3) THE SEQUENTIAL fine_grained_binary_search_local FIRST HUNTS FOR A FAILING
    #     LOWER BOUND BY SHRINKING 0.99x REPEATEDLY (variable length again) --
    #     REALIZED THIS STEP ISN'T ACTUALLY NEEDED AT ALL: lambda=0 means "no
    #     perturbation", which is JUST THE ORIGINAL IMAGE, and the original image
    #     is by definition NOT adversarial. So lo=0 is ALWAYS a valid failing bound,
    #     no searching required -- just start the bracket at [0, known_success] and
    #     go straight to bisecting. Simpler AND removes another variable-length step.
    #  4) "WHICH IMAGES ARE STILL BEING ATTACKED" -- tracked with a boolean tensor
    #     called `active`, one entry per image. An image goes inactive once it runs
    #     out of query budget OR its optimizer gets stuck (same alpha<1e-4 / beta
    #     shrink logic as the sequential version, just per-image now). IMPORTANT:
    #     going "inactive" does NOT mean that image stops being computed on --
    #     every image still goes through every tensor op every iteration (you can't
    #     shrink a tensor mid-loop), it just means torch.where(...) throws away
    #     that image's new result and keeps its old xg/gg/gg_dct frozen instead.
    #     The whole outer loop only stops once EVERY image is inactive.
    # ==================================================================

    def _get_dct_basis(self, patch_size, device):
        # BUILDS THE "D" MATRIX SUCH THAT D @ PATCH @ D.T = cv2.dct(PATCH), FOR
        # ANY PATCH OF SHAPE (patch_size, patch_size). THIS IS WHAT LETS US DO
        # DCT ON THE WHOLE BATCH AT ONCE INSTEAD OF ONE PATCH AT A TIME LIKE cv2.
        # D IS ORTHONORMAL (D @ D.T = IDENTITY) SO THE INVERSE IS JUST D.T @ F @ D
        # -- NO SEPARATE "IDCT MATRIX" NEEDED, SAME D DOES BOTH DIRECTIONS.
        # CACHED PER (patch_size, device) SINCE IT'S THE SAME MATRIX EVERY CALL --
        # NO POINT REBUILDING IT EVERY SINGLE generate_adv_batch CALL.
        if not hasattr(self, '_dct_basis_cache'):
            self._dct_basis_cache = {}
        key = (patch_size, str(device))
        if key not in self._dct_basis_cache:
            n = torch.arange(patch_size, dtype=torch.float64)
            k = torch.arange(patch_size, dtype=torch.float64).unsqueeze(1)
            D = torch.cos(np.pi / patch_size * (n + 0.5) * k)          # DCT-II BASIS FORMULA
            D[0, :] *= np.sqrt(1.0 / patch_size)                       # ORTHONORMAL SCALING, ROW 0 (DC TERM)
            D[1:, :] *= np.sqrt(2.0 / patch_size)                      # ORTHONORMAL SCALING, ALL OTHER ROWS
            self._dct_basis_cache[key] = D.to(dtype=torch.float32, device=device)
        return self._dct_basis_cache[key]

    def _patches_from_image(self, x, patch_num):
        # SLICES A WHOLE BATCH OF IMAGES INTO THEIR PATCHES IN ONE SHOT, NO PYTHON
        # LOOP -- THIS IS THE VECTORIZED EQUIVALENT OF THE "row = r*patch_size;
        # col = c*patch_size; patch = image[row:row+patch_size, col:col+patch_size]"
        # LOOP FROM THE ORIGINAL DCT_trans/IDCT_trans. [B,C,H,W] -> [B,C,nH,nW,d,d]
        B, C, H, W = x.shape
        patch_size = H // patch_num
        x = x.view(B, C, patch_num, patch_size, patch_num, patch_size)
        return x.permute(0, 1, 2, 4, 3, 5), patch_size          # REARRANGE SO (nH,nW) SIT TOGETHER, THEN (d,d) TOGETHER

    def _image_from_patches(self, patches, patch_num, patch_size):
        # UNDOES _patches_from_image -- STITCHES THE PATCHES BACK INTO ONE FULL IMAGE PER BATCH ROW
        B, C = patches.shape[0], patches.shape[1]
        H = W = patch_num * patch_size
        x = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        return x.view(B, C, H, W)

    def DCT_trans_batch(self, x0_batch, patch_num):
        # BATCHED VERSION OF DCT_trans -- SAME MATH (PER-PATCH 2D DCT), BUT ALL
        # PATCHES, ALL CHANNELS, ALL B IMAGES DONE IN ONE MATRIX MULTIPLY INSTEAD
        # OF THE ORIGINAL'S TRIPLE-NESTED PYTHON LOOP (r, c, channel) CALLING
        # cv2.dct ONE PATCH AT A TIME. x0_batch: [B,3,H,W]
        patches, patch_size = self._patches_from_image(x0_batch, patch_num)
        D = self._get_dct_basis(patch_size, x0_batch.device)
        dct_patches = torch.matmul(D, patches)                       # D @ PATCH
        dct_patches = torch.matmul(dct_patches, D.transpose(0, 1))    # (D @ PATCH) @ D.T  = FULL 2D DCT
        return self._image_from_patches(dct_patches, patch_num, patch_size)

    def IDCT_trans_batch(self, x_dct_batch, patch_num):
        # SAME IDEA AS ABOVE BUT INVERSE -- D.T @ F @ D UNDOES THE DCT SINCE D IS ORTHONORMAL
        patches, patch_size = self._patches_from_image(x_dct_batch, patch_num)
        D = self._get_dct_basis(patch_size, x_dct_batch.device)
        idct_patches = torch.matmul(D.transpose(0, 1), patches)       # D.T @ F
        idct_patches = torch.matmul(idct_patches, D)                  # (D.T @ F) @ D  = FULL 2D IDCT
        return self._image_from_patches(idct_patches, patch_num, patch_size)
    @torch.no_grad()
    def generate_adv_batch(self, x0_batch, patch_num, dct_theta_batch):
        # BATCHED VERSION OF generate_adv -- SAME THREE STEPS (DCT ORIGINAL, ADD
        # PERTURBATION IN DCT SPACE, IDCT BACK TO PIXELS), JUST FOR B IMAGES AT
        # ONCE. dct_theta_batch IS ALREADY lambda*theta' (SCALED, MASKED
        # DIRECTION) FOR EVERY IMAGE IN THE BATCH. RETURNS (adv, prub), BOTH
        # [B,3,H,W]. ASSUMES image_size DIVIDES EVENLY BY patch_num (TRUE FOR
        # EVERY DATASET_CONFIGS ENTRY, SO THE ORIGINAL'S "LEFTOVER EDGE STRIP"
        # PATCHING ISN'T NEEDED HERE).
        x_dct = self.DCT_trans_batch(x0_batch, patch_num)
        adv = self.IDCT_trans_batch(x_dct + dct_theta_batch, patch_num)
        adv = adv.clamp(0, 1)
        prub = adv - x0_batch                                          # PIXEL-SPACE PERTURBATION, PER IMAGE
        return adv, prub
    @torch.no_grad()
    def get_results_batch(self, x_batch):
        # BATCHED VERSION OF get_results -- ONE MODEL FORWARD PASS FOR ALL B
        # IMAGES INSTEAD OF B SEPARATE CALLS. THIS IS WHERE THE ACTUAL SPEEDUP
        # COMES FROM: EVERY PLACE THAT USED TO QUERY ONE IMAGE AT A TIME NOW
        # QUERIES THE WHOLE BATCH IN ONE GPU CALL.
        x = self.normalize(x_batch.to(device))
        class_prob = self._forward_logits(x)
        img_class = class_prob.max(1)[1]                       # [B] PREDICTED CLASS PER IMAGE
        top_prob = torch.softmax(class_prob, dim=1).max(1)[0]   # [B] CONFIDENCE PER IMAGE
        return img_class, top_prob
    @torch.no_grad()
    def get_label_batch(self, x_batch):
        # SAME AS ABOVE, JUST THE CLASS, NO CONFIDENCE (MATCHES get_label)
        x = self.normalize(x_batch.to(device))
        class_prob = self._forward_logits(x)
        return class_prob.max(1)[1]
    @torch.no_grad()
    def Mask_weight_batch(self, x0_batch, dimen_size, alp, patch_size, use_variance_weight=True):
        # BATCHED VERSION OF Mask_weight -- BUILDS A SEPARATE MASK M PER IMAGE,
        # SINCE EACH IMAGE HAS ITS OWN PATCH TEXTURE (VARIANCE). ONLY RUNS ONCE
        # PER attack_untargeted_batch CALL (NOT PER-ITERATION), SO THE SMALL
        # REMAINING PYTHON LOOP BELOW (OVER PATCH POSITIONS, NOT OVER BATCH OR
        # CHANNEL) COSTS NOTHING NOTICEABLE.
        B, C, H, W = x0_batch.shape
        n = H // patch_size
        patches, _ = self._patches_from_image(x0_batch, n)
        # patches: [B,C,n,n,patch_size,patch_size] -- COMBINE CHANNEL+SPATIAL INTO ONE
        # FLAT DIM PER PATCH SO torch.var CAN COMPUTE ONE SCALAR PER PATCH, MATCHING
        # THE ORIGINAL'S torch.var(patch) WHICH ALSO POOLED ALL CHANNELS TOGETHER.
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, n, n, -1)
        variances = patches.var(dim=-1, unbiased=False)                      # [B,n,n] -- ONE q_i PER PATCH PER IMAGE
        max_var = variances.reshape(B, -1).max(dim=1)[0].view(B, 1, 1)       # max(Q), PER IMAGE
        norm_var = variances / (max_var + 1e-12)                             # Eq. 6: q_i' = q_i / max(Q)

        weight = (norm_var * alp) if use_variance_weight else torch.ones_like(norm_var)   # Eq. 7 vs FLAT-1 ABLATION

        mask = torch.zeros(B, C, H, W, device=x0_batch.device)
        for r in range(n):
            for c in range(n):
                # STAMP EACH PATCH'S WEIGHT INTO ITS LOW-FREQUENCY CORNER, SAME
                # BROADCAST PATTERN AS THE SEQUENTIAL Mask_weight, JUST DONE FOR
                # ALL B IMAGES' CORNERS AT ONCE VIA THE [:,1,1,1] BROADCAST.
                mask[:, :, r*patch_size:r*patch_size+dimen_size, c*patch_size:c*patch_size+dimen_size] = \
                    weight[:, r, c].view(B, 1, 1, 1)
        return mask
    @torch.no_grad()
    def _bisect_batch(self, x0_batch, y0_batch, patch_num, theta_unit, hi, active, n_iters):
        # BATCHED, FIXED-ITERATION BINARY SEARCH -- REPLACES BOTH
        # fine_grained_binary_search AND fine_grained_binary_search_local FROM
        # THE SEQUENTIAL VERSION. lo ALWAYS STARTS AT 0 (SEE THE BIG COMMENT
        # BLOCK ABOVE -- lambda=0 IS ALWAYS A GUARANTEED-FAILING BOUND, SO THE
        # ORIGINAL'S "HUNT FOR A FAILING LOWER BOUND BY SHRINKING 0.99x" STEP
        # ISN'T NEEDED AT ALL). RUNS n_iters HALVINGS UNCONDITIONALLY FOR EVERY
        # IMAGE -- ROWS THAT WOULD'VE CONVERGED SOONER JUST KEEP HALVING AN
        # ALREADY-TINY GAP, HARMLESS, JUST SLIGHTLY WASTEFUL.
        # ROWS WHERE active IS False ARE COMPLETELY UNTOUCHED (frozen).
        # RETURNS: (converged_hi [B] = BEST LAMBDA FOUND, best_l2_at_success [B]
        # = PIXEL-SPACE DISTORTION AT THAT LAMBDA, queries_spent [B])
        B = x0_batch.shape[0]
        dev = x0_batch.device
        lo = torch.zeros(B, device=dev)                          # lambda=0 -- ALWAYS FAILS, NO SEARCH NEEDED
        hi = hi.clone()                                          # KNOWN-SUCCESSFUL STARTING DISTANCE, PER IMAGE
        best_l2 = torch.full((B,), float('inf'), device=dev)
        total_q = torch.zeros(B, dtype=torch.long, device=dev)
        for _ in range(n_iters):
            mid = (lo + hi) / 2.0                                 # MIDPOINT OF THE BRACKET, PER IMAGE
            adv, prub = self.generate_adv_batch(x0_batch, patch_num, mid.view(B,1,1,1) * theta_unit)
            adv_class, _ = self.get_results_batch(adv)            # ONE QUERY, WHOLE BATCH AT ONCE
            succ = (adv_class != y0_batch) & active                # STILL ADVERSARIAL AT THIS MIDPOINT?
            total_q += active.long()                               # ONLY COUNT A QUERY FOR ROWS STILL ACTIVE
            l2 = prub.flatten(1).norm(dim=1)
            best_l2 = torch.where(succ, torch.minimum(best_l2, l2), best_l2)
            hi = torch.where(succ, mid, hi)                        # SUCCEEDED -> TIGHTEN THE KNOWN-GOOD SIDE DOWN
            lo = torch.where(succ, lo, mid)                        # FAILED -> TIGHTEN THE KNOWN-BAD SIDE UP
        # print(f"Bisect point before margin factor: {hi} ")       
        # margin_factor = 1.003                                      # push 0.3% further past the boundary than the search strictly requires
        # hi = hi * margin_factor
        # print(f"Bisect point after margin factor: {hi} ")
        return hi, best_l2, total_q
    @torch.no_grad()
    def sign_grad_batch(self, x0_batch, y0_batch, patch_num, theta_dct, initial_lbd_dct, dct_mask, h, active):
        # BATCHED VERSION OF sign_grad_v1 -- SAME K=200 RANDOM PROBES, SAME
        # SIGN-FLIP LOGIC (Eq. 11-12), BUT EVERY PROBE'S QUERY NOW COVERS ALL B
        # IMAGES IN ONE MODEL CALL INSTEAD OF ONE IMAGE AT A TIME. THE K-LOOP
        # ITSELF IS STILL SEQUENTIAL (200 PASSES), BUT EACH PASS IS NOW B TIMES
        # CHEAPER PER-IMAGE THAN THE ORIGINAL SINCE IT'S ONE BATCHED GPU CALL
        # INSTEAD OF B SEPARATE ONES. h CAN BE A SCALAR OR A [B] TENSOR (SINCE
        # THE OUTER LOOP SHRINKS beta PER-IMAGE ON A STUCK OPTIMIZER, NOT GLOBALLY).
        B = x0_batch.shape[0]
        K = self.k   # 200
        sign_grad = torch.zeros_like(theta_dct)
        h_ = h.view(B,1,1,1) if torch.is_tensor(h) else h
        for _ in range(K):
            u = torch.randn_like(theta_dct) * dct_mask                # mu(j) = mu(j) * M(mask), SAME AS SEQUENTIAL
            u = u / u.flatten(1).norm(dim=1).clamp_min(1e-12).view(B,1,1,1)   # NORMALIZED, PER IMAGE
            new_theta = theta_dct + h_ * u                            # theta' + epsilon*mu(j)
            new_theta = new_theta / new_theta.flatten(1).norm(dim=1).clamp_min(1e-12).view(B,1,1,1)
            adv, _ = self.generate_adv_batch(x0_batch, patch_num, initial_lbd_dct.view(B,1,1,1) * new_theta)
            adv_class, _ = self.get_results_batch(adv)                # ONE QUERY, WHOLE BATCH AT ONCE
            sign = torch.where(adv_class != y0_batch, torch.tensor(-1.0, device=x0_batch.device),
                                                        torch.tensor(1.0, device=x0_batch.device))  # ADVERSARIAL -> -1, PER IMAGE
            sign_grad += u * sign.view(B,1,1,1)                        # ACCUMULATE SIGNED DIRECTION, PER IMAGE
        sign_grad /= K                                                  # (1/J) NORMALIZE, PER IMAGE
        return sign_grad
    
    @torch.no_grad()
    def attack_untargeted_batch(self, x0_batch, y0_batch, patch_num, alpha=0.2, beta=0.001,
                                 iterations=1000, query_limit=4000, use_sign_opt_plus=False,
                                 num_directions=200, bisect_iters=20, bisect_iters_local=15,  
                                 dimen_size=None, alp=4, verbose_every=10):
        # BATCHED VERSION OF attack_untargeted -- SAME TWO PHASES (RANDOM DIRECTION
        # SEARCH, THEN GRADIENT DESCENT), SAME OVERALL LOGIC, JUST EVERY VARIABLE
        # THAT WAS ONE NUMBER/ONE IMAGE BEFORE IS NOW A [B]-LENGTH TENSOR OR A
        # [B,3,H,W] TENSOR. x0_batch: [B,3,H,W], y0_batch: [B] (CLEAN PREDICTED
        # LABELS). RETURNS adv[B,3,H,W], distortion[B], success[B] bool,
        # queries[B] long, prub[B,3,H,W] -- ONE RESULT PER IMAGE IN THE BATCH.
        B = x0_batch.shape[0]
        dev = device   # use the module-level GPU device, not whatever device x0_batch happened to arrive on
        x0_batch = x0_batch.to(dev)
        y0_batch = y0_batch.to(dev)

        if dimen_size is None:
            dimen_size = DATASET_CONFIGS[self.dataset]['dimen_size']
        patch_size = int(self.image_size / patch_num)

        dct_mask = self.Mask_weight_batch(x0_batch, dimen_size, alp, patch_size)  # ONE MASK M PER IMAGE, BUILT ONCE

        # ---- PHASE 1: INITIAL DIRECTION SEARCH, BATCHED ----
        # SAME AS THE SEQUENTIAL VERSION'S "for i in range(num_directions):" LOOP --
        # TRY num_directions RANDOM MASKED DIRECTIONS, KEEP WHICHEVER GIVES THE
        # SMALLEST DISTORTION PER IMAGE. NO EARLY EXIT HERE EITHER IN THE
        # SEQUENTIAL VERSION -- ALL num_directions ALWAYS GET TRIED, SO THIS
        # BATCHES CLEANLY WITHOUT NEEDING ANY PER-IMAGE SKIP LOGIC.
        print(f"[batch] Searching for initial direction on {num_directions} random directions, batch size {B}...")
        query_count = torch.zeros(B, dtype=torch.long, device=dev)
        g_theta = torch.full((B,), float('inf'), device=dev)       # BEST DISTORTION FOUND SO FAR, PER IMAGE
        g_dct = torch.full((B,), float('inf'), device=dev)          # BEST LAMBDA (DCT SCALE) FOR THAT DIRECTION
        best_theta_dct = torch.zeros_like(x0_batch)                 # BEST THETA'/||THETA'|| SO FAR, PER IMAGE

        for _ in range(num_directions):
            query_count += 1
            theta = torch.randn_like(x0_batch) * dct_mask           # theta' = theta * M, FOR EVERY IMAGE AT ONCE
            adv, prub = self.generate_adv_batch(x0_batch, patch_num, theta)
            adv_class, _ = self.get_results_batch(adv)
            succeeded = (adv_class != y0_batch)                     # PER-IMAGE: DID THIS RANDOM DIRECTION FOOL THE MODEL?
            if not succeeded.any():
                continue                                             # NOBODY IN THE BATCH SUCCEEDED THIS ROUND, SKIP THE REFINE STEP

            theta_norm = theta.flatten(1).norm(dim=1).clamp_min(1e-12)   # ||theta'||, PER IMAGE
            theta_unit = theta / theta_norm.view(B,1,1,1)                # theta'/||theta'||, PER IMAGE

            # REFINE ONLY THE IMAGES THAT SUCCEEDED THIS ROUND (active=succeeded) --
            # OTHER IMAGES' ROWS ARE PASSED THROUGH UNTOUCHED INSIDE _bisect_batch.
            hi, best_l2, bs_q = self._bisect_batch(x0_batch, y0_batch, patch_num, theta_unit,
                                                     hi=theta_norm, active=succeeded, n_iters=bisect_iters)
            query_count += bs_q
            improved = succeeded & (best_l2 < g_theta)               # IS THIS DIRECTION BETTER THAN THE BEST FOUND SO FAR FOR THIS IMAGE?
            g_theta = torch.where(improved, best_l2, g_theta)
            g_dct = torch.where(improved, hi, g_dct)
            best_theta_dct = torch.where(improved.view(B,1,1,1), theta_unit, best_theta_dct)

        found_initial = (g_theta < float('inf'))
        print(f"[batch] Initial direction found for {found_initial.sum().item()}/{B} images "
              f"(mean queries so far: {query_count.float().mean().item():.0f})")
        # IMAGES WITH NO INITIAL DIRECTION FOUND STAY INACTIVE FROM THE START --
        # THEY'LL BE REPORTED AS FAILURES. (NO FALLBACK FLAT-MASK RETRY HERE, UNLIKE
        # THE SEQUENTIAL VERSION'S "if success_flag == False:" BLOCK -- KNOWN GAP,
        # CAN BE ADDED LATER IF IT TURNS OUT TO MATTER FOR YOUR DATA)

        # ---- PHASE 2: GRADIENT DESCENT, BATCHED, PER-IMAGE active MASKING ----
        xg = best_theta_dct.clone()          # xg IS THE BEST (THETA'/||THETA'||), PER IMAGE
        gg = g_theta.clone()                 # gg IS THE BEST RAW L2 DISTORTION, PER IMAGE
        gg_dct = g_dct.clone()               # gg_dct IS THE BEST LAMBDA FOR xg, PER IMAGE
        alpha_t = torch.full((B,), float(alpha), device=dev)   # STEP SIZE, PER IMAGE (EACH IMAGE ADAPTS INDEPENDENTLY)
        beta_t = torch.full((B,), float(beta), device=dev)     # PROBE MAGNITUDE, PER IMAGE
        active = found_initial.clone()       # WHICH IMAGES ARE STILL BEING REFINED

        for i in range(iterations):
            if not active.any():
                break                          # EVERY IMAGE DONE (BUDGET EXHAUSTED OR STUCK) -- STOP THE WHOLE BATCH

            # sign_grad_batch RETURNS THE AVERAGE OF ALL mu(j) THAT NUDGED xg TOWARD
            # ADVERSARIAL, SAME AS THE SEQUENTIAL sign_grad_v1, BUT ONE PER IMAGE AT ONCE
            sign_gradient = self.sign_grad_batch(x0_batch, y0_batch, patch_num, xg, gg_dct, dct_mask,
                                                   h=beta_t, active=active)
            # K=200 QUERIES SPENT FOR EVERY STILL-ACTIVE IMAGE THIS ITERATION, 0 FOR INACTIVE ONES
            query_count += torch.where(active, torch.full((B,), self.k, dtype=torch.long, device=dev),
                                                  torch.zeros(B, dtype=torch.long, device=dev))

            best_cand_theta = xg.clone()       # "NEXT THETA" CANDIDATE TRACKER, STARTS AS "NO IMPROVEMENT FOUND YET"
            best_cand_g2 = gg.clone()
            best_cand_lbd = gg_dct.clone()
            improved_this_iter = torch.zeros(B, dtype=torch.bool, device=dev)   # DID ANY CANDIDATE BEAT gg THIS ITERATION?

            # ---- FIXED 5-CANDIDATE INCREASING-ALPHA SWEEP ----
            # SEQUENTIAL VERSION TRIES UP TO 5 STEPS AND STOPS EARLY THE MOMENT ONE
            # FAILS TO IMPROVE. CAN'T DO THAT PER-IMAGE IN A BATCHED TENSOR OP (SEE
            # HEADER COMMENT), SO INSTEAD: ALWAYS RUN ALL 5, KEEP WHICHEVER WAS BEST
            # PER IMAGE. IMAGES THAT WOULD'VE STOPPED EARLY JUST KEEP TESTING
            # CANDIDATES THAT DON'T BEAT THEIR CURRENT BEST -- WASTED QUERIES, NOT
            # A CORRECTNESS ISSUE.
            cur_alpha = alpha_t.clone()
            for _ in range(5):
                new_theta = xg - cur_alpha.view(B,1,1,1) * sign_gradient   # UPDATED THETA' FOR THIS ATTEMPT, PER IMAGE
                new_theta = new_theta / new_theta.flatten(1).norm(dim=1).clamp_min(1e-12).view(B,1,1,1)

                if use_sign_opt_plus:
                    # SIGN-OPT+ GATE: ONE CHEAP QUERY AT THE CURRENT BEST-IN-THIS-
                    # ATTEMPT-SEQUENCE LAMBDA (best_cand_lbd, WHICH TIGHTENS AS BETTER
                    # CANDIDATES ARE FOUND WITHIN THESE 5 ATTEMPTS) BEFORE PAYING FOR
                    # THE FULL BISECTION. SAME GATE AS THE SEQUENTIAL AD+ PATH, JUST
                    # CHECKED FOR ALL B IMAGES AT ONCE.
                    check_lbd = torch.where(improved_this_iter, best_cand_lbd, gg_dct)
                    adv_check, _ = self.generate_adv_batch(x0_batch, patch_num, check_lbd.view(B,1,1,1) * new_theta)
                    check_class, _ = self.get_results_batch(adv_check)
                    gate_pass = (check_class != y0_batch) & active         # STILL ADVERSARIAL AT THE GATE DISTANCE, AND STILL ACTIVE
                    query_count += active.long()
                else:
                    gate_pass = active.clone()                            # PLAIN AD: NO GATE, EVERY ACTIVE IMAGE GOES STRAIGHT TO BISECTION

                if gate_pass.any():
                    new_lbd, _, bs_q = self._bisect_batch(x0_batch, y0_batch, patch_num, new_theta,
                                                            hi=gg_dct, active=gate_pass, n_iters=bisect_iters_local)
                    query_count += bs_q
                    adv, prub = self.generate_adv_batch(x0_batch, patch_num, new_lbd.view(B,1,1,1) * new_theta)
                    new_g2 = prub.flatten(1).norm(dim=1)                   # NEW RAW L2 DISTORTION FOR THIS CANDIDATE
                    better = gate_pass & (new_g2 < best_cand_g2)           # BEATS THE BEST CANDIDATE FOUND SO FAR THIS ITERATION?
                    best_cand_theta = torch.where(better.view(B,1,1,1), new_theta, best_cand_theta)
                    best_cand_g2 = torch.where(better, new_g2, best_cand_g2)
                    best_cand_lbd = torch.where(better, new_lbd, best_cand_lbd)
                    improved_this_iter = improved_this_iter | better

                cur_alpha = torch.where(gate_pass, torch.clamp(cur_alpha * 2, max=1e6), cur_alpha)   # GRADUALLY INCREASING STEP SIZE, CAPPED

            # ---- FIXED 5-CANDIDATE DECREASING-ALPHA FALLBACK, ONLY FOR IMAGES THAT DIDN'T IMPROVE ABOVE ----
            # MIRRORS THE SEQUENTIAL "if min_g2 >= gg:" FALLBACK BLOCK -- ONLY RUNS
            # FOR THE SUBSET need_fallback THAT THE INCREASING SWEEP DIDN'T HELP.
            need_fallback = active & (~improved_this_iter)
            cur_alpha_dec = alpha_t.clone()
            for _ in range(5):
                cur_alpha_dec = cur_alpha_dec * 0.25
                new_theta = xg - cur_alpha_dec.view(B,1,1,1) * sign_gradient
                new_theta = new_theta / new_theta.flatten(1).norm(dim=1).clamp_min(1e-12).view(B,1,1,1)

                if use_sign_opt_plus:
                    adv_check, _ = self.generate_adv_batch(x0_batch, patch_num, gg_dct.view(B,1,1,1) * new_theta)
                    check_class, _ = self.get_results_batch(adv_check)
                    gate_pass = (check_class != y0_batch) & need_fallback
                    query_count += need_fallback.long()
                else:
                    gate_pass = need_fallback.clone()

                if gate_pass.any():
                    new_lbd, _, bs_q = self._bisect_batch(x0_batch, y0_batch, patch_num, new_theta,
                                                            hi=gg_dct, active=gate_pass, n_iters=bisect_iters_local)
                    query_count += bs_q
                    adv, prub = self.generate_adv_batch(x0_batch, patch_num, new_lbd.view(B,1,1,1) * new_theta)
                    new_g2 = prub.flatten(1).norm(dim=1)
                    better = gate_pass & (new_g2 < gg)
                    best_cand_theta = torch.where(better.view(B,1,1,1), new_theta, best_cand_theta)
                    best_cand_g2 = torch.where(better, new_g2, best_cand_g2)
                    best_cand_lbd = torch.where(better, new_lbd, best_cand_lbd)
                    improved_this_iter = improved_this_iter | better
                    need_fallback = need_fallback & (~better)             # STOP RETRYING AN IMAGE ONCE IT IMPROVED

            # ---- PER-IMAGE STUCK DETECTION (MIRRORS THE SEQUENTIAL alpha<1e-4 RESET) ----
            stuck = active & (~improved_this_iter)                        # BOTH SWEEPS FAILED TO IMPROVE THIS IMAGE THIS ITERATION
            alpha_t = torch.where(stuck, torch.full((B,), 1.0, device=dev), torch.clamp(alpha_t * 2, max=1e6))
            beta_t = torch.where(stuck, beta_t * 0.1, beta_t)             # SHRINK THE PROBE MAGNITUDE FOR STUCK IMAGES
            active = active & (beta_t >= 1e-8)                            # GIVE UP ON AN IMAGE ONCE beta COLLAPSES

            # IF ALL ATTEMPTS FAILED FOR AN IMAGE, best_cand_* WILL STILL EQUAL ITS
            # OLD xg/gg/gg_dct (I.E. "NOT MOVING"), SAME AS THE SEQUENTIAL VERSION
            xg = best_cand_theta
            gg = best_cand_g2
            gg_dct = best_cand_lbd

            active = active & (query_count <= query_limit)                # DROP IMAGES THAT RAN OUT OF QUERY BUDGET

            if (i + 1) % verbose_every == 0:
                finite_gg = gg[torch.isfinite(gg)]
                mean_gg = finite_gg.mean().item() if finite_gg.numel() > 0 else float('nan')
                print(f"[batch] iter {i+1:4d}  active={int(active.sum().item())}/{B}"
                      f"mean_distortion={mean_gg:.4f}  mean_queries={query_count.float().mean().item():.0f}")

        # in attack_untargeted_batch, right before the final generate_adv_batch call:
        # This is applying margin factor to the final lambda, instead of applying it to every call on the bisect batch
        margin_factor = 1.02                # 1.01 to 1.03
        gg_dct_final = gg_dct * margin_factor   # apply the margin ONCE, not compounded across iterations
        adv_final, prub_final = self.generate_adv_batch(x0_batch, patch_num, gg_dct_final.view(B,1,1,1) * xg)
        
        
        # adv_final, prub_final = self.generate_adv_batch(x0_batch, patch_num, gg_dct.view(B,1,1,1) * xg)
        
        
        
        # ---- FINAL VERIFICATION: does the actual returned adv image still fool the model? ----
        # This is the true, in-memory ASR check -- same guarantee as the .npy lossless test,
        # done live here so there's no need for a separate diagnostic or disk round-trip to
        # trust the number. Also catches the case where Phase 2's line search accepted a
        # candidate based on smaller perturbation SIZE alone without re-confirming it still
        # fools the model (the hi bracket in _bisect_batch is never itself re-verified before
        # being returned/accepted).
        final_pred = self.get_label_batch(adv_final)
        success = (final_pred != y0_batch) & found_initial   # must BOTH have found a direction in Phase 1 AND still fool the model right now
        query_count += 1 
        return adv_final, gg, success, query_count, prub_final
