import torch
import torch.nn.functional as F
import utils
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


trf = T.Compose([T.ToPILImage(),
				 T.ToTensor(),
				 T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
unloader = T.ToPILImage()

device = torch.device('cuda',0)

class SimP:
    
    def __init__(self, model, dataset, image_size, k=200):
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
        
    def normalize(self, x):
        return utils.apply_normalization(x, self.dataset)

    '''def get_probs(self, x, y):
        output = self.model(self.normalize(x.to(device))).cpu()
        probs = torch.index_select(F.softmax(output, dim=-1).data, 1, y)
        return torch.diag(probs)
    
    def get_preds(self, x):
        output = self.model(self.normalize(x.to(device))).cpu()
        _, preds = output.data.max(1)
        return preds'''
    
    def get_results(self, x):
        x=self.normalize(x.to(device))
        ########## Vision Transformers
        class_prob, atten = self.model(x)     
        ##########  CNNs
        #class_prob = self.model(x)     

        img_class = class_prob.max(1)[1]
        prob = torch.max(torch.softmax(class_prob[0],dim=0))
        return img_class, prob

    def get_label(self, x):
        x=self.normalize(x.to(device))
        ########## Vision Transformers
        class_prob, atten = self.model(x)     
        ##########  CNNs        
        #class_prob = self.model(x)
        img_class = class_prob.max(1)[1]
        prob = torch.max(torch.softmax(class_prob[0],dim=0))
        return img_class

    # 20-line implementation of SimBA for single image input
    def simba_single(self, x, y, num_iters=10000, epsilon=0.2, targeted=False):
        n_dims = x.view(1, -1).size(1)
        perm = torch.randperm(n_dims)
        x = x.unsqueeze(0)
        last_prob = self.get_probs(x, y)
        for i in range(num_iters):
            diff = torch.zeros(n_dims)
            diff[perm[i]] = epsilon
            left_prob = self.get_probs((x - diff.view(x.size())).clamp(0, 1), y)
            if targeted != (left_prob < last_prob):
                x = (x - diff.view(x.size())).clamp(0, 1)
                last_prob = left_prob
            else:
                right_prob = self.get_probs((x + diff.view(x.size())).clamp(0, 1), y)
                if targeted != (right_prob < last_prob):
                    x = (x + diff.view(x.size())).clamp(0, 1)
                    last_prob = right_prob
            if i % 10 == 0:
                print(last_prob)
        return x.squeeze()

    # runs simba on a batch of images <images_batch> with true labels (for untargeted attack) or target labels
    # (for targeted attack) <labels_batch>

    def generate_adv(self, x0, patch_num,dct_theta):
        x = np.array(x0.clone())
        x_dct = self.DCT_trans(x, patch_num)
        adv = self.IDCT_trans(x_dct + dct_theta,patch_num)
        patch_size = int(224/patch_num)
        if 224 % patch_num !=0:
            adv[:,0:224,patch_num*patch_size:224] = x0[:,0:224,patch_num*patch_size:224]
            adv[:,patch_num*patch_size:224,0:224] = x0[:,patch_num*patch_size:224,0:224]
        #adv = self.IDCT_trans(x_dct,patch_num)
        adv = torch.tensor(adv, dtype=torch.float)
        adv = adv.clamp(0, 1)
        #adv = adv.unsqueeze(0)
        prub = adv - x0
        return adv,prub


    def simp_batch(self, images_batch, labels_batch, patch_size, max_iters, num, epsilon=0.0, linf_bound=0.0,
                    order='rand', targeted=False, pixel_attack=False, log_every=1):
        batch_size = images_batch.size(0)
        image_size = images_batch.size(2)

        patch_num = int(image_size / patch_size)
        #patch_num = patch_num * patch_num
        ori_path = 'D:\\zc\\simple-patch-master-plus\\save\\ori'
        adv_path = 'D:\\zc\\simple-patch-master-plus\\save\\adv'
        prub_path = 'D:\\zc\\simple-patch-master-plus\\save\\prub'

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
                          distortion=None, svm=False, momentum=0.0, stopping=0.0001):
        model = self.model
        query_count = 0
        ls_total = 0
        success_thold = 5.0
        best_zero_num = 0
        num_directions = 100
        dimen_size = 4
        alp = 4
        success_flag = False
        best_theta_dct, g_theta, g_dct  = None, float('inf'), float('inf')
        print("Searching for the initial direction on %d random directions: " % (num_directions))
        #dct_mask = np.ones_like(x0.cpu())        
        #####################################################################
        '''dct_mask = np.zeros_like(x0.cpu())
        for j in range(patch_num):
            for k in range(patch_num):
                dct_mask[:,j*16:j*16+3,k*16:k*16+3] = 1'''
        patch_size = int(224/patch_num)
        dct_mask = self.Mask_weight(x0, dimen_size,alp,patch_size)
        dct_mask = dct_mask.squeeze(0)
        #####################################################################
        for i in range(num_directions):
            query_count += 1
            x0 = x0.cpu().clone()
            dct_theta = np.random.randn(*np.array(x0).shape) # gaussian distortion
            dct_theta = dct_theta * dct_mask
            #print(dct_theta)
            #print(np.count_nonzero(dct_theta))

            adv, prub= self.generate_adv(x0,patch_num,dct_theta * dct_mask)
            # register adv directions
            adv_class, adv_prob = self.get_results(adv.unsqueeze(0).to(device))
            if adv_class != y0:
                success_flag = True
                initial_lbd_dct = LA.norm(dct_theta)
                initial_l2 = LA.norm(prub)

                prub /= initial_l2 # l2 normalize
                dct_theta /= initial_lbd_dct
                lbd_dct, lbd_l2, count = self.fine_grained_binary_search(model, x0, y0, patch_num, dct_theta, initial_l2,  initial_lbd_dct, g_theta, g_dct)

                query_count += count
                if lbd_l2 < g_theta:
                    g_theta, best_theta_dct, g_dct  = lbd_l2, dct_theta, lbd_dct
                    print("--------> Found l2distortion %.4f,dctdistortion%.4f" % (g_theta, g_dct))
                    if round(g_theta, 3) == 0.000:
                        print("not need to be attacked")
                        return x0.to(device), 0, True, query_count, torch.zeros(3,224,224)
            
        ###############################################################
        if success_flag == False:
            dct_mask = np.zeros_like(x0.cpu())
            for j in range(patch_num):
                for k in range(patch_num):
                    dct_mask[:,j*16:j*16+dimen_size,k*16:k*16+dimen_size] = 1
            for i in range(query_limit - query_count):
                query_count += 1
                dct_theta = np.random.randn(*np.array(x0).shape) # gaussian distortion
                dct_theta = dct_theta * dct_mask

                adv, prub= self.generate_adv(x0,patch_num,dct_theta * dct_mask)
                adv_class, adv_prob = self.get_results(adv.unsqueeze(0).to(device))
                if adv_class != y0:
                    success_flag = True
                    initial_lbd_dct = LA.norm(dct_theta)
                    initial_l2 = LA.norm(prub)

                    prub /= initial_l2 # l2 normalize
                    dct_theta /= initial_lbd_dct
                    lbd_dct, lbd_l2, count = self.fine_grained_binary_search(model, x0, y0, patch_num, dct_theta, initial_l2,  initial_lbd_dct, g_theta, g_dct)

                    query_count += count
                    if lbd_l2 < g_theta:
                        g_theta, best_theta_dct, g_dct  = lbd_l2, dct_theta, lbd_dct
                        print("--------> Found l2distortion %.4f,dctdistortion%.4f" % (g_theta, g_dct))
                    break                   
            ##########################################################

        ## fail if cannot find a adv direction within 200 Gaussian
        if g_theta == float('inf'):
            print("Couldn't find valid initial, failed")
            return x0.to(device), 0, False, query_count, torch.zeros(3,224,224)
        if round(g_theta, 3) == 0.000:
            print("not need to be attacked")
            return x0.to(device), 0, True, query_count, torch.zeros(3,224,224)
        print("==========> Found best distortion %.4f"
              "using %d queries" % (g_theta,query_count))

        #### Begin Gradient Descent.
        xg, gg, gg_dct = best_theta_dct, g_theta, g_dct#
        vg = np.zeros_like(xg)
        distortions = [gg]
        for i in range(iterations):
            sign_gradient, grad_queries = self.sign_grad_v1(x0, y0, patch_num, xg, gg_dct, dct_mask, h=beta)
            ls_count = 0
            min_theta_dct = xg ## next theta
            min_g2 = gg ## current g_theta
            min_lbd_dct = gg_dct ## velocity (for momentum only)
            for _ in range(5):
                new_theta_dct = xg - alpha * sign_gradient
                new_theta_dct /= LA.norm(new_theta_dct)

                new_lbd_dct, count = self.fine_grained_binary_search_local(
                    model, x0, y0, patch_num, new_theta_dct, initial_lbd = min_g2, initial_lbd_dct = gg_dct,tol=beta/500)
                ls_count += count
                adv, prub= self.generate_adv(x0,patch_num,new_lbd_dct*new_theta_dct)
                new_g2 = LA.norm(prub)                

                alpha = alpha * 2 # gradually increasing step size
                if new_g2 < min_g2:
                    min_theta_dct = new_theta_dct
                    min_lbd_dct = new_lbd_dct
                    min_g2 = new_g2
                else:
                    break

            if min_g2 >= gg: ## if the above code failed for the init alpha, we then try to decrease alpha
                for _ in range(5):
                    alpha = alpha * 0.25
                    new_theta_dct = xg - alpha * sign_gradient
                    new_theta_dct /= LA.norm(new_theta_dct)

                    new_lbd_dct, count = self.fine_grained_binary_search_local(
                        model, x0, y0, patch_num, new_theta_dct, initial_lbd = min_g2, initial_lbd_dct = gg_dct,tol=beta/500)
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

        return adv.unsqueeze(0).to(device), gg, True, query_count, adv-x0

    def fine_grained_binary_search_local(self, model, x0, y0,patch_num, theta_dct, initial_lbd = 1.0, initial_lbd_dct = 1.0, tol=5e-3):
        nquery = 0
        lbd = initial_lbd
        lbd_dct = initial_lbd_dct
        adv, prub = self.generate_adv(x0,patch_num,lbd_dct*theta_dct)
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
            while self.get_label(adv.unsqueeze(0).to(device)) != y0 :
                lbd_dct_lo = lbd_dct_lo*0.99
                nquery += 1
                adv, prub = self.generate_adv(x0,patch_num,lbd_dct_lo*theta_dct)
                if lbd_dct_lo < 1e-4:
                    break

        while (lbd_dct_hi - lbd_dct_lo) > tol:
            lbd_dct_mid = (lbd_dct_lo + lbd_dct_hi)/2.0
            nquery += 1
            adv, prub = self.generate_adv(x0,patch_num,lbd_dct_mid*theta_dct)
            adv_class = self.get_label(adv.unsqueeze(0).to(device))
            if adv_class != y0:
                lbd_dct_hi = lbd_dct_mid
            else:
                lbd_dct_lo = lbd_dct_mid
        return lbd_dct_hi, nquery

    def sign_grad_v1(self, x0, y0, patch_num, theta_dct, initial_lbd_dct,dct_mask, h=0.001, D=4, target=None):
    
        K = self.k #200
        sign_grad = np.zeros(theta_dct.shape)
        queries = 0

        for iii in range(K): # for each u
            u = np.random.randn(*theta_dct.shape)
            u = u * dct_mask
            u /= LA.norm(u)
            new_theta_dct = theta_dct + h*u
            new_theta_dct /= LA.norm(new_theta_dct)
            sign = 1
            #####################################################################            
            adv, prub = self.generate_adv(x0,patch_num,initial_lbd_dct*new_theta_dct)
            adv_class, adv_prob = self.get_results(adv.unsqueeze(0).to(device))  
            #####################################################################
            if (target is not None and adv_class == target):
                sign = -1

            if (target is None and adv_class != y0): # success
                sign = -1

            queries += 1
            sign_grad += u*sign
        
        sign_grad /= K
        
        return sign_grad, queries

    def Mask_weight(self, big_image, dimen_size,alp,patch_size)  :
        #x_dct = np.zeros(*image.shape())
        big_image = big_image.unsqueeze(0)
        # 定义小图的大小和列数
        num_patches_per_row = int(224 // patch_size)

        # 计算行数和总小图数量
        num_patches_per_col = num_patches_per_row
        num_patches = num_patches_per_row * num_patches_per_col

        # 提取小图，并计算每张小图的方差
        #patches = np.zeros((num_patches, 1, 3, patch_size, patch_size))
        variances = []
        for i in range(num_patches_per_col):
            for j in range(num_patches_per_row):
                patch = big_image[:, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                #patches[i * num_patches_per_row + j] = patch
                variance = torch.var(patch)  # 计算方差
                variances.append(variance.cpu())
        max_var = np.max(variances)
        min_var = np.min(variances)

        # variances -= min_var
        # variances /= (max_var - min_var) 
        variances /= max_var

        dct_mask = np.zeros_like(big_image.cpu())
        for r in range(num_patches_per_row):
            for c in range(num_patches_per_col):
                # if variances[r * num_patches_per_row + c] >= np.percentile(variances, 50):
                #     dct_mask[:,:,r*16:r*16+dimen_size,c*16:c*16+dimen_size] = 1
                # else:
                #dct_mask[:,:,r*patch_size:r*patch_size+dimen_size,c*patch_size:c*patch_size+dimen_size] = variances[r * num_patches_per_row + c] * alp 
                dct_mask[:,:,r*patch_size:r*patch_size+dimen_size,c*patch_size:c*patch_size+dimen_size] = 1 

        # 输出结果
        # print(patches.shape)  # (196, 1, 3, 16, 16)
        # print(variances)  # 输出每张小图的方差值
        return dct_mask



    def DCT_trans(self, image, patch_num)  :
        #x_dct = np.zeros(*image.shape())
        x_dct = np.zeros_like(image)
        patch_size = int(224/patch_num)
        for r in range(patch_num):
            for c in range(patch_num):
                row = r * patch_size
                col = c * patch_size
                x_0 = np.array(image[0])  #rgb三通道数据
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
        patch_size = int(224/patch_num)
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
            lbd_dct = current_best_dct
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
            else:
                lbd_dct_lo = lbd_dct_mid
        return lbd_dct_hi, lbd, nquery
    

