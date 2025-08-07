import math
import os
import numpy as np
from PIL import Image
from scipy.signal import convolve2d


# target:目标图像  ref:参考图像 
def PSNR(target, ref):
    target = np.array(target, dtype=np.float32)
    ref = np.array(ref, dtype=np.float32)

    if target.shape != ref.shape:
        raise ValueError('输入图像的大小应该一致！')

    diff = ref - target
    diff = diff.flatten('C')

    rmse = math.sqrt( np.mean(diff ** 2.) )
    if rmse == 0 :
        return 0
    psnr = 20 * math.log10(np.max(target) / rmse)

    return psnr


def SSIM(target, ref, K1=0.01, K2=0.03, gaussian_kernel_sigma = 1.5, gaussian_kernel_width = 11, L=255):
    # 高斯核，方差为1.5，滑窗为11*11
    gaussian_kernel = np.zeros((gaussian_kernel_width,gaussian_kernel_width))
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i,j] = (1 / (2 * math.pi * (gaussian_kernel_sigma ** 2))) * math.exp(-(((i-5) ** 2)+((j - 5) ** 2)) / (2 * (gaussian_kernel_sigma ** 2)))

    target = np.array(target, dtype=np.float32)
    ref = np.array(ref, dtype=np.float32)
    if target.shape != ref.shape:
        raise ValueError('输入图像的大小应该一致！')

    target_window = convolve2d(target, np.rot90(gaussian_kernel, 2), mode='valid')
    ref_window = convolve2d(ref, np.rot90(gaussian_kernel, 2), mode='valid')

    mu1_sq = target_window * target_window
    mu2_sq = ref_window * ref_window
    mu1_mu2 = target_window * ref_window

    sigma1_sq = convolve2d(target * target, np.rot90(gaussian_kernel, 2), mode='valid') - mu1_sq
    sigma2_sq = convolve2d(ref * ref, np.rot90(gaussian_kernel, 2), mode='valid') - mu2_sq
    sigma12 = convolve2d(target * ref, np.rot90(gaussian_kernel, 2), mode='valid') - mu1_mu2

    C1 = (K1*L)**2
    C2 = (K2*L)**2
    ssim_array = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim = np.mean(np.mean(ssim_array))

    return ssim



if __name__ == '__main__':
    #target = Image.open('save/adv_images/adv0queriestensor(1172.).jpg').convert('L')
    #ref = Image.open('save/adv_images/adv0queriestensor(1172.).jpg').convert('L')

    '''psnr = PSNR(target, ref)
    print('PSNR为:{}'.format(psnr))'''


 
    adv_dir = 'D:\zc\simple-patch-master-plus1\save\VIT-B4000-16\\3-1\\adv'
    original_dir = 'D:\\zc\\simple-patch-master-plus1\\save\\ori'

    imgList = os.listdir(original_dir)
    #adv_imgList = os.listdir(adv_dir)

    def get_file_name_with_number(num):
        str1 = 'adv'
        str2='ori'        
        str3=num
        str_orig = ('{}{}.').format(str2,str3)
        str_adv = ('{}{}').format(str1,str3)        
        file_orig = [n for n in os.listdir(original_dir) if str_orig in n][0]
        file_adv = ''
        flag = False
        for n in os.listdir(adv_dir):
            if str_adv in n:
                file_adv = n
                flag = True
                break
                
        return flag, file_orig, file_adv

    ssim = 0
    psnr = 0
    i = 0
    ssim_list = []
    psnr_list = []
    for count in range(0, len(imgList)):
        flag, orig, adv = get_file_name_with_number(count)
        if flag == False:
            continue
        orig_path = os.path.join(original_dir,orig)
        adv_path = os.path.join(adv_dir,adv)
        target = Image.open(orig_path).convert('L')
        ref = Image.open(adv_path).convert('L')

        s= SSIM(target, ref)
        ssim += s
        ssim_list.append(s)

        temp = PSNR(target, ref)
        if(temp==0):
            i+=1
            continue
        psnr += temp
        psnr_list.append(temp)
    ssim /= len(imgList)
    psnr /= len(imgList)  - i 
    print('SSIM均值为:{}'.format(ssim))
    print('SSIM中值为:{}'.format(np.median(ssim_list)))
    print('psnr均值为:{}'.format(psnr))
    print('psnr中值为:{}'.format(np.median(psnr_list)))