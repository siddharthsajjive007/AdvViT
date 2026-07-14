import os
import torch
import zipfile
import random
from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

from simp import SimP, DATASET, DATASET_CONFIGS
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

device = torch.device('cuda', 0)
print('CUDA available:', torch.cuda.is_available())
print('Device:', device)

'''
MENTION THE DATASET IN simp.py AND THE MODEL USED IN THIS FILE BELOW
'''
MODEL_ARCH = 'resnet18_cifar10'   # 'resnet50' | 'DeiT_B' | 'DeiT_S' | 'DeiT_T' | 'resnet18_cifar10' | 'resnet50_gtsrb32'| 'deit_cifar10' | 'ViT'

# ZIP_PATH = "/home/HDD/ATAF/Datasets/ImageNetDataset/ATAF-Framework-Ready/ImageNet-3599-Targeted.zip"   #IMAGENET
ZIP_PATH = "/home/HDD/ATAF/Datasets/CIFAR10-Dataset/CIFAR-10-60k-targeted.zip"    #CIFAR10
# ZIP_PATH = "/home/HDD/ATAF/Datasets/GTSRB//GTSRB_test_ataf.zip"


# Pull resolution/patch geometry from simp.py's DATASET_CONFIGS instead of
# hardcoding -- this is the single source of truth shared with SimP itself,
# so IMAGE_SIZE/PATCH_SIZE here always match whatever SimP actually uses
# internally, regardless of which DATASET is configured in simp.py.
config = DATASET_CONFIGS[DATASET]
IMAGE_SIZE = config['size']
PATCH_SIZE = config['patch_size']


def load_model(model_arch='ViT', device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_arch == 'resnet50':
        net = torch.load(
            "/home/siddarth/ATTACK/resnet50_testing_model.pth",
            map_location=device,
            weights_only=False
        )
    elif model_arch == 'resnet18_cifar10':
        net = torch.load(
            '/home/HDD/ATAF/Model-files/Patchfool/pytorch_models/cifar10/resnet18_cifar10.pth',
            map_location=device
        )
    elif model_arch == 'resnet50_cifar10':
        net = torch.load(
            '/home/HDD/ATAF/Model-files/Patchfool/pytorch_models/cifar10/resnet50_cifar10.pth',
            map_location=device
        )
    elif model_arch == 'deit_cifar10':
        net = torch.load(
            '/home/HDD/ATAF/Model-files/Patchfool/pytorch_models/cifar10/deit_cifar10_epoch_0070_ataf_ready.pth',
            map_location=device)
    elif model_arch == 'resnet50_gtsrb32':
        net = torch.load(
            '/home/HDD/ATAF/Model-files/GTSRB/gtsrb_resnet50_32.pth',
            map_location=device)
    elif model_arch == 'DeiT_T':
        net = torch.load(
            '/home/HDD/ATAF/Model-files/Patchfool/pytorch_models/deit_variations/deit_tiny_patch16_224_ataf_ready.pth',
            map_location=device
    )
    elif model_arch == 'DeiT_S':
        net = torch.load(
            '/home/HDD/ATAF/Model-files/Patchfool/pytorch_models/deit_variations/deit_small_patch16_224_ataf_ready.pth',
            map_location=device
        )
    elif model_arch == 'DeiT_B':
        net = torch.load(
            '/home/HDD/ATAF/Model-files/Patchfool/pytorch_models/deit_variations/deit_base_patch16_224_ataf_ready.pth',
            map_location=device
        )
    elif model_arch == 'ViT':
        net = torch.load(
            '/home/HDD/ATAF/Model-files/ViT/vit_base_patch16_224_pytorch_complete.zip',
            map_location=device
        )
    else:
        raise ValueError(f"Unsupported model architecture: {model_arch}")

    net = net.to(device)
    net.eval()
    return net


def load_random_image_from_zip(zip_path, image_size, device):
    """
    Loads a random image from the zip's images/ folder,
    matching the format of the original single-image loader.
    Returns x0: [3, IMAGE_SIZE, IMAGE_SIZE] tensor, range [0,1]
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        # list all image files inside the zip's images/ folder
        image_files = [
            f for f in z.namelist()
            if f.startswith('images/') and f.lower().endswith(('.jpeg', '.jpg', '.png'))
        ]
        
        # pick one at random
        chosen_file = random.choice(image_files)
        
        with z.open(chosen_file) as f:
            img = Image.open(f).convert('RGB').resize((image_size, image_size))
    
    x0 = T.ToTensor()(img).to(device)   # [3, IMAGE_SIZE, IMAGE_SIZE], range [0,1]
    return x0, chosen_file



# ── load model once ──
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading {MODEL_ARCH} on {device}...")
model = load_model(MODEL_ARCH, device)


x0, chosen_filename = load_random_image_from_zip(ZIP_PATH, IMAGE_SIZE, device)
print(f"Loaded random image: {chosen_filename}")


#Get prediction
attacker = SimP(model, DATASET, image_size=IMAGE_SIZE)
with torch.no_grad():
    y0 = attacker.get_label(x0.unsqueeze(0))
print(f'Clean prediction (class index): {y0.item()}')


#================AD+ ATTACK==============================
import time

print(MODEL_ARCH)
patch_size = PATCH_SIZE
patch_num = IMAGE_SIZE // PATCH_SIZE
QUERY_LIMIT = 3000

t0 = time.time()
adv, distortion, is_success, nqueries, prub = attacker.attack_untargeted(
    x0, y0, ori_probal=None, patch_num=patch_num, query_limit=QUERY_LIMIT,
    use_sign_opt_plus=True
)
elapsed = time.time() - t0

print(f'Attack finished in {elapsed:.1f}s ({elapsed/60:.2f} min).')


# Results — attack outcome, L2 distortion, SSIM, PSNR

adv_np = adv.squeeze(0).detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
ori_np = x0.detach().cpu().permute(1, 2, 0).numpy()

ssim_val = ssim_fn(ori_np, adv_np, channel_axis=2, data_range=1.0)
psnr_val = psnr_fn(ori_np, adv_np, data_range=1.0)

with torch.no_grad():
    adv_pred = attacker.get_label(adv.to(device) if adv.dim() == 4 else adv.unsqueeze(0).to(device))

print('=' * 50)
print(f'{"Success:":<20}{bool(is_success)}')
print(f'{"Queries used:":<20}{nqueries} / {QUERY_LIMIT}')
print(f'{"L2 distortion:":<20}{distortion:.4f}')
print(f'{"SSIM:":<20}{ssim_val:.4f}   (1.0 = identical, closer to 1 = more visually similar)')
print(f'{"PSNR:":<20}{psnr_val:.2f} dB   (higher = more visually similar)')
print(f'{"Original class:":<20}{y0.item()}')
print(f'{"Adversarial class:":<20}{adv_pred.item()}')
print('=' * 50)



#============================== VISUALIZATION =============================================



# amplify the perturbation for visibility -- raw diff is usually near-invisible
diff = adv_np - ori_np
diff_vis = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

fig, axes = plt.subplots(1, 3, figsize=(13, 5))

axes[0].imshow(ori_np)
axes[0].set_title(f'Original\n(class {y0.item()})')
axes[0].axis('off')

axes[1].imshow(adv_np)
axes[1].set_title(f'Adversarial\n(class {adv_pred.item()})')
axes[1].axis('off')

axes[2].imshow(diff_vis)
axes[2].set_title('Perturbation\n(contrast-stretched for visibility)')
axes[2].axis('off')

fig.suptitle(f'L2={distortion:.3f}  SSIM={ssim_val:.4f}  PSNR={psnr_val:.2f}dB  queries={nqueries}', fontsize=12)
plt.tight_layout()

# Save the full 3-panel figure -- Linux-side path, matches what
SAVE_NAME = f'animal_{QUERY_LIMIT}_{MODEL_ARCH}_{DATASET}'
print("Image saved to AdvViT/save") 
save_dir = '/home/siddarth/AdvViT/OUTPUT'
os.makedirs(save_dir, exist_ok=True)
fig.savefig(os.path.join(save_dir, f'output_ad+_{SAVE_NAME}.png'), dpi=150, bbox_inches='tight')
plt.show()

