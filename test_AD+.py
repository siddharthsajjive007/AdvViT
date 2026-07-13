import os
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

from simp import SimP
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

device = torch.device('cuda', 0)
print('CUDA available:', torch.cuda.is_available())
print('Device:', device)

# ---- uncomment ONE of these ----
# MODEL_CHOICE = 'DeiT_T'
# MODEL_CHOICE = 'DeiT_S'
MODEL_CHOICE = 'DeiT_B'
# MODEL_CHOICE = 'ResNet18'
# MODEL_CHOICE = 'ResNet50'
# MODEL_CHOICE = 'ResNet101'
# MODEL_CHOICE = 'ResNet152'
# MODEL_CHOICE = 'VGG16'
# MODEL_CHOICE = 'Swin_T'
# MODEL_CHOICE = 'Swin_S'
# MODEL_CHOICE = 'Swin_B'
# MODEL_CHOICE = 'ViT_B'   # patch_size=32, not 16 -- see note above

if MODEL_CHOICE == 'DeiT_T':
    from models.DeiT import deit_tiny_patch16_224
    model = deit_tiny_patch16_224(pretrained=True).to(device).eval()
elif MODEL_CHOICE == 'DeiT_S':
    from models.DeiT import deit_small_patch16_224
    model = deit_small_patch16_224(pretrained=True).to(device).eval()
elif MODEL_CHOICE == 'DeiT_B':
    from models.DeiT import deit_base_patch16_224
    model = deit_base_patch16_224(pretrained=True).to(device).eval()
elif MODEL_CHOICE == 'ResNet18':
    import torchvision
    model = torchvision.models.resnet18(pretrained=True).to(device).eval()
elif MODEL_CHOICE == 'ResNet50':
    from models.resnet import ResNet50
    model = ResNet50(pretrained=True).to(device).eval()
elif MODEL_CHOICE == 'ResNet101':
    from models.resnet import ResNet101
    model = ResNet101(pretrained=True).to(device).eval()
elif MODEL_CHOICE == 'ResNet152':
    from models.resnet import ResNet152
    model = ResNet152(pretrained=True).to(device).eval()
elif MODEL_CHOICE == 'VGG16':
    import torchvision
    model = torchvision.models.vgg16(pretrained=True).to(device).eval()
elif MODEL_CHOICE == 'Swin_T':
    import timm
    model = timm.models.create_model('swin_tiny_patch4_window7_224', pretrained=True).to(device).eval()
elif MODEL_CHOICE == 'Swin_S':
    import timm
    model = timm.models.create_model('swin_small_patch4_window7_224', pretrained=True).to(device).eval()
elif MODEL_CHOICE == 'Swin_B':
    import timm
    model = timm.models.create_model('swin_base_patch4_window7_224', pretrained=True).to(device).eval()
elif MODEL_CHOICE == 'ViT_B':
    import timm
    model = timm.create_model('vit_base_patch32_224.augreg_in1k', pretrained=True).to(device).eval()
else:
    raise ValueError(f'Unknown MODEL_CHOICE: {MODEL_CHOICE}')

print(f'Loaded {MODEL_CHOICE}')


IMAGE_PATH = '/home/siddharthsajjive/TEA/ATTACK/ILSVRC2012_val_pairs/2b.JPEG' 
#IMAGE_PATH = '/home/siddharthsajjive/AdvViT/save/ori/ori0.jpg' # <-- change this

img = Image.open(IMAGE_PATH).convert('RGB').resize((224, 224))
x0 = T.ToTensor()(img).to(device)  # [3,224,224], range [0,1]



#Get prediction
attacker = SimP(model, 'imagenet', image_size=224)
with torch.no_grad():
    y0 = attacker.get_label(x0.unsqueeze(0))
print(f'Clean prediction (class index): {y0.item()}')



#AD+ ATTACK========================================
import time

print(MODEL_CHOICE)
patch_size = 16
patch_num = 224 // patch_size
QUERY_LIMIT = 3000

t0 = time.time()
adv, distortion, is_success, nqueries, prub = attacker.attack_untargeted(
    x0, y0, ori_probal=None, patch_num=patch_num, query_limit=QUERY_LIMIT,
    use_sign_opt_plus=True
)
elapsed = time.time() - t0

print(f'Attack finished in {elapsed:.1f}s ({elapsed/60:.2f} min).')

# print('\nRunning AD+ (Sign-OPT+ gated)...')
# t0 = time.time()
# adv_adp, distortion_adp, success_adp, queries_adp, prub_adp = attacker.attack_untargeted(
#     x0, y0, ori_probal=None, patch_num=patch_num,
#     query_limit=QUERY_LIMIT, use_sign_opt_plus=True
# )
# time_adp = time.time() - t0
# print(f'AD+ finished in {time_adp:.1f}s ({time_adp/60:.2f} min)')

# print(f'\nBoth runs finished. AD+ was {time_ad/time_adp:.2f}x the speed of AD (>1 means AD+ faster).')




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


SAVE_NAME = f'insect_{MODEL_CHOICE}'
print("Image saved to AdvViT/save")  # <-- change this per run, e.g. 'dog_photo', 'run2', etc.

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
# \\wsl.localhost\Ubuntu\home\siddharthsajjive\AdvViT\save maps to
SAVE_NAME = f'animal_{QUERY_LIMIT}_{MODEL_CHOICE}'
print("Image saved to AdvViT/save") 
save_dir = '/home/siddharthsajjive/AdvViT/save'
os.makedirs(save_dir, exist_ok=True)
fig.savefig(os.path.join(save_dir, f'output_ad+_{SAVE_NAME}.png'), dpi=150, bbox_inches='tight')
plt.show()

