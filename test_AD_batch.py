"""
Batched AdvViT test -- loads TOTAL_IMAGES images from a zip, processes them
in chunks of INFERENCE_BATCH_SIZE at a time through attack_untargeted_batch,
and reports aggregate stats (success rate, avg/median distortion, avg
queries) across the whole run, same convention as run_simp.py's
num_runs/batch_size split.
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
import os
import math
import time
import zipfile
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import csv
from simp_batch import SimP, DATASET, DATASET_CONFIGS   # <-- update if your batched file has a different name
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
import lpips
import cv2


device = torch.device('cuda', 0)
print('CUDA available:', torch.cuda.is_available())
print('Device:', device)

'''
CHANGE DATASET IN SIMP_BATCH THEN CHANGE FOLDER NAME, MODEL_ARCH AND ZIP_PATH HERE

'''
#PARAMETERS TO EXPERIMENT
QUERY_LIMIT = 10000
USE_SIGN_OPT_PLUS = True
BISECT_ITERS = 20             #was 20
BISECT_ITERS_LOCAL = 15        #was 15
DIMEN_SIZE = 4                # is 4 for imagenet as per DATASET_CONFIG dictionary, if set here this will be 
                              # the value tested, if the argument is removed or set to 'None' on attack_untargeted_batch
                              # in this file, it reverts to the config dataset dictionary value


# margin size scale disabled in bisect batch for this run

# ── how many images total, and how many go through one attack_untargeted_batch call ──
TOTAL_IMAGES = 3599
# Max safe on RTX 3090 24GB for this batched Sign-OPT path (~16GB peak near 1025)
INFERENCE_BATCH_SIZE = 500
MODEL_ARCH = 'DeiT_B'      # 'resnet50' | 'DeiT_B' | 'DeiT_S' | 'DeiT_T' | 'resnet18_cifar10' | 'resnet50_gtsrb32'| 'deit_cifar10' | 'ViT'
ZIP_PATH = "/home/HDD/ATAF/Datasets/ImageNetDataset/ATAF-Framework-Ready/ImageNet-3599-Targeted.zip"   #IMAGENET
# ZIP_PATH = "/home/HDD/ATAF/Datasets/CIFAR10-Dataset/CIFAR-10-60k-targeted.zip"    #CIFAR10
# ZIP_PATH = "/home/HDD/ATAF/Datasets/GTSRB//GTSRB_test_ataf.zip"

# ------IMAGE SAVE FOLDER PATH-----------------------
out_dir = f'/home/siddarth/AdvViT/OUTPUT/batch_run_imagenet_{MODEL_ARCH}_{TOTAL_IMAGES}(marginend1.02)'
os.makedirs(out_dir, exist_ok=True)
ori_dir = os.path.join(out_dir, 'ori')
adv_dir = os.path.join(out_dir, 'adv')
os.makedirs(ori_dir, exist_ok=True)
os.makedirs(adv_dir, exist_ok=True)
# ── CSV output path ──
csv_path = os.path.join(out_dir, f'results_{DATASET}_{MODEL_ARCH}.csv')
csv_rows = []
ori_csv_path = os.path.join(ori_dir, 'labels.csv')
adv_csv_path = os.path.join(adv_dir, 'labels.csv')
ori_label_rows = []
adv_label_rows = []
#==============================================================================

config = DATASET_CONFIGS[DATASET]
IMAGE_SIZE = config['size']
PATCH_SIZE = config['patch_size']

# =====================Quick sanity check==============================================

# time loading the whole zip and running the attack.
# Quick sanity check: catch a MODEL_ARCH / DATASET / ZIP_PATH mismatch before
# wasting time loading the whole zip and running the attack.
MODEL_ARCH_DATASET = {
    'resnet50': 'IMAGENET_3599', 'DeiT_T': 'IMAGENET_3599', 'DeiT_S': 'IMAGENET_3599',
    'DeiT_B': 'IMAGENET_3599', 'ViT': 'IMAGENET_3599',
    'resnet18_cifar10': 'CIFAR', 'resnet50_cifar10': 'CIFAR', 'deit_cifar10': 'CIFAR',
    'resnet50_gtsrb32': 'GTSRB'}

_expected = MODEL_ARCH_DATASET.get(MODEL_ARCH)
if _expected is not None and _expected != DATASET:
    raise ValueError(
        f"MODEL_ARCH='{MODEL_ARCH}' expects DATASET='{_expected}', "
        f"but simp_batch has DATASET='{DATASET}'. Fix one or the other before running."
    )

# Check ZIP_PATH contains a keyword matching DATASET -- catches e.g. DATASET='CIFAR'
# while ZIP_PATH still points at the ImageNet or GTSRB zip.
DATASET_ZIP_KEYWORDS = {
    'IMAGENET': ['imagenet'],
    'IMAGENET_3599': ['imagenet'],
    'CIFAR': ['cifar'],
    'GTSRB': ['gtsrb'],
    'MNIST': ['mnist'],
}
_zip_keywords = DATASET_ZIP_KEYWORDS.get(DATASET, [])
if _zip_keywords and not any(kw in ZIP_PATH.lower() for kw in _zip_keywords):
    raise ValueError(
        f"DATASET='{DATASET}' expects ZIP_PATH to reference one of {_zip_keywords}, "
        f"but ZIP_PATH='{ZIP_PATH}' doesn't. Fix ZIP_PATH or DATASET before running."
    )

#================================END OF CHECK===================================================

# NOW ALSO RETURNS THE ORIGINAL LABELS FROM THE CSV FILE

def load_images_from_zip(zip_path, image_size, num_images):
    """
    Loads the first num_images images (sorted by filename, for consistent
    ordering across runs) from the zip's images/ folder, along with their
    GROUND-TRUTH original_label from the zip's own labels.csv.
    Returns:
        x0_batch: [num_images, 3, image_size, image_size] CPU tensor, range [0,1]
        chosen_files: list of filenames used (bare, no 'images/' prefix), same order as x0_batch's rows
        y0_batch: [num_images] LongTensor of ground-truth original_label values, same order
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        # ---- read the ground-truth label CSV bundled in the zip ----
        with z.open('labels.csv') as f:
            reader = csv.DictReader(line.decode('utf-8') for line in f)
            label_lookup = {row['filename']: int(row['original_label']) for row in reader}

        image_files = sorted(
            f for f in z.namelist()
            if f.startswith('images/') and f.lower().endswith(('.jpeg', '.jpg', '.png'))
        )
        if num_images > len(image_files):
            raise ValueError(
                f"Requested {num_images} images but the zip only contains {len(image_files)} "
                f"under 'images/'."
            )
        chosen_files_full = image_files[:num_images]          # e.g. 'images/ILSVRC2012_val_00048454.JPEG'
        chosen_files = [f.split('/', 1)[1] for f in chosen_files_full]  # strip 'images/' prefix to match labels.csv

        imgs = []
        labels = []
        for full_path, bare_name in zip(chosen_files_full, chosen_files):
            if bare_name not in label_lookup:
                raise KeyError(f"'{bare_name}' has no entry in labels.csv -- can't get its ground-truth label.")
            with z.open(full_path) as f:
                img = Image.open(f).convert('RGB').resize((image_size, image_size))
                imgs.append(T.ToTensor()(img))
            labels.append(label_lookup[bare_name])

    return torch.stack(imgs), chosen_files, torch.tensor(labels, dtype=torch.long)


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


print(f"Loading {MODEL_ARCH}...")
model = load_model(MODEL_ARCH, device)
attacker = SimP(model, DATASET, image_size=IMAGE_SIZE)
print(f"Loading {TOTAL_IMAGES} images from {ZIP_PATH}...")

x0_all, filenames_all, y0_all = load_images_from_zip(ZIP_PATH, IMAGE_SIZE, TOTAL_IMAGES)

# ---- FILTER: only attack images the model already correctly classifies ----
# Without this, some images have y0 (ground truth) != model's clean prediction,
# meaning the "attack" needs zero effort (or none at all) -- these produce
# degenerate near-zero-distortion results that don't represent genuine attacks.
clean_pred_list = []
with torch.no_grad():
    for start in range(0, TOTAL_IMAGES, INFERENCE_BATCH_SIZE):
        end = min(start + INFERENCE_BATCH_SIZE, TOTAL_IMAGES)
        clean_pred_list.append(attacker.get_label_batch(x0_all[start:end].to(device)).cpu())
clean_pred_all = torch.cat(clean_pred_list, dim=0)

correctly_classified = (clean_pred_all == y0_all)

print(f"Model correctly classifies {correctly_classified.sum().item()}/{len(y0_all)} images "
      f"before attack -- only attacking those.")

x0_all = x0_all[correctly_classified]
y0_all = y0_all[correctly_classified]
filenames_all = [f for f, keep in zip(filenames_all, correctly_classified.tolist()) if keep]
TOTAL_IMAGES = x0_all.shape[0]   # update to reflect the actual, filtered count

print(f"Proceeding with {TOTAL_IMAGES} images.")
patch_num = IMAGE_SIZE // PATCH_SIZE


# ── aggregate stats, same convention as run_simp.py ──
all_distortions = []
all_successes = []
all_queries = []
all_ssim = []
all_psnr = []
all_linf = []
all_lpips = []

loss_fn_lpips = lpips.LPIPS(net='alex').to(device)

num_chunks = math.ceil(TOTAL_IMAGES / INFERENCE_BATCH_SIZE)
t_start_total = time.time()

for chunk_idx in range(num_chunks):
    start = chunk_idx * INFERENCE_BATCH_SIZE
    end = min(start + INFERENCE_BATCH_SIZE, TOTAL_IMAGES)
    x0_chunk = x0_all[start:end].to(device)
    y0_chunk = y0_all[start:end].to(device)
    files_chunk = filenames_all[start:end]

    print(f"\n=== Chunk {chunk_idx+1}/{num_chunks}: images {start}-{end-1} "
          f"({end-start} images) ===")
    t0 = time.time()
    adv_chunk, distortion_chunk, success_chunk, queries_chunk, prub_chunk = attacker.attack_untargeted_batch(
        x0_chunk, y0_chunk, patch_num,
        query_limit=QUERY_LIMIT, use_sign_opt_plus=USE_SIGN_OPT_PLUS, bisect_iters=BISECT_ITERS, 
        bisect_iters_local=BISECT_ITERS_LOCAL,   # was 15, trying for lower values to converge less on to the boundary
        dimen_size=DIMEN_SIZE
        )
    elapsed = time.time() - t0
    n_this_chunk = end - start
    print(f"Chunk finished in {elapsed:.1f}s ({elapsed/n_this_chunk:.1f}s/image avg for this chunk)")

    with torch.no_grad():
        adv_pred_chunk = attacker.get_label_batch(adv_chunk)

    for i in range(n_this_chunk):
        global_idx = start + i
        ori_np = x0_chunk[i].detach().cpu().permute(1, 2, 0).numpy()
        adv_np = adv_chunk[i].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        ssim_val = ssim_fn(ori_np, adv_np, channel_axis=2, data_range=1.0)
        psnr_val = psnr_fn(ori_np, adv_np, data_range=1.0)

        succ = bool(success_chunk[i].item())
        dist = float(distortion_chunk[i].item())
        nq = int(queries_chunk[i].item())

         # ---- L-infinity distortion (max pixel-wise change, torch tensors -- more precise than the numpy round-trip) ----
        linf_val = (adv_chunk[i] - x0_chunk[i]).abs().max().item()

        # ============= lpips ==================================
        with torch.no_grad():
            img1_lpips = (x0_chunk[i].unsqueeze(0) * 2 - 1).to(device)
            img2_lpips = (adv_chunk[i].unsqueeze(0).clamp(0, 1) * 2 - 1).to(device)
            lpips_val = loss_fn_lpips(img1_lpips, img2_lpips).item()


        all_successes.append(succ)
        all_queries.append(nq)
        if succ:
            all_distortions.append(dist)
            all_ssim.append(ssim_val)
            all_psnr.append(psnr_val)
            all_linf.append(linf_val)
            all_lpips.append(lpips_val)



        print(f'  [{global_idx}] {files_chunk[i]}  success={succ}  '
              f'L2={dist:.4f}   Linf={linf_val:.4f}  LPIPS={lpips_val:.4f}  queries={nq}  SSIM={ssim_val:.4f}  PSNR={psnr_val:.2f}dB  '
              f'orig_class={y0_chunk[i].item()}  adv_class={adv_pred_chunk[i].item()}')
        
        csv_rows.append({
            'index': global_idx,
            'filename': files_chunk[i],
            'asr': succ,
            'l2_distortion': dist,
            'linf_distortion': linf_val,
            'lpips': lpips_val,
            'queries': nq,
            'ssim': ssim_val,
            'psnr': psnr_val,
            'orig_class': y0_chunk[i].item(),
            'adv_class': adv_pred_chunk[i].item()
        })
        #===================SAVING ORIGINAL AND ADV IMAGES==========================
        ori_filename = f'ori_{global_idx}_{DATASET}_{MODEL_ARCH}.png'
        adv_filename = f'adv_{global_idx}_{DATASET}_{MODEL_ARCH}.png'


        plt.imsave(os.path.join(ori_dir, ori_filename), ori_np)
        plt.imsave(os.path.join(adv_dir, adv_filename), adv_np)

        ori_label_rows.append({'filename': ori_filename, 'label': y0_chunk[i].item()})
        adv_label_rows.append({'filename': adv_filename, 'label': adv_pred_chunk[i].item()})
        #=================================================================================

        fig, axes = plt.subplots(1, 3, figsize=(13, 5))
        diff = adv_np - ori_np
        diff_vis = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        axes[0].imshow(ori_np); axes[0].set_title(f'Original\n(class {y0_chunk[i].item()})'); axes[0].axis('off')
        axes[1].imshow(adv_np); axes[1].set_title(f'Adversarial\n(class {adv_pred_chunk[i].item()})'); axes[1].axis('off')
        axes[2].imshow(diff_vis); axes[2].set_title('Perturbation'); axes[2].axis('off')
        fig.suptitle(f'L2={dist:.3f}  Linf={linf_val:.3f}  LPIPS={lpips_val:.4f} '    
                     f'SSIM={ssim_val:.4f}  PSNR={psnr_val:.2f}dB  queries={nq}', fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        fig.savefig(os.path.join(out_dir, f'img{global_idx}_{DATASET}_{MODEL_ARCH}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

total_elapsed = time.time() - t_start_total

# after the chunk loop finishes, before writing the CSV:
for row in csv_rows:
    row['total_time_seconds'] = round(total_elapsed, 1)


csv_rows.append({
    'index': 'TOTAL',
    'filename': '',
    'asr': f'{sum(all_successes)}/{TOTAL_IMAGES}',
    'l2_distortion': round(np.mean(all_distortions), 4) if all_distortions else '',
    'linf_distortion': round(np.mean(all_linf), 4) if all_linf else '',
    'lpips': round(np.mean(all_lpips), 4) if all_lpips else '',
    'queries': round(np.mean(all_queries)),
    'ssim': round(np.mean(all_ssim), 4) if all_ssim else '',
    'psnr': round(np.mean(all_psnr), 2) if all_psnr else '',
    'orig_class': '',
    'adv_class': '',
    'total_time_seconds': round(total_elapsed, 1),
})


# ── final aggregate report ──
print('\n' + '=' * 60)
print(f'RUN SUMMARY -- {TOTAL_IMAGES} images, batch size {INFERENCE_BATCH_SIZE}, '
      f'{num_chunks} chunks, model={MODEL_ARCH}, dataset={DATASET}')
print('=' * 60)
print(f'Total time:        {total_elapsed:.1f}s ({total_elapsed/60:.2f} min), '
      f'{total_elapsed/TOTAL_IMAGES:.1f}s/image average')
print(f'asr:      {sum(all_successes)}/{TOTAL_IMAGES} '
      f'({100*sum(all_successes)/TOTAL_IMAGES:.1f}%)')
if all_distortions:
    print(f'Avg L2 distortion      :    {np.mean(all_distortions):.4f}')
    print(f'Median L2 distortion   :    {np.median(all_distortions):.4f}')
    print(f'Avg Linf distortion    :    {np.mean(all_linf):.4f}')
    print(f'Avg LPIPS              :    {np.mean(all_lpips):.4f}')
    print(f'Avg SSIM               :    {np.mean(all_ssim):.4f}')
    print(f'Avg PSNR               :    {np.mean(all_psnr):.2f} dB')
else:
    print('No successful attacks -- no distortion/SSIM/PSNR stats to report.')
print(f'Avg queries used (all images):    {np.mean(all_queries):.0f}')
print(f'Results saved to: {out_dir}')


# ── write per-image results to CSV ──
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
    'index', 'filename', 'asr', 'l2_distortion', 'linf_distortion', 'lpips',
    'queries', 'ssim', 'psnr', 'orig_class', 'adv_class', 'total_time_seconds'
    ])
    writer.writeheader()
    writer.writerows(csv_rows)
print(f'Per-image results saved to: {csv_path}')

# -----write ori and adv label csv--------------------------
with open(ori_csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['filename', 'label'])
    writer.writeheader()
    writer.writerows(ori_label_rows)

with open(adv_csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['filename', 'label'])
    writer.writeheader()
    writer.writerows(adv_label_rows)

print(f'Original image labels saved to: {ori_csv_path}')
print(f'Adversarial image labels saved to: {adv_csv_path}')
print(f"Images finished in {total_elapsed:.1f}s")