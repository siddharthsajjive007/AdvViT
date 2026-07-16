"""
check_adv_accuracy.py

Verifies the ori/ and adv/ image folders + labels.csv produced by
test_AD_batch.py: reloads every saved PNG from disk, runs it back through
the model fresh, and checks:

  1. ROUND-TRIP INTEGRITY (did saving to PNG and reloading corrupt anything):
     - fresh prediction on ori/<file> should match the recorded label in
       ori/labels.csv
     - fresh prediction on adv/<file> should match the recorded label in
       adv/labels.csv

  2. ADVERSARIAL ACCURACY (the standard security metric):
     - fraction of adversarial images the model STILL correctly classifies
       as the original class, despite the perturbation. LOWER is better for
       the attack (means the attack is working). This is 1 - ASR-on-reload.

  3. ASR ON RELOAD (complement of #2, stated the way you described it):
     - fraction of adversarial images where the model's fresh prediction is
       NOT the original label (i.e. still successfully fooled after the
       PNG round-trip).

Run standalone: python check_adv_accuracy.py
"""
import os
import csv
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import cv2
from simp_batch import SimP, DATASET, DATASET_CONFIGS   # <-- update if your batched file has a different name

device = torch.device('cuda', 0)
print('CUDA available:', torch.cuda.is_available())
print('Device:', device)

# ── MUST match whatever generated the images you're checking ──
MODEL_ARCH = 'resnet50'
CHECK_BATCH_SIZE = 100    # how many images go through the model per forward pass while checking

# ── folder produced by test_AD_batch.py ──
out_dir = '/home/siddarth/AdvViT/OUTPUT/batch_run_imagenet_100'   # <-- point this at the actual run folder you want to check
ori_dir = os.path.join(out_dir, 'ori')
adv_dir = os.path.join(out_dir, 'adv')
ori_csv_path = os.path.join(ori_dir, 'labels.csv')
adv_csv_path = os.path.join(adv_dir, 'labels.csv')

_cfg = DATASET_CONFIGS[DATASET]
IMAGE_SIZE = _cfg['size']


def load_model(model_arch, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_arch == 'resnet50':
        net = torch.load(
            "/home/siddarth/ATTACK/resnet50_testing_model.pth",
            map_location=device, weights_only=False
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
            map_location=device
        )
    elif model_arch == 'resnet50_gtsrb32':
        net = torch.load(
            '/home/HDD/ATAF/Model-files/GTSRB/gtsrb_resnet50_32.pth',
            map_location=device
        )
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


def load_labels_csv(csv_path):
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({'filename': r['filename'], 'label': int(r['label'])})
    return rows


def load_images_batch(folder, filenames, image_size):
    imgs = []
    for fname in filenames:
        img = Image.open(os.path.join(folder, fname)).convert('RGB').resize((image_size, image_size))
        imgs.append(T.ToTensor()(img))
    return torch.stack(imgs)

def load_images_batch_npy(folder, filenames, image_size):
    imgs = []
    for fname in filenames:
        npy_fname = fname.replace('.png', '.npy')
        arr = np.load(os.path.join(folder, npy_fname))   # HWC, float32, [0,1], exact bits preserved
        imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(imgs)


print(f"Loading {MODEL_ARCH}...")
model = load_model(MODEL_ARCH, device)
attacker = SimP(model, DATASET, image_size=IMAGE_SIZE)

print(f"Reading {ori_csv_path}")
ori_rows = load_labels_csv(ori_csv_path)
print(f"Reading {adv_csv_path}")
adv_rows = load_labels_csv(adv_csv_path)

assert len(ori_rows) == len(adv_rows), (
    f"ori/labels.csv has {len(ori_rows)} rows but adv/labels.csv has {len(adv_rows)} -- "
    f"these should correspond 1:1, something's inconsistent."
)
N = len(ori_rows)
print(f"Checking {N} image pairs...")

# ── round-trip integrity counters ──
ori_mismatch = 0     # fresh prediction on ori/<file> != recorded ori label
adv_recorded_mismatch = 0   # fresh prediction on adv/<file> != recorded adv label

# ── the actual security metrics ──
still_correct = 0    # model still predicts the ORIGINAL label on the adversarial image (attack failed on reload)
still_fooled = 0     # model predicts something other than the original label (attack still works on reload)
npy_reverted = 0    
npy_still_fooled = 0

per_image_results = []

with torch.no_grad():
    for start in range(0, N, CHECK_BATCH_SIZE):
        end = min(start + CHECK_BATCH_SIZE, N)
        ori_chunk = ori_rows[start:end]
        adv_chunk = adv_rows[start:end]

        ori_filenames = [r['filename'] for r in ori_chunk]
        adv_filenames = [r['filename'] for r in adv_chunk]
        ori_recorded_labels = torch.tensor([r['label'] for r in ori_chunk], device=device)
        adv_recorded_labels = torch.tensor([r['label'] for r in adv_chunk], device=device)

        ori_imgs = load_images_batch(ori_dir, ori_filenames, IMAGE_SIZE).to(device)
        adv_imgs = load_images_batch(adv_dir, adv_filenames, IMAGE_SIZE).to(device)

        ori_fresh_pred = attacker.get_label_batch(ori_imgs)
        adv_fresh_pred = attacker.get_label_batch(adv_imgs)

        adv_imgs_npy = load_images_batch_npy(adv_dir, adv_filenames, IMAGE_SIZE).to(device)
        adv_fresh_pred_npy = attacker.get_label_batch(adv_imgs_npy)

        for i in range(end - start):
            global_idx = start + i
            ori_ok = (ori_fresh_pred[i].item() == ori_recorded_labels[i].item())
            adv_ok = (adv_fresh_pred[i].item() == adv_recorded_labels[i].item())
            fooled = (adv_fresh_pred[i].item() != ori_recorded_labels[i].item())

            if not ori_ok:
                ori_mismatch += 1
            if not adv_ok:
                adv_recorded_mismatch += 1
            if fooled:
                still_fooled += 1
            else:
                still_correct += 1

            npy_fooled = (adv_fresh_pred_npy[i].item() != ori_recorded_labels[i].item())
            if npy_fooled:
                npy_still_fooled += 1
            else:
                npy_reverted += 1

            per_image_results.append({
                'index': global_idx,
                'ori_filename': ori_filenames[i],
                'adv_filename': adv_filenames[i],
                'ori_recorded_label': ori_recorded_labels[i].item(),
                'adv_recorded_label': adv_recorded_labels[i].item(),
                'ori_fresh_pred': ori_fresh_pred[i].item(),
                'adv_fresh_pred': adv_fresh_pred[i].item(),
                'ori_matches_recorded': ori_ok,
                'adversarial_accuracy': adv_ok,        # <-- this IS your adversarial accuracy, per image
                'still_fooled_on_reload': fooled,
            })

        print(f"  checked {end}/{N}")

# ── write per-image verification results ──
verify_csv_path = os.path.join(out_dir, 'adv_accuracy_check.csv')
with open(verify_csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'index', 'ori_filename', 'adv_filename', 'ori_recorded_label', 'adv_recorded_label',
        'ori_fresh_pred', 'adv_fresh_pred', 'ori_matches_recorded', 'adversarial_accuracy',
        'still_fooled_on_reload'
    ])
    writer.writeheader()
    writer.writerows(per_image_results)

print('\n' + '=' * 60)
print('ADVERSARIAL ACCURACY (reloaded adv image predicts the recorded adversarial label)')
print('=' * 60)
print(f'Adversarial accuracy: {N - adv_recorded_mismatch}/{N} '
      f'({100*(N-adv_recorded_mismatch)/N:.1f}%)   <- HIGHER means the saved adv image '
      f'still reliably fools the model into the same class the attack achieved')

print('\n' + '=' * 60)
print('ATTACK SUCCESS ON RELOAD (reloaded adv image does NOT predict the original label)')
print('=' * 60)
print(f'Still fooled (ASR):    {still_fooled}/{N} ({100*still_fooled/N:.1f}%)')
print(f'Reverted to original:  {still_correct}/{N} ({100*still_correct/N:.1f}%)   '
      f'<- these would indicate the attack effect didn\'t survive the PNG round-trip')

print('\n' + '=' * 60)
print('ATTACK SUCCESS ON RELOAD FROM .NPY (zero-precision-loss diagnostic)')
print('=' * 60)
print(f'Still fooled (ASR):    {npy_still_fooled}/{N} ({100*npy_still_fooled/N:.1f}%)')
print(f'Reverted to original:  {npy_reverted}/{N} ({100*npy_reverted/N:.1f}%)')