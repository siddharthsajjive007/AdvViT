"""
check_adv_accuracy.py

ASR: read directly from the main results CSV (test_AD_batch.py's own
success/asr column, computed inside attack_untargeted_batch against ground
truth -- this IS the trusted number now, no separate in-memory vs npy
comparison needed).

Adversarial accuracy: reloads the saved adv/ PNGs, runs them back through
the model, and compares against the recorded adversarial label (adv/labels.csv)
-- but ONLY for the subset of images where the attack itself reported
success=True. Images that were never genuinely adversarial (attack failed)
are excluded from this metric entirely, rather than diluting the denominator.

Also reports round-trip integrity on the original images (sanity check only).
"""
import os
import csv
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np

from simp_batch import SimP, DATASET, DATASET_CONFIGS   # <-- update if your batched file has a different name

device = torch.device('cuda', 0)
print('CUDA available:', torch.cuda.is_available())
print('Device:', device)

# ── MUST match whatever generated the images you're checking ──
MODEL_ARCH = 'resnet50'
CHECK_BATCH_SIZE = 100

# ── folder produced by test_AD_batch.py ──
out_dir = '/home/siddarth/AdvViT/OUTPUT/batch_run_imagenet_200'   # <-- point this at the actual run folder
ori_dir = os.path.join(out_dir, 'ori')
adv_dir = os.path.join(out_dir, 'adv')
ori_csv_path = os.path.join(ori_dir, 'labels.csv')
adv_csv_path = os.path.join(adv_dir, 'labels.csv')
main_results_csv_path = os.path.join(out_dir, f'results_{DATASET}_{MODEL_ARCH}.csv')  # <-- written by test_AD_batch.py

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


def load_main_results_success(csv_path):
    """Reads the 'asr' column from test_AD_batch.py's own main results CSV --
    this is the ground-truth-verified success flag computed inside
    attack_untargeted_batch itself (final_pred != y0_batch, & found_initial).
    Returns a dict: index (int) -> success (bool)."""
    success_by_index = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r['index'] == 'TOTAL':      # skip the summary row if present
                continue
            success_by_index[int(r['index'])] = (r['asr'] in ('True', 'true', '1'))
    return success_by_index


def load_images_batch(folder, filenames, image_size):
    imgs = []
    for fname in filenames:
        img = Image.open(os.path.join(folder, fname)).convert('RGB').resize((image_size, image_size))
        imgs.append(T.ToTensor()(img))
    return torch.stack(imgs)


print(f"Loading {MODEL_ARCH}...")
model = load_model(MODEL_ARCH, device)
attacker = SimP(model, DATASET, image_size=IMAGE_SIZE)

print(f"Reading {ori_csv_path}")
ori_rows = load_labels_csv(ori_csv_path)
print(f"Reading {adv_csv_path}")
adv_rows = load_labels_csv(adv_csv_path)
print(f"Reading {main_results_csv_path}")
success_by_index = load_main_results_success(main_results_csv_path)

assert len(ori_rows) == len(adv_rows), (
    f"ori/labels.csv has {len(ori_rows)} rows but adv/labels.csv has {len(adv_rows)} -- "
    f"these should correspond 1:1, something's inconsistent."
)
N = len(ori_rows)
print(f"Checking {N} images...")

# ── ASR: read directly from the attack's own trusted output, not recomputed here ──
asr_count = sum(1 for v in success_by_index.values() if v)
print(f"\nASR (from attack's own ground-truth-verified success flag): "
      f"{asr_count}/{N} ({100*asr_count/N:.1f}%)")

# ── round-trip integrity on originals (sanity check only) ──
ori_mismatch = 0

# ── adversarial accuracy, computed ONLY over the successful subset ──
successful_indices = [i for i in range(N) if success_by_index.get(i, False)]
adv_accuracy_match = 0   # reloaded PNG prediction matches the recorded adversarial label
adv_accuracy_total = len(successful_indices)

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

        for i in range(end - start):
            global_idx = start + i
            was_successful = success_by_index.get(global_idx, False)

            ori_ok = (ori_fresh_pred[i].item() == ori_recorded_labels[i].item())
            if not ori_ok:
                ori_mismatch += 1

            adv_acc_match_this_image = None
            if was_successful:
                adv_acc_match_this_image = (adv_fresh_pred[i].item() == adv_recorded_labels[i].item())
                if adv_acc_match_this_image:
                    adv_accuracy_match += 1

            per_image_results.append({
                'index': global_idx,
                'ori_filename': ori_filenames[i],
                'adv_filename': adv_filenames[i],
                'ori_recorded_label': ori_recorded_labels[i].item(),
                'adv_recorded_label': adv_recorded_labels[i].item(),
                'ori_fresh_pred': ori_fresh_pred[i].item(),
                'adv_fresh_pred': adv_fresh_pred[i].item(),
                'ori_matches_recorded': ori_ok,
                'was_successful_attack': was_successful,
                'included_in_adv_accuracy': was_successful,
                'adv_accuracy_match': adv_acc_match_this_image,   # None if not in the successful subset
            })

        print(f"  checked {end}/{N}")

# ── write per-image verification results ──
verify_csv_path = os.path.join(out_dir, 'adv_accuracy_check.csv')
with open(verify_csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'index', 'ori_filename', 'adv_filename', 'ori_recorded_label', 'adv_recorded_label',
        'ori_fresh_pred', 'adv_fresh_pred', 'ori_matches_recorded', 'was_successful_attack',
        'included_in_adv_accuracy', 'adv_accuracy_match'
    ])
    writer.writeheader()
    writer.writerows(per_image_results)

print('\n' + '=' * 60)
print('ORIGINAL IMAGE ROUND-TRIP CHECK (sanity check only)')
print('=' * 60)
print(f'ori/ predictions matching recorded ground-truth label: {N - ori_mismatch}/{N} '
      f'({100*(N-ori_mismatch)/N:.1f}%)')

print('\n' + '=' * 60)
print('ADVERSARIAL ACCURACY (successful-attack subset only, NOT the full dataset)')
print('=' * 60)
print(f'Subset size (images with success=True): {adv_accuracy_total}/{N}')
if adv_accuracy_total > 0:
    print(f'Adversarial accuracy: {adv_accuracy_match}/{adv_accuracy_total} '
          f'({100*adv_accuracy_match/adv_accuracy_total:.1f}%)   '
          f'<- of the images that were genuinely adversarial, this fraction '
          f'still predicts the same adversarial class after PNG reload')
else:
    print('No successful attacks in this run -- nothing to compute adversarial accuracy over.')

print(f'\nPer-image results saved to: {verify_csv_path}')
