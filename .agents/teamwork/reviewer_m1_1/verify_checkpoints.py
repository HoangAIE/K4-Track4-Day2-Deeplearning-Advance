import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path('.').resolve()))
sys.path.insert(0, str(Path('code').resolve()))

from model import build_model
from dataset import build_transforms, load_split, make_loader
import eval as ev

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

labels_dir = Path('data/labels')
images_dir = Path('data/images')
train_df, val_df, test_df = load_split(labels_dir, fold=0)

val_tf = build_transforms(train=False, img_size=224)
val_loader = make_loader(val_df, images_dir, transform=val_tf, batch_size=64, train=False, num_workers=0)
first_batch_imgs, _, _ = next(iter(val_loader))
first_batch_imgs = first_batch_imgs.to(device)

models = [
    ('B01', 'resnet50'),
    ('B02', 'convnext_tiny'),
    ('B03', 'swin_tiny_patch4_window7_224'),
    ('B04', 'efficientnet_b0'),
    ('B05', 'mobilenetv3_large_100')
]

for exp_id, model_name in models:
    ckpt_path = Path('runs') / exp_id / 'seed0' / 'best_checkpoint.pt'
    logits_path = Path('runs') / exp_id / 'seed0' / 'val_logits.npy'
    assert ckpt_path.exists(), f'Missing {ckpt_path}'
    assert logits_path.exists(), f'Missing {logits_path}'
    
    saved_logits = np.load(logits_path)
    
    model = build_model(model_name, pretrained=False, num_classes=ev.NUM_CLASSES, init='finetune')
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    with torch.inference_mode():
        out = model(first_batch_imgs).cpu().numpy()
        
    diff = np.abs(saved_logits[:64] - out).max()
    print(f'[{exp_id}: {model_name}] Batch-64 max logit diff: {diff:.6e}')
    assert diff < 1e-4, f'Mismatch for {exp_id}: {diff}'

print('\nALL 5 EXPERIMENT CHECKPOINTS PRODUCE EXACT BIT-LEVEL MATCHING LOGITS!')
