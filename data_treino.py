#!/usr/bin/env python3
"""
Treino CLIP (cabeça linear) para Soil types
Versão completa com --csv/--train-csv como alias sem conflito
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # evita warning do tokenizers

import time
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import CLIPProcessor, CLIPModel


# ===============================
# Dataset Customizado
# ===============================
class SoilDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = str(row['image_path']).strip()
        cand1 = self.root_dir / rel_path
        cand2 = self.root_dir / str(row.get('label', '')) / rel_path
        if cand1.exists():
            img_path = cand1
        elif cand2.exists():
            img_path = cand2
        else:
            raise FileNotFoundError(f"Imagem não encontrada: tentou {cand1} e {cand2}")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        text = str(row['text']) if 'text' in row.index else ""
        label = int(row['label']) if 'label' in row.index else -1
        return img, text, label


# ===============================
# Helpers
# ===============================
def normalize_feat(tensor):
    return tensor / (tensor.norm(dim=-1, keepdim=True) + 1e-10)


def get_current_lr(optimizer):
    return optimizer.param_groups[0]['lr']


# ===============================
# Treino / Validação
# ===============================
def train_one_epoch(clip, classifier, dataloader, optimizer, loss_fn, device):
    clip.eval()
    classifier.train()
    running_loss = 0.0
    n = 0
    for imgs, texts, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            img_feats = clip.get_image_features(pixel_values=imgs)
            img_feats = normalize_feat(img_feats)

        logits = classifier(img_feats)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)
        running_loss += loss.item() * batch_size
        n += batch_size
    return running_loss / max(1, n)


def eval_one_epoch(clip, classifier, dataloader, loss_fn, device):
    clip.eval()
    classifier.eval()
    running_loss = 0.0
    n = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, texts, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            img_feats = clip.get_image_features(pixel_values=imgs)
            img_feats = normalize_feat(img_feats)

            logits = classifier(img_feats)
            loss = loss_fn(logits, labels)

            batch_size = imgs.size(0)
            running_loss += loss.item() * batch_size
            n += batch_size

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size
    avg_loss = running_loss / max(1, n)
    acc = correct / max(1, total)
    return avg_loss, acc


# ===============================
# Main training routine
# ===============================
def run(args):
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    print("Device:", device)

    print("Carregando CLIP:", args.model_name)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    clip = CLIPModel.from_pretrained(args.model_name)
    clip.to(device)
    clip.eval()

    img_mean = processor.feature_extractor.image_mean
    img_std = processor.feature_extractor.image_std
    print("Image mean/std:", img_mean, img_std)

    # Transforms
    transform_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.9,1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])

    # Datasets & loaders
    train_ds = SoilDataset(args.train_csv, args.root, transform=transform_train)
    val_ds = SoilDataset(args.val_csv, args.root, transform=transform_eval)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=0, pin_memory=True)

    print(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")
    print("num_workers (train):", args.num_workers)

    # Cabeça linear
    proj_dim = clip.config.projection_dim
    classifier = nn.Linear(proj_dim, args.num_classes).to(device)
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print("Parâmetros treináveis (classifier):", trainable_params)

    optimizer = optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience)
    last_lr = get_current_lr(optimizer)

    best_val_loss = float("inf")
    best_epoch = -1

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(clip, classifier, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = eval_one_epoch(clip, classifier, val_loader, loss_fn, device)

        # Scheduler
        scheduler.step(val_loss)
        new_lr = get_current_lr(optimizer)
        if new_lr != last_lr:
            print(f"[Epoch {epoch}] ReduceLROnPlateau: lr reduzida de {last_lr:.6g} -> {new_lr:.6g} (val_loss={val_loss:.4f})")
            last_lr = new_lr

        # Salvar melhor modelo
        os.makedirs(args.save_dir, exist_ok=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(classifier.state_dict(), os.path.join(args.save_dir, "classifier_best.pth"))
            print(f"[Epoch {epoch}] Novo best val_loss={val_loss:.4f} -> salvando classifier_best.pth")

        torch.save(classifier.state_dict(), os.path.join(args.save_dir, "classifier_last.pth"))

        t1 = time.time()
        print(f"Epoch {epoch}/{args.epochs} - train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={(t1-t0):.1f}s")

    elapsed = time.time() - start_time
    print(f"Treino finalizado em {elapsed/60:.2f} min. Melhor epoch {best_epoch} com val_loss={best_val_loss:.4f}")

    # Salvar config
    cfg = vars(args)
    cfg.update({"best_epoch": best_epoch, "best_val_loss": best_val_loss})
    with open(os.path.join(args.save_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# ===============================
# Parse args
# ===============================
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Treino CLIP (cabeça linear) para Soil types")

    # --csv e --train-csv como alias
    p.add_argument(
        "--csv", "--train-csv", dest="train_csv", type=str, required=True,
        help="CSV de treino (pode usar --csv ou --train-csv)"
    )

    p.add_argument("--val-csv", type=str, default="val.csv",
                   help="CSV de validação (image_path,text,label)")

    p.add_argument("--root", type=str, default="images", help="Pasta raiz das imagens")

    # Hiperparâmetros
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-classes", type=int, default=5)
    p.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")

    # Salvar
    p.add_argument("--save-dir", type=str, default="models/clip_soil_finetuned")

    # DataLoader
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--early-stop", type=int, default=None)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--lr-patience", type=int, default=5)

    args = p.parse_args()
    return args


# ===============================
# Entrypoint
# ===============================
if __name__ == "__main__":
    args = parse_args()
    run(args)

