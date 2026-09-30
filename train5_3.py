import torch.nn.functional as F
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from dataset import MyDataset
import time
import numpy as np


class LabelFixDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label - 1  # 1-6 → 0-5


IMG_SIZE = 160

data_train_transformer = transforms.Compose([
    transforms.Resize(192),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

data_val_transformer = transforms.Compose([
    transforms.Resize(192),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        from torchvision.models import ResNet34_Weights
        bb = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        self.conv1   = bb.conv1
        self.bn1     = bb.bn1
        self.relu    = bb.relu
        self.maxpool = bb.maxpool
        self.layer1  = bb.layer1
        self.layer2  = bb.layer2
        self.layer3  = bb.layer3
        self.layer4  = bb.layer4

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.mean(dim=[2, 3])
        return self.classifier(x)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-F.cross_entropy(inputs, targets, reduction='none'))
        return ((1 - pt) ** self.gamma * ce).mean()


def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)



def train_epoch(model, loader, criterion, optimizer, device, use_mixup):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_mixup and np.random.random() < 0.5:
            mixed, ya, yb, lam = mixup_data(inputs, labels)
            outputs = model(mixed)
            loss = mixup_criterion(criterion, outputs, ya, yb, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # 每类准确率 | Per-class accuracy
    per_class = {}
    for c in range(6):
        idxs = [i for i, l in enumerate(all_labels) if l == c]
        if idxs:
            acc = sum(all_preds[i] == all_labels[i] for i in idxs) / len(idxs) * 100
            per_class[c] = acc

    return total_loss / len(loader), 100. * correct / total, per_class


if __name__ == '__main__':
    device = get_device()
    print(f"Using device: {device}")

    train_dataset = LabelFixDataset(MyDataset('data/train/train.txt', data_train_transformer))
    val_dataset   = LabelFixDataset(MyDataset('data/test/val.txt',   data_val_transformer))
    print(f"Train set: {len(train_dataset)}, Val set: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

    model = EmotionClassifier(num_classes=6).to(device)
    print(f"Parameter count: {sum(p.numel() for p in model.parameters()):,}")

    WARMUP_EPOCHS = 5
    TOTAL_EPOCHS  = 80
    WARMUP_BB_LR  = 1e-5
    WARMUP_CLS_LR = 5e-4
    MAIN_BB_LR    = 5e-5
    MAIN_CLS_LR   = 5e-4

    # 类别权重（标签 0-5 对应训练集样本数 1229/1512/2340/2758/3091/2119） | Class weights (labels 0-5 correspond to training sample counts 1229/1512/2340/2758/3091/2119)
    CLASS_COUNTS  = torch.tensor([1229, 1512, 2340, 2758, 3091, 2119], dtype=torch.float)
    class_weights = (CLASS_COUNTS.sum() / (6 * CLASS_COUNTS)).to(device)
    criterion = FocalLoss(weight=class_weights, gamma=2.0)

    best_val_acc     = 0.0
    patience         = 20
    patience_counter = 0
    start_epoch      = 0
    in_main_phase    = False

    CKPT_PATH       = 'checkpoint_v3_focal.pth'
    BEST_MODEL_PATH = 'best_model_v3_focal.pth'

    print(f"\n{'='*70}")
    print(f" Improved scheme v3.3 — ResNet34 + FocalLoss + 160x160")
    print(f"{'='*70}")
    print(f"  Input: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Class weights: {[f'{w:.3f}' for w in class_weights.cpu().tolist()]}")
    print(f"  Warmup {WARMUP_EPOCHS} epochs (no Mixup)")
    print(f"  Main training {TOTAL_EPOCHS-WARMUP_EPOCHS} epochs (Mixup α=0.4)")
    print(f"  Backbone: ResNet34 (21M parameters)")
    print(f"  Loss: FocalLoss γ=2.0 + class weights")
    print(f"  Checkpoint: {CKPT_PATH}")
    print(f"{'='*70}\n")

    optimizer = AdamW([
        {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 'lr': WARMUP_BB_LR},
        {'params': model.classifier.parameters(), 'lr': WARMUP_CLS_LR},
    ], weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-7)

    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device)
        if 'main_phase' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch      = ckpt['epoch'] + 1
            best_val_acc     = ckpt['best_val_acc']
            patience_counter = ckpt['patience_counter']
            in_main_phase    = ckpt['main_phase']
            print(f"✓ Resumed: epoch {start_epoch}, best {best_val_acc:.2f}%")
        else:
            print("⚠ Checkpoint format incompatible, starting from scratch")
            os.remove(CKPT_PATH)
    else:
        print("Training from scratch")

    EMOTION_NAMES = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    start_time = time.time()

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        epoch_start = time.time()

        use_mixup = (epoch >= WARMUP_EPOCHS)

        if epoch == WARMUP_EPOCHS and not in_main_phase:
            in_main_phase = True
            for g in optimizer.param_groups:
                if g['lr'] == WARMUP_BB_LR or g['initial_lr'] == WARMUP_BB_LR:
                    g['lr'] = MAIN_BB_LR
            print(f"\n[Main training] backbone LR → {MAIN_BB_LR}, Mixup enabled")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, use_mixup)
        val_loss, val_acc, per_class = validate(model, val_loader, criterion, device)
        scheduler.step()

        lrs = [f"{g['lr']:.1e}" for g in optimizer.param_groups]
        elapsed = time.time() - epoch_start
        tag = "[W]" if epoch < WARMUP_EPOCHS else "   "

        print(f"{tag} Epoch [{epoch+1:3d}/{TOTAL_EPOCHS}] {elapsed:.1f}s  LR: {'/'.join(lrs)}")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%", end='')

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_acc': val_acc}, BEST_MODEL_PATH)
            print(f"  ✓ New best: {best_val_acc:.2f}%")
            if val_acc >= 80.0:
                print("  🎯 Reached 80% target!")
            elif val_acc >= 75.0:
                print("  ✓ Surpassed 75%!")
        else:
            patience_counter += 1
            print(f"  (Best: {best_val_acc:.2f}%, {patience_counter}/{patience})")

        # 每类准确率 | Per-class accuracy
        class_accs = '  '.join(f"{EMOTION_NAMES[c]}:{per_class[c]:.1f}%" for c in range(6))
        print(f"  Per-class: {class_accs}")

        if train_acc - val_acc > 15:
            print(f"  ⚠ Overfitting (gap: {train_acc - val_acc:.1f}%)")

        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc':         best_val_acc,
            'patience_counter':     patience_counter,
            'main_phase':           in_main_phase,
        }, CKPT_PATH)

        print("-" * 70)

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered! No improvement for {patience} epochs")
            break

    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
