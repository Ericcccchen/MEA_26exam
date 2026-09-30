# EmotionAI — Facial Emotion Recognition System

A deep learning web application that recognizes 6 facial emotions from uploaded images in real time. Built with a fine-tuned ResNet-34 backbone, Focal Loss, and a Flask web interface with bilingual (Chinese/English) support.

---

## Demo

Upload any face image via the web interface and the model returns a ranked confidence distribution across 6 emotion classes:

| Emotion | Label |
|---------|-------|
| 😠 Anger | `anger` |
| 😒 Disgust | `disgust` |
| 😨 Fear | `fear` |
| 😄 Happy | `happy` |
| 😐 Neutral | `neutral` |
| 😢 Sad | `sad` |

---

## Architecture

```
Input Image (160×160)
        │
   ResNet-34 Backbone (pretrained on ImageNet, layers 1–4)
        │
   Global Average Pooling  →  512-dim feature vector
        │
   Classifier Head:
     Dropout(0.4) → Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→6)
        │
   Softmax → 6-class probability distribution
```

**Total parameters:** ~21.4 million

---

## Training Details

| Setting | Value |
|---------|-------|
| Backbone | ResNet-34 (ImageNet pretrained) |
| Input size | 160 × 160 |
| Loss function | Focal Loss (γ=2.0) + class weights |
| Optimizer | AdamW (weight decay=0.01) |
| LR schedule | CosineAnnealingLR |
| Warmup | 5 epochs, backbone LR=1e-5, head LR=5e-4 |
| Main training | 75 epochs, backbone LR=5e-5, Mixup α=0.4 |
| Gradient clipping | max_norm=1.0 |
| Early stopping | patience=20 |
| Best val accuracy | **77.67%** (stopped at epoch 55) |

### Data Augmentation (training)
- Resize → 192, RandomCrop → 160
- RandomHorizontalFlip (p=0.5)
- ColorJitter (brightness=0.3, contrast=0.3, saturation=0.2)
- RandomRotation (±10°)
- ImageNet normalization

### Dataset
- **Training set:** 13,049 images
- **Validation set:** 11,488 images
- **Class distribution (train):** anger 1229 / disgust 1512 / fear 2340 / happy 2758 / neutral 3091 / sad 2119
- Class imbalance handled via inverse-frequency weights in Focal Loss

---

## Project Structure

```
.
├── train5_3.py           # Training script (ResNet-34 + FocalLoss + Mixup)
├── dataset.py            # Custom Dataset class (reads image path lists)
├── app.py                # Flask web application (inference server)
├── templates/
│   ├── home.html         # Landing page (bilingual, dark theme)
│   └── index.html        # Prediction tool page
├── data/
│   ├── train/
│   │   └── train.txt     # One line per sample: <image_path> <label>
│   └── test/
│       └── val.txt
├── best_model_v3_focal.pth    # Best checkpoint (saved by val accuracy)
└── checkpoint_v3_focal.pth    # Latest training checkpoint (resumable)
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision flask pillow numpy
```

### 2. Run the web app

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

- `/` — Homepage with project overview
- `/tool` — Upload a face image and get emotion predictions

### 3. Train from scratch

Prepare your dataset as text files where each line is:
```
data/train/<emotion>/image.jpg <label_1_to_6>
```

Then run:
```bash
python train5_3.py
```

Training auto-resumes from `checkpoint_v3_focal.pth` if it exists.

---

## Hardware Support

The training script auto-selects the best available device:

| Device | Support |
|--------|---------|
| CUDA GPU | ✅ |
| Apple MPS (M1/M2/M3) | ✅ |
| CPU | ✅ (fallback) |

---

## Training Curve (best run)

```
Epoch  1 → Val 53.47%
Epoch  5 → Val 69.82%  (end of warmup)
Epoch  7 → Val 72.98%  (Mixup enabled)
Epoch 11 → Val 75.15%  ✓ Surpassed 75%
Epoch 28 → Val 77.61%
Epoch 35 → Val 77.67%  ← Best
Epoch 55 → Early stop (patience=20 reached)
```

---

## Tech Stack

- **PyTorch** — model definition, training loop
- **torchvision** — ResNet-34 pretrained weights, transforms
- **Flask** — REST API + HTML template serving
- **Pillow** — image loading and preprocessing

---

## License

MIT
