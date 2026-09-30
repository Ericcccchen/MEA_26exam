import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from PIL import Image

app = Flask(__name__)

EMOTION_LABELS = ['disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_CN = ['Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_EMOJI = ['😒', '😨', '😄', '😐', '😢', '😲']

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        bb = models.resnet34(weights=None)

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


# loading
device = torch.device('cpu')
model = EmotionClassifier(6).to(device)

MODEL_PATH = 'best_model_v3_focal.pth'
if os.path.exists(MODEL_PATH):
    ckpt = torch.load(MODEL_PATH, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    print(f'Model loaded successfully: {MODEL_PATH}')
else:
    print(f'Warning: {MODEL_PATH} not found, using random weights')

model.eval()

# Image preprocessing (consistent with the validation set)
transform = transforms.Compose([
    transforms.Resize(192),
    transforms.CenterCrop(160),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# routes
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/tool')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file received'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Filename is empty'}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]

        results = []
        for i in range(6):
            results.append({
                'label': EMOTION_LABELS[i],
                'label_cn': EMOTION_CN[i],
                'emoji': EMOTION_EMOJI[i],
                'prob': round(probs[i].item() * 100, 1)
            })

        results.sort(key=lambda x: x['prob'], reverse=True)
        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, port=5000)
