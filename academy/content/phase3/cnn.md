---
{
  "id": "p3-cnn",
  "title": "Convolutional Neural Networks",
  "phase": 3,
  "difficulty": "intermediate",
  "estimated_hours": 14,
  "objectives": [
    "Understand convolution operation and its role in feature extraction",
    "Design CNN architectures for image tasks",
    "Explain pooling, padding, and stride",
    "Apply transfer learning and data augmentation"
  ],
  "concepts": [
    "Convolution vs cross-correlation",
    "Filters/kernels and feature maps",
    "Padding (valid, same) and stride",
    "Pooling: max, average, global",
    "Receptive field",
    "Parameter sharing and translation equivariance",
    "Batch normalization in CNNs",
    "Classic architectures: LeNet, AlexNet, VGG, ResNet"
  ],
  "formulas": [
    {"name": "Conv output size", "latex": "O = \\left\\lfloor\\frac{I - K + 2P}{S}\\right\\rfloor + 1", "note": "I=input, K=kernel, P=padding, S=stride."},
    {"name": "Conv operation", "latex": "y[i,j] = \\sum_{m,n} x[i+m, j+n] \\cdot k[m,n]", "note": "Cross-correlation (no kernel flip in DL)."},
    {"name": "Params per conv layer", "latex": "K_h \\cdot K_w \\cdot C_{in} \\cdot C_{out} + C_{out}", "note": "Much fewer than dense layer."}
  ],
  "references": [
    "Goodfellow et al. ch. 9",
    "LeCun et al. 1998 LeNet paper",
    "He et al. 2016 'Deep Residual Learning' (ResNet)",
    "PyTorch nn.Conv2d documentation"
  ]
}
---

## Penjelasan

CNN adalah arsitektur khusus untuk data grid (gambar, audio spectrogram). Kunci inovasinya: **parameter sharing** (filter yang sama slide di seluruh input) dan **translation equivariance** (pola bergeser -> output bergeser).

**Convolution.** Filter kecil (mis. 3x3) slide di atas input, menghitung dot product di setiap posisi -> feature map. Setiap filter mendeteksi pola tertentu (edge, corner, texture). Layer awal mendeteksi edges; layer tengah shapes; layer akhir objects.

**Padding & stride.** Padding 'same' menambahkan border nol agar output size = input size; 'valid' tidak padding -> output lebih kecil. Stride >1 melompat filter -> downsampling (output lebih kecil, komputasi lebih cepat).

**Pooling.** Max pooling mengambil nilai maksimum di window -> translasi invariance lokal + downsampling. Average pooling merata-ratakan. Global average pooling (GAP) di akhir -> vector per channel -> fully connected.

**Parameter efficiency.** Conv layer 3x3, 64->128 channels: 3·3·64·128 + 128 = 73,856 params. Dense layer 784->256: 200k+ params. CNN jauh lebih efisien untuk gambar.

**Arsitektur modern.**
- *LeNet* (1998): pioneer digit recognition.
- *AlexNet* (2012): ImageNet breakthrough, ReLU + dropout + GPU.
- *VGG* (2014): deep 3x3 conv stack, simple uniform architecture.
- *ResNet* (2015): residual connections enable 100+ layers; skip connections solve vanishing gradient.
- *EfficientNet*: compound scaling (depth+width+resolution).

## Konsep Penting

1. **Conv != correlation** — DL menggunakan cross-correlation (tanpa flip kernel); mathematically equivalent setelah training.
2. **Receptive field** — area input yang mempengaruhi satu output neuron; harus cukup besar untuk capture konteks.
3. **Spatial dims matter** — conv preserves spatial structure; flatten terlalu dini menghilangkan informasi lokasi.
4. **Data augmentation crucial** — flip, crop, color jitter, mixup; CNN butuh augmentasi untuk generalisasi.
5. **Transfer learning** — pretrained backbone (ResNet, EfficientNet) + fine-tune head sangat efektif untuk data kecil.

## Formula

- Output size: O = floor((I + 2P − K)/S) + 1.
- Receptive field grows with depth and kernel size.
- Params: K_h·K_w·C_in·C_out + C_out.

## Contoh Kasus

Klasifikasi CIFAR-10 dengan ResNet-18: ~11M parameters, ~95% accuracy dengan augmentation. Bandingkan dengan MLP yang butuh flatten -> kehilangan spatial info -> akurasi lebih rendah dan parameter lebih banyak.

## Implementasi

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # -> 1x1
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)      # (B, 128, 1, 1)
        x = x.flatten(1)          # (B, 128)
        return self.classifier(x)

# transfer learning
import torchvision
backbone = torchvision.models.resnet18(weights='IMAGENET1K_V1')
backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
```

## Latihan

```json
[
  {
    "id": "ex1",
    "type": "concept",
    "question": "Convolutional layers achieve parameter efficiency through:",
    "options": [
      "Parameter sharing — same filter applied across all spatial positions",
      "Using fewer layers",
      "Smaller input images",
      "Removing biases"
    ],
    "answer": 0,
    "explanation": "One filter detects a pattern anywhere in image — massive parameter sharing vs dense layers.",
    "points": 1
  },
  {
    "id": "ex2",
    "type": "math",
    "question": "Input 32x32, kernel 3x3, padding 1, stride 1. Output spatial size?",
    "answer": 32,
    "tolerance": 0.1,
    "hints": ["O = (I + 2P - K)/S + 1 = (32+2-3)/1+1"],
    "explanation": "Same padding preserves spatial dimensions.",
    "points": 1
  },
  {
    "id": "ex3",
    "type": "concept",
    "question": "Max pooling primarily provides:",
    "options": [
      "Local translation invariance and downsampling",
      "More parameters",
      "Better gradient flow",
      "Color normalization"
    ],
    "answer": 0,
    "explanation": "Max over window -> small shifts don't change output; reduces spatial dims.",
    "points": 1
  },
  {
    "id": "ex4",
    "type": "coding",
    "question": "Implement `conv2d_output_size(I, K, P, S)` returning output spatial dimension.",
    "starter_code": "import math\n\ndef conv2d_output_size(I, K, P=0, S=1):\n    pass\n",
    "function": "conv2d_output_size",
    "tests": [
      {"call": "conv2d_output_size(32, 3, 1, 1)", "expected": 32},
      {"call": "conv2d_output_size(32, 3, 0, 2)", "expected": 15},
      {"call": "conv2d_output_size(224, 7, 3, 2)", "expected": 112}
    ],
    "hints": ["floor((I + 2P - K)/S) + 1"],
    "explanation": "Standard conv output size formula.",
    "points": 1
  },
  {
    "id": "ex5",
    "type": "open",
    "question": "You have 500 labeled images of rare bird species. Describe your approach: architecture choice, data strategy, training procedure, and how you'd evaluate success.",
    "rubric": [
      "Transfer learning from ImageNet-pretrained backbone",
          "Heavy data augmentation (flip, crop, color, mixup)",
      "Fine-tune with discriminative lr (lower for backbone, higher for head)",
      " stratified split, augmentation on train only, evaluate on held-out test"
        ],
    "points": 2
  }
]
```

## Referensi

- Goodfellow et al. ch. 9 (convolutional networks)
- LeCun et al. (1998) 'Gradient-based learning applied to document recognition'
- He et al. (2016) 'Deep Residual Learning for Image Recognition'
- PyTorch torchvision models documentation
- 'A guide to convolution arithmetic' — Dumoulin & Visin
