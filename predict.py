"""
Recognise your OWN handwriting.

Usage:
    python predict.py my_digit.png --dataset mnist
    python predict.py my_letter.png --dataset emnist

Tips for a good photo/scan:
  - Write ONE character, big and thick, with a dark pen on white paper
  - Crop the image so the character fills most of it
  - Any pen colour works (blue is fine). The script removes shadows, crops to the
    character, thickens thin lines and centres it, like the training images.
  - Add --save-debug to save "what_model_sees.png" so you can check the cleaned-up image.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("image_path")
parser.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
parser.add_argument("--no-invert", action="store_true",
                    help="use if your image is already white writing on black")
parser.add_argument("--save-debug", action="store_true",
                    help="save the cleaned 28x28 image the model actually sees")
args = parser.parse_args()


# The model definition must match the one in train.py exactly
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# Load the saved model
checkpoint = torch.load(f"model_{args.dataset}.pth", map_location="cpu")
class_names = checkpoint["class_names"]
model = SimpleCNN(len(class_names))
model.load_state_dict(checkpoint["weights"])
model.eval()

# ---- PREPROCESS START ----
def prepare_image(path, invert=True):
    """Turn a photo of a handwritten character into a clean 28x28 MNIST-style image."""
    img = Image.open(path).convert("L")

    # work at a manageable size
    img.thumbnail((800, 800))

    if invert:
        # Remove uneven lighting/shadows: estimate the paper brightness, then
        # see how much darker each pixel is than the paper around it.
        paper = img.filter(ImageFilter.MaxFilter(31)).filter(ImageFilter.GaussianBlur(15))
        diff = np.asarray(paper, dtype=np.float32) - np.asarray(img, dtype=np.float32)
        diff = np.clip(diff, 0, None)
    else:
        diff = np.asarray(img, dtype=np.float32)

    # scale so the darkest ink = 255, and drop faint noise
    diff = diff / max(diff.max(), 1) * 255
    diff[diff < 90] = 0

    ink = Image.fromarray(diff.astype("uint8"))

    # crop to the character
    box = ink.getbbox()
    if box is None:
        raise SystemExit("Could not find any writing in the image.")
    ink = ink.crop(box)

    # shrink first (fast), then thicken thin pen lines so they survive the
    # final shrink to 28x28 (MNIST strokes are chunky)
    scale = 100 / max(ink.size)
    ink = ink.resize((max(1, round(ink.width * scale)), max(1, round(ink.height * scale))), Image.LANCZOS)
    ink = ink.filter(ImageFilter.MaxFilter(9))
    ink = ink.filter(ImageFilter.GaussianBlur(1.2))

    # make it square (keep the shape, don't stretch), then fit into 20x20 like MNIST
    side = max(ink.size)
    square = Image.new("L", (side, side), 0)
    square.paste(ink, ((side - ink.width) // 2, (side - ink.height) // 2))
    square = square.resize((20, 20), Image.LANCZOS)

    # centre it in a 28x28 canvas
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(square, (4, 4))
    return canvas
# ---- PREPROCESS END ----


# Prepare the image the same way the training images were prepared
img = prepare_image(args.image_path, invert=not args.no_invert)
if args.save_debug:
    img.resize((280, 280), Image.NEAREST).save("what_model_sees.png")
    print("Saved what_model_sees.png")

tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])(img).unsqueeze(0)                                # add a "batch" dimension

with torch.no_grad():
    probabilities = F.softmax(model(tensor), dim=1)[0]

top3 = probabilities.topk(3)
print("\nI think this is:")
for prob, idx in zip(top3.values, top3.indices):
    print(f"  {class_names[idx]}   ({prob.item() * 100:.1f}% sure)")