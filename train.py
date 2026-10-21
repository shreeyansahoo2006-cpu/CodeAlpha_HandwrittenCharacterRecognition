"""
Handwritten Character Recognition - training script

Usage:
    python train.py --dataset mnist     # digits 0-9
    python train.py --dataset emnist    # letters A-Z

The first run downloads the dataset automatically (MNIST ~10 MB, EMNIST ~560 MB).
"""

import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


# ----------------------------------------------------------------------------
# 1. Settings you can change from the command line
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.001)
args = parser.parse_args()

# Use the GPU if there is one, otherwise the normal CPU. Both work fine here.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")


# ----------------------------------------------------------------------------
# 2. Load the data
# ----------------------------------------------------------------------------
# Images are turned into tensors and normalised (pixel values centred around 0).
# This helps the network learn faster.
steps = []

# EMNIST images are stored rotated/flipped, so we fix that first.
if args.dataset == "emnist":
    steps.append(transforms.Lambda(lambda img: TF.hflip(TF.rotate(img, -90))))

steps += [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
transform = transforms.Compose(steps)

if args.dataset == "mnist":
    train_data = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("data", train=False, download=True, transform=transform)
    class_names = [str(i) for i in range(10)]
else:
    # "letters" split = 26 classes, A to Z (upper and lower case merged)
    train_data = datasets.EMNIST("data", split="letters", train=True, download=True, transform=transform)
    test_data = datasets.EMNIST("data", split="letters", train=False, download=True, transform=transform)
    class_names = [chr(ord("A") + i) for i in range(26)]

# EMNIST letters are labelled 1..26, but PyTorch wants 0..25, so we shift them.
label_shift = 1 if args.dataset == "emnist" else 0

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=True)
print(f"Training images: {len(train_data)}   Test images: {len(test_data)}")


# ----------------------------------------------------------------------------
# 3. The CNN model
# ----------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Convolution layers look for patterns (edges, curves, loops...)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14 -> 14x14
        # Dropout randomly switches off some neurons while training so the
        # model doesn't just memorise the training images.
        self.dropout = nn.Dropout(0.3)
        # Fully connected layers make the final decision.
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 28x28 -> 14x14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 14x14 -> 7x7
        x = x.flatten(start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


model = SimpleCNN(num_classes=len(class_names)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


# ----------------------------------------------------------------------------
# 4. Training and testing functions
# ----------------------------------------------------------------------------
def train_one_epoch():
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), (labels - label_shift).to(device)

        optimizer.zero_grad()              # clear old gradients
        predictions = model(images)        # forward pass
        loss = loss_fn(predictions, labels)
        loss.backward()                    # work out how to improve
        optimizer.step()                   # update the weights

        total_loss += loss.item()
    return total_loss / len(train_loader)


def test():
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), (labels - label_shift).to(device)
            guesses = model(images).argmax(dim=1)
            correct += (guesses == labels).sum().item()
    return 100 * correct / len(test_data)


# ----------------------------------------------------------------------------
# 5. Run the training
# ----------------------------------------------------------------------------
losses, accuracies = [], []

for epoch in range(1, args.epochs + 1):
    loss = train_one_epoch()
    acc = test()
    losses.append(loss)
    accuracies.append(acc)
    print(f"Epoch {epoch}/{args.epochs}  loss: {loss:.4f}  test accuracy: {acc:.2f}%")

# Save the trained model so we can use it later without retraining
save_name = f"model_{args.dataset}.pth"
torch.save({"weights": model.state_dict(), "class_names": class_names}, save_name)
print(f"Model saved to {save_name}")


# ----------------------------------------------------------------------------
# 6. Make some pictures for your report
# ----------------------------------------------------------------------------
# Graph of loss and accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(range(1, len(losses) + 1), losses, marker="o")
ax1.set_title("Training loss")
ax1.set_xlabel("Epoch")
ax2.plot(range(1, len(accuracies) + 1), accuracies, marker="o", color="green")
ax2.set_title("Test accuracy (%)")
ax2.set_xlabel("Epoch")
plt.tight_layout()
plt.savefig(f"training_curves_{args.dataset}.png")

# A grid of test images with what the model predicted
model.eval()
images, labels = next(iter(test_loader))
with torch.no_grad():
    guesses = model(images[:12].to(device)).argmax(dim=1).cpu()

plt.figure(figsize=(10, 5))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(images[i][0], cmap="gray")
    real = class_names[labels[i] - label_shift]
    guess = class_names[guesses[i]]
    plt.title(f"True: {real} | Predicted: {guess}",
              color="green" if real == guess else "red", fontsize=9)
    plt.axis("off")
plt.tight_layout()
plt.savefig(f"sample_predictions_{args.dataset}.png")
print("Saved training curves and sample predictions as .png files")