import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Load Dataset
# ==========================================

data = np.load("dataset_augmented.npz")

X = data["X"]   # (samples, 50, features)
y = data["y"]

print("Dataset Shape:", X.shape)

# Fix labels
y = y - 1

# ==========================================
# Train-Test Split (80:20)
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# Normalize Data
# ==========================================

mean = X_train.mean()
std = X_train.std() + 1e-8

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


# ==========================================
# PyTorch Dataset Class
# ==========================================

class ActivityDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        # Convert to (channels, length)
        x = self.X[idx].permute(1, 0)

        return x, self.y[idx]


train_dataset = ActivityDataset(X_train, y_train)
test_dataset = ActivityDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# ==========================================
# Residual Block
# ==========================================

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1
                )
            )

        self.relu = nn.ReLU()

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# ==========================================
# RCNN Model (4 Residual Blocks)
# ==========================================

class RCNN(nn.Module):

    def __init__(self, input_channels, num_classes):

        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv1d(
                input_channels,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # 4 Residual Blocks
        self.layer1 = ResidualBlock(32, 32)
        self.layer2 = ResidualBlock(32, 64)
        self.layer3 = ResidualBlock(64, 128)
        self.layer4 = ResidualBlock(128, 128)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.initial(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)

        x = x.squeeze(-1)

        x = self.fc(x)

        return x


# ==========================================
# Initialize Model
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

input_channels = X_train.shape[2]
num_classes = len(np.unique(y))

model = RCNN(
    input_channels,
    num_classes
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# Training Loop
# ==========================================

num_epochs = 3

train_losses = []

for epoch in range(num_epochs):

    model.train()

    running_loss = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    train_losses.append(epoch_loss)

    print(
        f"Epoch {epoch+1}/{num_epochs}, "
        f"Loss: {epoch_loss:.4f}"
    )


# ==========================================
# Testing
# ==========================================

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(device)

        outputs = model(inputs)

        _, preds = torch.max(
            outputs,
            1
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# ==========================================
# Accuracy
# ==========================================

accuracy = np.mean(
    np.array(all_preds)
    == np.array(all_labels)
)

print("\nTest Accuracy:", accuracy)


# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_preds
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("True")

plt.title("Confusion Matrix")

plt.show()