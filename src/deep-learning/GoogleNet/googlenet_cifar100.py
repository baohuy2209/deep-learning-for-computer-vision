import torch
import torch.nn as nn


class BaseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BaseConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)  # thêm BN, ổn định hơn LRN
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class InceptionModule(nn.Module):
    def __init__(
        self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_proj
    ):
        super(InceptionModule, self).__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True),
        )

        self.b2 = nn.Sequential(
            BaseConv2d(in_channels, n3x3red, kernel_size=1),
            BaseConv2d(n3x3red, n3x3, kernel_size=3, padding=1),
        )

        self.b3 = nn.Sequential(
            BaseConv2d(in_channels, n5x5red, kernel_size=1),
            BaseConv2d(n5x5red, n5x5, kernel_size=5, padding=2),
        )

        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            BaseConv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat(
            [self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1
        )


class GoogLeNet_CIFAR100(nn.Module):
    """
    Thay đổi so với bản gốc:
    1. Stem: Conv 7x7 stride2 → Conv 3x3 stride1 (giữ spatial 32x32)
    2. Bỏ LRN → BatchNorm (ổn định hơn)
    3. Bỏ MaxPool đầu (ảnh nhỏ không cần)
    4. Bỏ Auxiliary Classifier (CIFAR không cần, dataset nhỏ)
    5. Output: 1000 → 100 classes
    6. AvgPool 7x7 → AdaptiveAvgPool (tự động fit mọi input size)
    """

    def __init__(self, num_classes=100, dropout=0.4):
        super(GoogLeNet_CIFAR100, self).__init__()

        # -------------------------------------------------------
        # Stem — giữ nguyên 32x32, không stride không maxpool
        # -------------------------------------------------------
        self.stem = nn.Sequential(
            BaseConv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 32x32
            BaseConv2d(64, 64, kernel_size=1),  # 32x32
            BaseConv2d(64, 192, kernel_size=3, padding=1),  # 32x32
            nn.MaxPool2d(3, stride=2, padding=1),  # 16x16
        )

        # -------------------------------------------------------
        # Block 3
        # -------------------------------------------------------
        self.inception3a = InceptionModule(
            192, 64, 96, 128, 16, 32, 32
        )  # out: 256
        self.inception3b = InceptionModule(
            256, 128, 128, 192, 32, 96, 64
        )  # out: 480
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)  # 8x8

        # -------------------------------------------------------
        # Block 4
        # -------------------------------------------------------
        self.inception4a = InceptionModule(
            480, 192, 96, 208, 16, 48, 64
        )  # out: 512
        self.inception4b = InceptionModule(
            512, 160, 112, 224, 24, 64, 64
        )  # out: 512
        self.inception4c = InceptionModule(
            512, 128, 128, 256, 24, 64, 64
        )  # out: 512
        self.inception4d = InceptionModule(
            512, 112, 144, 288, 32, 64, 64
        )  # out: 528
        self.inception4e = InceptionModule(
            528, 256, 160, 320, 32, 128, 128
        )  # out: 832
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)  # 4x4

        # -------------------------------------------------------
        # Block 5
        # -------------------------------------------------------
        self.inception5a = InceptionModule(
            832, 256, 160, 320, 32, 128, 128
        )  # out: 832
        self.inception5b = InceptionModule(
            832, 384, 192, 384, 48, 128, 128
        )  # out: 1024

        # -------------------------------------------------------
        # Classifier
        # -------------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # bất kể size → (1,1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1024, num_classes)  # 1000 → 100

        # Weight initialization
        self._initialize_weights()

    def forward(self, x):
        # Stem
        x = self.stem(x)  # (B, 192, 16, 16)

        # Block 3
        x = self.inception3a(x)  # (B, 256, 16, 16)
        x = self.inception3b(x)  # (B, 480, 16, 16)
        x = self.maxpool3(x)  # (B, 480,  8,  8)

        # Block 4
        x = self.inception4a(x)  # (B, 512,  8,  8)
        x = self.inception4b(x)  # (B, 512,  8,  8)
        x = self.inception4c(x)  # (B, 512,  8,  8)
        x = self.inception4d(x)  # (B, 528,  8,  8)
        x = self.inception4e(x)  # (B, 832,  8,  8)
        x = self.maxpool4(x)  # (B, 832,  4,  4)

        # Block 5
        x = self.inception5a(x)  # (B, 832,  4,  4)
        x = self.inception5b(x)  # (B,1024,  4,  4)

        # Classifier
        x = self.avgpool(x)  # (B,1024,  1,  1)
        x = torch.flatten(x, 1)  # (B,1024)
        x = self.dropout(x)
        x = self.fc(x)  # (B, 100)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


# ============================================================
# TRAIN
# ============================================================
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Data
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)  # mean CIFAR-100
        ),  # std  CIFAR-100
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        ),
    ]
)

train_dataset = datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transform_train
)
test_dataset = datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transform_test
)

train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=2
)
test_loader = DataLoader(
    test_dataset, batch_size=128, shuffle=False, num_workers=2
)

# Model
model = GoogLeNet_CIFAR100(num_classes=100).to(device)
criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)  # label smoothing giảm overfit
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=3
)

# Kiểm tra output shape
x = torch.randn(2, 3, 32, 32).to(device)
print(f"Output shape: {model(x).shape}")  # → (2, 100)
