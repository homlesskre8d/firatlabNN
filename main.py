import os
from PIL import Image
import scipy.io as matreader
import time
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModule, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(96, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            FireModule(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(256, 32, 128, 128),
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(512, 64, 256, 256),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, self.num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)

class CarModelsDataset(Dataset):
    def __init__(self, annotations_file, images_dir, transform=None):
        data = matreader.loadmat(annotations_file)
        self.annotations = data['annotations']
        self.images_dir = images_dir
        self.transform = transform

        # Фильтрация: оставляем только существующие файлы
        self.valid_indices = []
        for idx in range(len(self.annotations[0])):
            fname = self.annotations[0, idx]['fname'][0]
            image_path = os.path.join(self.images_dir, fname)
            if os.path.exists(image_path):
                self.valid_indices.append(idx)
            else:
                print(f"Файл {image_path} не найден. Пропуск.")

        # Проверка, что датасет не пуст
        if len(self.valid_indices) == 0:
            raise ValueError(f"Датасет пуст. Ни один файл не найден в директории {images_dir}.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]  # Используем отфильтрованные индексы
        annotation = self.annotations[0, actual_idx]
        fname = annotation['fname'][0]
        fname = str(fname)
        image_path = os.path.join(self.images_dir, fname)

        # Открываем изображение
        image = Image.open(image_path).convert('RGB')
        class_label = int(annotation['class'][0][0]) - 1

        if self.transform:
            image = self.transform(image)
        return image, class_label
def train_test(model, optimizer, train_loader, test_loader, epochs):
        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_dataset)
            epoch_accuracy = 100 * correct / total
            print(f"Эпоха {epoch + 1}, Время обучения: {round(time.time() - start_time, 2)}c., Потери: {epoch_loss:.4f}, Точность: {epoch_accuracy:.2f}%")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        print(f"Тестовая точность: {test_accuracy:.2f}%")
if __name__ == '__main__':
    cars_meta = matreader.loadmat('cars_meta.mat')
    class_names = [name[0] for name in cars_meta['class_names'][0]]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    try:
        train_dataset = CarModelsDataset('cars_train_annos.mat',
                                        'cars_train',
                                        transform=transform)

        test_dataset = CarModelsDataset('cars_test_annos_withlabels_eval.mat',
                                       'cars_test',
                                       transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SqueezeNet(num_classes=len(class_names)).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        print("Обучение SqueezeNet с Adam")
        train_test(model, optimizer, train_loader, test_loader, epochs=10)

    except ValueError as e:
        print(f"Ошибка: {e}")