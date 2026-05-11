'''
CNN model for CIFAR-10 object classification (32x32 RGB, 10 classes)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random


class _AugmentedImageDataset(Dataset):
    '''
    Custom Dataset that applies random augmentation at fetch time.
    Augmentations: random horizontal flip + random crop (pad 4, crop 32x32).
    Random crop is the most impactful single augmentation for CIFAR (+3-5% accuracy).
    '''
    def __init__(self, X_tensor, y_tensor, augment=True):
        self.X       = X_tensor
        self.y       = y_tensor
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                x = torch.flip(x, dims=[-1])
            # Random crop: pad 4px on each side then crop back to 32x32
            # Teaches the network that objects can appear at different positions
            _, H, W = x.shape
            pad = 4
            x = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='reflect')
            top  = random.randint(0, 2 * pad)
            left = random.randint(0, 2 * pad)
            x = x[:, top:top + H, left:left + W]
        return x, y


class Method_CNN_CIFAR(method, nn.Module):
    data = None
    max_epoch     = 80
    learning_rate = 0.1   # SGD typically starts at a higher LR than Adam
    batch_size    = 64

    train_losses     = []
    train_accuracies = []

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Block 1: 3x32x32 → 32x16x16
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        # Block 2: 32x16x16 → 64x8x8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # Block 3: 64x8x8 → 128x4x4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        # Block 4: extra depth without another pool — keeps spatial size at 4x4
        # Adding this 4th block increases representational power without
        # reducing spatial resolution further (which is already very small)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # AdaptiveAvgPool collapses 256x4x4 → 256x2x2 = 1024 features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        self.fc1     = nn.Linear(256 * 2 * 2, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.relu(self.bn4(self.conv4(x)))   # no pool here
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

    def _train_model(self, X, y):
        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.LongTensor(np.array(y))

        aug_dataset = _AugmentedImageDataset(X_tensor, y_tensor, augment=True)
        loader      = DataLoader(aug_dataset, batch_size=self.batch_size, shuffle=True)

        # SGD + momentum is the standard optimizer for CIFAR.
        # It generalizes better than Adam when paired with cosine LR annealing.
        # weight_decay=5e-4 is the conventional CIFAR regularization strength.
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=5e-4
        )
        # Smoothly decays LR from 0.1 → ~0 over max_epoch steps
        scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epoch)
        loss_function = nn.CrossEntropyLoss()

        self.train_losses     = []
        self.train_accuracies = []

        for epoch in range(self.max_epoch):
            epoch_loss = 0.0
            correct    = 0
            total      = 0

            super().train()

            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                y_pred = self.forward(X_batch)
                loss   = loss_function(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * X_batch.size(0)
                preds      = y_pred.max(1)[1]
                correct    += (preds == y_batch).sum().item()
                total      += X_batch.size(0)

            avg_loss = epoch_loss / total
            accuracy = correct / total
            self.train_losses.append(avg_loss)
            self.train_accuracies.append(accuracy)

            scheduler.step()

            if epoch % 10 == 0 or epoch == self.max_epoch - 1:
                print(f'Epoch {epoch}/{self.max_epoch}  Loss: {avg_loss:.4f}  Accuracy: {accuracy:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}')

    def _test_model(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(np.array(X))
            y_pred   = self.forward(X_tensor)
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self._train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self._test_model(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
