'''
CNN model for ORL face recognition (112x92 grayscale, 40 classes)
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
    This multiplies effective training set size for small datasets like ORL.
    Augmentations applied: random horizontal flip, random small shift.
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
            # Random horizontal flip (valid for faces — people can be mirrored)
            if random.random() > 0.5:
                x = torch.flip(x, dims=[-1])
            # Random small shift: pad 8px each side, then crop back to original size
            # This simulates slight position variation in the input
            _, H, W = x.shape
            pad = 8
            x = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='reflect')
            top  = random.randint(0, 2 * pad)
            left = random.randint(0, 2 * pad)
            x = x[:, top:top + H, left:left + W]
        return x, y


class Method_CNN_ORL(method, nn.Module):
    data = None
    max_epoch     = 100
    learning_rate = 1e-3
    batch_size    = 16    # small batch because only 360 training samples

    train_losses      = []
    train_accuracies  = []

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Block 1: 1x112x92 → 32x56x46
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        # Block 2: 32x56x46 → 64x28x23
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # Block 3: 64x28x23 → 128x14x11
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # AdaptiveAvgPool compresses each feature map to exactly 4x4
        # regardless of the input spatial size, giving 128*4*4=2048 features.
        # This is much smaller than the previous fixed flatten (19712),
        # dramatically reducing the FC layer size and overfitting risk.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1      = nn.Linear(128 * 4 * 4, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2      = nn.Linear(256, 40)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(self.relu(self.fc1(x)))
        return self.fc2(x)

    def _train_model(self, X, y):
        X_tensor = torch.FloatTensor(np.array(X))
        # ORL labels are 1-40; shift to 0-39 for CrossEntropyLoss
        y_tensor = torch.LongTensor(np.array(y) - 1)

        aug_dataset = _AugmentedImageDataset(X_tensor, y_tensor, augment=True)
        loader      = DataLoader(aug_dataset, batch_size=self.batch_size, shuffle=True)

        # weight_decay adds L2 regularization — penalizes large weights
        # and helps prevent overfitting on this tiny dataset
        optimizer     = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        # Halve the LR when training loss stops improving for 10 consecutive epochs
        scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
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

            scheduler.step(avg_loss)

            if epoch % 20 == 0 or epoch == self.max_epoch - 1:
                print(f'Epoch {epoch}/{self.max_epoch}  Loss: {avg_loss:.4f}  Accuracy: {accuracy:.4f}  LR: {optimizer.param_groups[0]["lr"]:.6f}')

    def _test_model(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(np.array(X))
            y_pred   = self.forward(X_tensor)
        # Shift predictions back to 1-40 to match original labels
        return y_pred.max(1)[1] + 1

    def run(self):
        print('method running...')
        print('--start training...')
        self._train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self._test_model(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
