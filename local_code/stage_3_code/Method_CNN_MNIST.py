'''
CNN model for MNIST handwritten digit classification (28x28 grayscale, 10 classes)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class Method_CNN_MNIST(method, nn.Module):
    data = None
    max_epoch = 30
    learning_rate = 1e-3
    batch_size = 64

    train_losses = []
    train_accuracies = []

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Block 1: 1x28x28 → 32x14x14
        # Conv preserves spatial dims with padding=1, MaxPool halves them
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        # Block 2: 32x14x14 → 64x7x7
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # After 2 pool layers: 64 * 7 * 7 = 3136
        self.fc1     = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)   # flatten
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

    def _train_model(self, X, y):
        # Convert list of numpy arrays to a single tensor: shape (N, 1, 28, 28)
        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.LongTensor(np.array(y))

        dataset = TensorDataset(X_tensor, y_tensor)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer     = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # Smoothly decays lr from learning_rate → ~0 over max_epoch steps,
        # avoiding oscillation near the optimum in the final epochs
        scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epoch)
        loss_function = nn.CrossEntropyLoss()

        self.train_losses     = []
        self.train_accuracies = []

        for epoch in range(self.max_epoch):
            epoch_loss = 0.0
            correct    = 0
            total      = 0

            # Put model in training mode (activates BatchNorm/Dropout)
            # Call nn.Module.train directly to avoid hitting our renamed method
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

            if epoch % 5 == 0 or epoch == self.max_epoch - 1:
                print(f'Epoch {epoch}/{self.max_epoch}  Loss: {avg_loss:.4f}  Accuracy: {accuracy:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}')

    def _test_model(self, X):
        # Switch to eval mode so BatchNorm uses running stats and Dropout is off
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
