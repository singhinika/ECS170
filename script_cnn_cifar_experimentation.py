'''
Ablation study for CNN on CIFAR-10.
Tests 4 configurations to isolate augmentation, optimizer choice, and depth.
Run from project root: python -m script.stage_3_script.script_cnn_cifar_experimentation
'''

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import random

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.base_class.method import method
from local_code.stage_3_code.plot_utils import plot_multiple_models_comparison


class _AugDataset(Dataset):
    def __init__(self, X, y, augment=True):
        self.X = X; self.y = y; self.augment = augment
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            if random.random() > 0.5:
                x = torch.flip(x, dims=[-1])
            _, H, W = x.shape; pad = 4
            x = nn.functional.pad(x, (pad,pad,pad,pad), mode='reflect')
            t = random.randint(0, 2*pad); l = random.randint(0, 2*pad)
            x = x[:, t:t+H, l:l+W]
        return x, self.y[idx]


class _CIFAR_NoAug(method, nn.Module):
    '''Variant A: same architecture + SGD+cosine, but NO augmentation.
    Isolates the accuracy gain from random crop + horizontal flip.'''
    max_epoch = 80; learning_rate = 0.1; batch_size = 64
    train_losses = []; train_accuracies = []
    def __init__(self):
        method.__init__(self, 'NoAug', ''); nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1);   self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1);  self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128, 3, padding=1);  self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,256, 3, padding=1); self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2,2); self.relu = nn.ReLU()
        self.ap = nn.AdaptiveAvgPool2d((2,2))
        self.fc1 = nn.Linear(256*2*2, 256); self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.ap(x).view(x.size(0), -1)
        return self.fc2(self.drop(self.relu(self.fc1(x))))


class _CIFAR_Adam(method, nn.Module):
    '''Variant B: same architecture + augmentation, but Adam instead of SGD+momentum.
    Tests whether optimizer choice matters for CIFAR.'''
    max_epoch = 80; learning_rate = 1e-3; batch_size = 64
    train_losses = []; train_accuracies = []
    def __init__(self):
        method.__init__(self, 'Adam', ''); nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1);   self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1);  self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128, 3, padding=1);  self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,256, 3, padding=1); self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2,2); self.relu = nn.ReLU()
        self.ap = nn.AdaptiveAvgPool2d((2,2))
        self.fc1 = nn.Linear(256*2*2, 256); self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.ap(x).view(x.size(0), -1)
        return self.fc2(self.drop(self.relu(self.fc1(x))))


class _CIFAR_Shallow(method, nn.Module):
    '''Variant C: 3 conv blocks (drop the 4th) with augmentation + SGD+cosine.
    Tests whether the extra depth in the baseline is necessary.'''
    max_epoch = 80; learning_rate = 0.1; batch_size = 64
    train_losses = []; train_accuracies = []
    def __init__(self):
        method.__init__(self, 'Shallow', ''); nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1);  self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128, 3, padding=1); self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2,2); self.relu = nn.ReLU()
        self.ap = nn.AdaptiveAvgPool2d((2,2))
        self.fc1 = nn.Linear(128*2*2, 256); self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.ap(x).view(x.size(0), -1)
        return self.fc2(self.drop(self.relu(self.fc1(x))))


def train_and_eval(model, X_tr, y_tr, X_te, y_te, augment=True, use_adam=False):
    Xtr = torch.FloatTensor(np.array(X_tr))
    ytr = torch.LongTensor(np.array(y_tr))
    ds = _AugDataset(Xtr, ytr, augment=augment)
    loader = DataLoader(ds, batch_size=model.batch_size, shuffle=True)

    if use_adam:
        opt = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=model.learning_rate,
                              momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=model.max_epoch)
    loss_fn = nn.CrossEntropyLoss()

    model.train_losses = []; model.train_accuracies = []
    for epoch in range(model.max_epoch):
        tot_loss = 0.0; correct = 0; total = 0
        nn.Module.train(model)
        for Xb, yb in loader:
            opt.zero_grad(); out = model(Xb)
            loss = loss_fn(out, yb); loss.backward(); opt.step()
            tot_loss += loss.item() * Xb.size(0)
            correct += (out.max(1)[1] == yb).sum().item(); total += Xb.size(0)
        model.train_losses.append(tot_loss / total)
        model.train_accuracies.append(correct / total)
        sched.step()
        if epoch % 20 == 0 or epoch == model.max_epoch - 1:
            print(f'  epoch {epoch}/{model.max_epoch}  loss={model.train_losses[-1]:.4f}  acc={model.train_accuracies[-1]:.4f}')

    model.eval()
    with torch.no_grad():
        Xte = torch.FloatTensor(np.array(X_te))
        preds = model(Xte).max(1)[1].numpy()
    trues = np.array(y_te)
    return {
        'accuracy':  accuracy_score(trues, preds),
        'precision': precision_score(trues, preds, average='macro', zero_division=0),
        'recall':    recall_score(trues, preds, average='macro', zero_division=0),
        'f1_macro':  f1_score(trues, preds, average='macro', zero_division=0),
    }, model.train_losses, model.train_accuracies


if __name__ == '__main__':
    np.random.seed(2); torch.manual_seed(2)

    d = Dataset_Loader('cifar', '')
    d.dataset_source_folder_path = './data/stage_3_data/'
    d.dataset_source_file_name   = 'CIFAR'
    d.dataset_type               = 'CIFAR'
    data = d.load()
    X_tr, y_tr = data['train']['X'], data['train']['y']
    X_te, y_te = data['test']['X'],  data['test']['y']

    from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
    baseline = Method_CNN_CIFAR('CNN-CIFAR-Baseline', '')
    baseline.data = data

    # (display name, model, augment, use_adam)
    configs = [
        ('Baseline (4 blocks, SGD+aug)',  baseline,          True,  False),
        ('No Augmentation',               _CIFAR_NoAug(),    False, False),
        ('Adam Optimizer',                _CIFAR_Adam(),     True,  True),
        ('Shallow (3 blocks)',             _CIFAR_Shallow(),  True,  False),
    ]

    results_table = {}
    comparison_data = {}

    for name, model, aug, use_adam in configs:
        print(f'\n========== {name} ==========')
        if name == 'Baseline (4 blocks, SGD+aug)':
            model._train_model(X_tr, y_tr)
            model.eval()
            with torch.no_grad():
                preds = model._test_model(X_te).numpy()
            trues = np.array(y_te)
            metrics = {
                'accuracy':  accuracy_score(trues, preds),
                'precision': precision_score(trues, preds, average='macro', zero_division=0),
                'recall':    recall_score(trues, preds, average='macro', zero_division=0),
                'f1_macro':  f1_score(trues, preds, average='macro', zero_division=0),
            }
            losses = model.train_losses; accs = model.train_accuracies
        else:
            metrics, losses, accs = train_and_eval(model, X_tr, y_tr, X_te, y_te,
                                                   augment=aug, use_adam=use_adam)
        results_table[name] = metrics
        comparison_data[name] = {'loss': losses, 'accuracy': accs}
        print(f'  Accuracy={metrics["accuracy"]:.4f}  Precision={metrics["precision"]:.4f}  Recall={metrics["recall"]:.4f}  F1={metrics["f1_macro"]:.4f}')

    print('\n\n========== ABLATION COMPARISON TABLE (CIFAR-10) ==========')
    print(f'{"Config":<35} {"Accuracy":>9} {"Precision":>10} {"Recall":>8} {"F1":>8}')
    print('-' * 75)
    for name, m in results_table.items():
        print(f'{name:<35} {m["accuracy"]:>9.4f} {m["precision"]:>10.4f} {m["recall"]:>8.4f} {m["f1_macro"]:>8.4f}')

    save_dir = './result/stage_3_result/CNN_CIFAR'
    plot_multiple_models_comparison(comparison_data, save_dir=save_dir)
    print(f'\nComparison plot saved to {save_dir}/models_comparison.png')
    print('========== Done ==========')
