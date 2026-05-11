'''
Ablation study for CNN on ORL face dataset.
Tests 4 configurations to isolate augmentation, depth, and regularization effects.
Run from project root: python -m script.stage_3_script.script_cnn_orl_experimentation
'''

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
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
            _, H, W = x.shape
            pad = 8
            x = nn.functional.pad(x, (pad,pad,pad,pad), mode='reflect')
            t = random.randint(0, 2*pad); l = random.randint(0, 2*pad)
            x = x[:, t:t+H, l:l+W]
        return x, self.y[idx]


def _build_3block_orl():
    '''3 conv blocks with BN + AdaptiveAvgPool — same as baseline.'''
    model = nn.Sequential()
    model.conv1 = nn.Conv2d(1, 32, 3, padding=1);  model.bn1 = nn.BatchNorm2d(32)
    model.conv2 = nn.Conv2d(32, 64, 3, padding=1); model.bn2 = nn.BatchNorm2d(64)
    model.conv3 = nn.Conv2d(64,128, 3, padding=1); model.bn3 = nn.BatchNorm2d(128)
    return model


class _ORL_NoAug(method, nn.Module):
    '''Variant A: same architecture as baseline but no data augmentation.
    Shows how much augmentation contributes on a tiny dataset.'''
    max_epoch = 100; learning_rate = 1e-3; batch_size = 16
    train_losses = []; train_accuracies = []
    def __init__(self):
        method.__init__(self, 'NoAug', ''); nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1);  self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128, 3, padding=1); self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2,2); self.relu = nn.ReLU()
        self.ap = nn.AdaptiveAvgPool2d((4,4))
        self.fc1 = nn.Linear(128*4*4, 256); self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 40)
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.ap(x).view(x.size(0), -1)
        return self.fc2(self.drop(self.relu(self.fc1(x))))


class _ORL_Shallow(method, nn.Module):
    '''Variant B: 2 conv blocks instead of 3.
    Tests whether the extra depth is necessary for face recognition.'''
    max_epoch = 100; learning_rate = 1e-3; batch_size = 16
    train_losses = []; train_accuracies = []
    def __init__(self):
        method.__init__(self, 'Shallow', ''); nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1);  self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2,2); self.relu = nn.ReLU()
        self.ap = nn.AdaptiveAvgPool2d((4,4))
        self.fc1 = nn.Linear(64*4*4, 256); self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 40)
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.ap(x).view(x.size(0), -1)
        return self.fc2(self.drop(self.relu(self.fc1(x))))


class _ORL_NoWeightDecay(method, nn.Module):
    '''Variant C: same architecture as baseline but no weight_decay and no LR scheduler.
    Tests the combined effect of weight decay + adaptive LR.'''
    max_epoch = 100; learning_rate = 1e-3; batch_size = 16
    train_losses = []; train_accuracies = []
    def __init__(self):
        method.__init__(self, 'NoWD', ''); nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1);  self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128, 3, padding=1); self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2,2); self.relu = nn.ReLU()
        self.ap = nn.AdaptiveAvgPool2d((4,4))
        self.fc1 = nn.Linear(128*4*4, 256); self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 40)
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.ap(x).view(x.size(0), -1)
        return self.fc2(self.drop(self.relu(self.fc1(x))))


def train_and_eval(model, X_tr, y_tr, X_te, y_te, augment=True, weight_decay=1e-4, use_scheduler=True):
    Xtr = torch.FloatTensor(np.array(X_tr))
    ytr = torch.LongTensor(np.array(y_tr) - 1)    # shift 1-40 → 0-39

    ds = _AugDataset(Xtr, ytr, augment=augment)
    loader = DataLoader(ds, batch_size=model.batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=model.learning_rate, weight_decay=weight_decay)
    if use_scheduler:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)
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
        avg_loss = tot_loss / total
        model.train_losses.append(avg_loss)
        model.train_accuracies.append(correct / total)
        if use_scheduler: sched.step(avg_loss)
        if epoch % 25 == 0 or epoch == model.max_epoch - 1:
            print(f'  epoch {epoch}/{model.max_epoch}  loss={avg_loss:.4f}  acc={correct/total:.4f}')

    model.eval()
    with torch.no_grad():
        Xte = torch.FloatTensor(np.array(X_te))
        preds = (model(Xte).max(1)[1] + 1).numpy()    # shift back 0-39 → 1-40
    trues = np.array(y_te)
    return {
        'accuracy':  accuracy_score(trues, preds),
        'precision': precision_score(trues, preds, average='macro', zero_division=0),
        'recall':    recall_score(trues, preds, average='macro', zero_division=0),
        'f1_macro':  f1_score(trues, preds, average='macro', zero_division=0),
    }, model.train_losses, model.train_accuracies


if __name__ == '__main__':
    np.random.seed(2); torch.manual_seed(2)

    d = Dataset_Loader('orl', '')
    d.dataset_source_folder_path = './data/stage_3_data/'
    d.dataset_source_file_name   = 'ORL'
    d.dataset_type               = 'ORL'
    data = d.load()
    X_tr, y_tr = data['train']['X'], data['train']['y']
    X_te, y_te = data['test']['X'],  data['test']['y']

    from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
    baseline = Method_CNN_ORL('CNN-ORL-Baseline', '')
    baseline.data = data

    configs = [
        # (display name, model, augment, weight_decay, use_scheduler)
        ('Baseline (aug + WD + sched)',  baseline,                True,  1e-4, True),
        ('No Augmentation',              _ORL_NoAug(),            False, 1e-4, True),
        ('Shallow (2 blocks)',            _ORL_Shallow(),          True,  1e-4, True),
        ('No Weight Decay / Scheduler',  _ORL_NoWeightDecay(),    True,  0.0,  False),
    ]

    results_table = {}
    comparison_data = {}

    for name, model, aug, wd, sched in configs:
        print(f'\n========== {name} ==========')
        if name == 'Baseline (aug + WD + sched)':
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
                                                   augment=aug, weight_decay=wd, use_scheduler=sched)
        results_table[name] = metrics
        comparison_data[name] = {'loss': losses, 'accuracy': accs}
        print(f'  Accuracy={metrics["accuracy"]:.4f}  Precision={metrics["precision"]:.4f}  Recall={metrics["recall"]:.4f}  F1={metrics["f1_macro"]:.4f}')

    print('\n\n========== ABLATION COMPARISON TABLE (ORL) ==========')
    print(f'{"Config":<35} {"Accuracy":>9} {"Precision":>10} {"Recall":>8} {"F1":>8}')
    print('-' * 75)
    for name, m in results_table.items():
        print(f'{name:<35} {m["accuracy"]:>9.4f} {m["precision"]:>10.4f} {m["recall"]:>8.4f} {m["f1_macro"]:>8.4f}')

    save_dir = './result/stage_3_result/CNN_ORL'
    plot_multiple_models_comparison(comparison_data, save_dir=save_dir)
    print(f'\nComparison plot saved to {save_dir}/models_comparison.png')
    print('========== Done ==========')
