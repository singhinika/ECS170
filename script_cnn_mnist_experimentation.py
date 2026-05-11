'''
Ablation study for CNN on MNIST.
Tests 4 configurations to isolate the effect of BatchNorm and network depth.
Run from project root: python -m script.stage_3_script.script_cnn_mnist_experimentation
'''

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.base_class.method import method
from local_code.stage_3_code.plot_utils import plot_multiple_models_comparison

# ---- inline variant model definitions ----

class _CNN_MNIST_NoBN(method, nn.Module):
    '''Variant A: remove BatchNorm to isolate its contribution.'''
    max_epoch = 30; learning_rate = 1e-3; batch_size = 64
    train_losses = []; train_accuracies = []
    def __init__(self):
        method.__init__(self, 'NoBN', '')
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(1,  32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2); self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 7 * 7, 128); self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class _CNN_MNIST_Shallow(method, nn.Module):
    '''Variant B: only 1 conv block — tests whether depth is necessary.'''
    max_epoch = 30; learning_rate = 1e-3; batch_size = 64
    train_losses = []; train_accuracies = []
    def __init__(self):
        method.__init__(self, 'Shallow', '')
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2); self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 14 * 14, 128); self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = x.view(x.size(0), -1)
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class _CNN_MNIST_Deep(method, nn.Module):
    '''Variant C: 3 conv blocks — tests whether extra depth helps on MNIST.'''
    max_epoch = 30; learning_rate = 1e-3; batch_size = 64
    train_losses = []; train_accuracies = []
    def __init__(self):
        method.__init__(self, 'Deep', '')
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(1,  32, 3, padding=1); self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1); self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2); self.relu = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = nn.Linear(128 * 2 * 2, 256); self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


def train_and_eval(model, X_train, y_train, X_test, y_test, scheduler_type='cosine'):
    X_tr = torch.FloatTensor(np.array(X_train))
    y_tr = torch.LongTensor(np.array(y_train))
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=model.batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    if scheduler_type == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=model.max_epoch)
    loss_fn = nn.CrossEntropyLoss()

    model.train_losses = []; model.train_accuracies = []
    for epoch in range(model.max_epoch):
        total_loss = 0.0; correct = 0; total = 0
        nn.Module.train(model)
        for Xb, yb in loader:
            opt.zero_grad()
            out = model(Xb)
            loss = loss_fn(out, yb)
            loss.backward(); opt.step()
            total_loss += loss.item() * Xb.size(0)
            correct += (out.max(1)[1] == yb).sum().item()
            total += Xb.size(0)
        model.train_losses.append(total_loss / total)
        model.train_accuracies.append(correct / total)
        if scheduler_type == 'cosine':
            sched.step()
        if epoch % 10 == 0 or epoch == model.max_epoch - 1:
            print(f'  epoch {epoch}/{model.max_epoch}  loss={model.train_losses[-1]:.4f}  acc={model.train_accuracies[-1]:.4f}')

    model.eval()
    with torch.no_grad():
        X_te = torch.FloatTensor(np.array(X_test))
        preds = model(X_te).max(1)[1].numpy()
    trues = np.array(y_test)
    return {
        'accuracy':  accuracy_score(trues, preds),
        'precision': precision_score(trues, preds, average='macro', zero_division=0),
        'recall':    recall_score(trues, preds, average='macro', zero_division=0),
        'f1_macro':  f1_score(trues, preds, average='macro', zero_division=0),
    }, model.train_losses, model.train_accuracies


if __name__ == '__main__':
    np.random.seed(2); torch.manual_seed(2)

    d = Dataset_Loader('mnist', '')
    d.dataset_source_folder_path = './data/stage_3_data/'
    d.dataset_source_file_name   = 'MNIST'
    d.dataset_type               = 'MNIST'
    data = d.load()
    X_train, y_train = data['train']['X'], data['train']['y']
    X_test,  y_test  = data['test']['X'],  data['test']['y']

    # Baseline: 2 conv blocks + BatchNorm (from main script, simplified inline)
    from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
    baseline = Method_CNN_MNIST('CNN-MNIST-Baseline', '')
    baseline.data = data

    configs = [
        ('Baseline (2 blocks + BN)',  baseline,           'cosine'),
        ('No BatchNorm',              _CNN_MNIST_NoBN(),  'cosine'),
        ('Shallow (1 block)',          _CNN_MNIST_Shallow(), 'cosine'),
        ('Deep (3 blocks)',            _CNN_MNIST_Deep(),  'cosine'),
    ]

    results_table = {}
    comparison_data = {}

    for name, model, sched in configs:
        print(f'\n========== {name} ==========')
        if name == 'Baseline (2 blocks + BN)':
            # use the same train path as the main script
            model._train_model(X_train, y_train)
            model.eval()
            with torch.no_grad():
                preds = model._test_model(X_test).numpy()
            trues = np.array(y_test)
            metrics = {
                'accuracy':  accuracy_score(trues, preds),
                'precision': precision_score(trues, preds, average='macro', zero_division=0),
                'recall':    recall_score(trues, preds, average='macro', zero_division=0),
                'f1_macro':  f1_score(trues, preds, average='macro', zero_division=0),
            }
            losses = model.train_losses; accs = model.train_accuracies
        else:
            metrics, losses, accs = train_and_eval(model, X_train, y_train, X_test, y_test, sched)

        results_table[name] = metrics
        comparison_data[name] = {'loss': losses, 'accuracy': accs}
        print(f'  Accuracy={metrics["accuracy"]:.4f}  Precision={metrics["precision"]:.4f}  Recall={metrics["recall"]:.4f}  F1={metrics["f1_macro"]:.4f}')

    # ---- comparison table ----
    print('\n\n========== ABLATION COMPARISON TABLE (MNIST) ==========')
    print(f'{"Config":<30} {"Accuracy":>9} {"Precision":>10} {"Recall":>8} {"F1":>8}')
    print('-' * 70)
    for name, m in results_table.items():
        print(f'{name:<30} {m["accuracy"]:>9.4f} {m["precision"]:>10.4f} {m["recall"]:>8.4f} {m["f1_macro"]:>8.4f}')

    # ---- save comparison plot ----
    save_dir = './result/stage_3_result/CNN_MNIST'
    plot_multiple_models_comparison(comparison_data, save_dir=save_dir)
    print(f'\nComparison plot saved to {save_dir}/models_comparison.png')
    print('========== Done ==========')
