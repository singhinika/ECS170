'''
RNN model for binary sentiment classification (positive / negative)
Supports cell_type in {'RNN', 'LSTM', 'GRU'} for easy comparison in stage 4-5
'''

from local_code.base_class.method import method
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time


class Method_RNN_Classification(method, nn.Module):
    data = None
    max_epoch = 10
    learning_rate = 1e-3
    batch_size = 64
    vocab_size = None       # set from loaded vocab before run()
    embed_dim = 128
    hidden_dim = 256
    num_layers = 2
    dropout = 0.3
    cell_type = 'LSTM'      # 'RNN' | 'LSTM' | 'GRU'
    bidirectional = False   # if True, FC input doubles to hidden_dim * 2

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

    def _build_layers(self):
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        rnn_drop = self.dropout if self.num_layers > 1 else 0.0
        if self.cell_type == 'LSTM':
            self.rnn = nn.LSTM(self.embed_dim, self.hidden_dim, self.num_layers,
                               batch_first=True, dropout=rnn_drop,
                               bidirectional=self.bidirectional)
        elif self.cell_type == 'GRU':
            self.rnn = nn.GRU(self.embed_dim, self.hidden_dim, self.num_layers,
                              batch_first=True, dropout=rnn_drop,
                              bidirectional=self.bidirectional)
        else:
            self.rnn = nn.RNN(self.embed_dim, self.hidden_dim, self.num_layers,
                              batch_first=True, dropout=rnn_drop, nonlinearity='tanh',
                              bidirectional=self.bidirectional)
        self.drop = nn.Dropout(self.dropout)
        fc_in = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        self.fc = nn.Linear(fc_in, 2)

    def forward(self, x):
        emb = self.embedding(x)             # (B, T, E)
        out, _ = self.rnn(emb)              # (B, T, H)  — _ discards hidden/cell state
        out = out.mean(dim=1)               # mean pooling over all timesteps — more stable than last-step for long sequences
        out = self.drop(out)
        return self.fc(out)                 # (B, 2)

    @staticmethod
    def _to_tensor(data):
        '''Convert a Python list-of-lists to a LongTensor via numpy (10-20x faster).'''
        return torch.from_numpy(np.array(data, dtype=np.int64))

    def train_model(self):
        device = next(self.parameters()).device

        print('  Converting training data to tensors...')
        X = self._to_tensor(self.data['train']['X'])
        y = self._to_tensor(self.data['train']['y'])
        loader = DataLoader(TensorDataset(X, y), batch_size=self.batch_size, shuffle=True)

        print('  Converting test data to tensors...')
        test_X = self._to_tensor(self.data['test']['X'])
        test_y = self._to_tensor(self.data['test']['y'])
        test_loader = DataLoader(TensorDataset(test_X, test_y),
                                 batch_size=self.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.train_losses, self.train_accuracies = [], []
        self.test_losses,  self.test_accuracies  = [], []
        n_batches = len(loader)

        for epoch in range(self.max_epoch):
            # --- training pass ---
            self.train()
            total_loss, correct, total = 0.0, 0, 0
            t0 = time.time()
            for batch_idx, (X_b, y_b) in enumerate(loader):
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                logits = self(X_b)
                loss = criterion(logits, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()

                total_loss += loss.item() * len(y_b)
                correct    += (logits.argmax(1) == y_b).sum().item()
                total      += len(y_b)

                if (batch_idx + 1) % 100 == 0:
                    print(f'    Epoch {epoch+1}/{self.max_epoch}  '
                          f'batch {batch_idx+1}/{n_batches}  '
                          f'loss={total_loss/total:.4f}', flush=True)

            train_loss = total_loss / total
            train_acc  = correct    / total
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # --- test evaluation pass (no gradients) ---
            self.eval()
            t_loss, t_correct, t_total = 0.0, 0, 0
            with torch.no_grad():
                for X_b, y_b in test_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    logits = self(X_b)
                    loss   = criterion(logits, y_b)
                    t_loss    += loss.item() * len(y_b)
                    t_correct += (logits.argmax(1) == y_b).sum().item()
                    t_total   += len(y_b)

            test_loss = t_loss    / t_total
            test_acc  = t_correct / t_total
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)

            elapsed = time.time() - t0
            print(f'  Epoch [{epoch+1:>2}/{self.max_epoch}]  '
                  f'Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  '
                  f'Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}  '
                  f'({elapsed:.0f}s)', flush=True)

    def test_model(self):
        self.eval()
        device = next(self.parameters()).device
        X = torch.tensor(self.data['test']['X'], dtype=torch.long)
        y = torch.tensor(self.data['test']['y'], dtype=torch.long)
        loader = DataLoader(TensorDataset(X, y), batch_size=self.batch_size, shuffle=False)

        all_preds, all_true = [], []
        with torch.no_grad():
            for X_b, y_b in loader:
                logits = self(X_b.to(device))
                all_preds.extend(logits.argmax(1).cpu().tolist())
                all_true.extend(y_b.tolist())
        return all_true, all_preds

    def run(self):
        self._build_layers()
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        self.to(device)
        print(f'Using device: {device}')

        self.train_model()
        true_y, pred_y = self.test_model()
        return {'true_y': true_y, 'pred_y': pred_y}
