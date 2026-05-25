'''
RNN language model for joke text generation
Trained with teacher forcing; generates jokes autoregressively from a seed of 3 words
Supports cell_type in {'RNN', 'LSTM', 'GRU'} for stage 4-5 comparisons
'''

from local_code.base_class.method import method
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import re


class Method_RNN_Generation(method, nn.Module):
    data = None
    max_epoch = 50
    learning_rate = 1e-3
    batch_size = 128
    vocab_size = None       # set from loaded vocab before run()
    embed_dim = 128
    hidden_dim = 256
    num_layers = 2
    dropout = 0.3
    cell_type = 'RNN'       # 'RNN' | 'LSTM' | 'GRU'

    train_losses = []

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

    def _build_layers(self):
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        rnn_drop = self.dropout if self.num_layers > 1 else 0.0
        if self.cell_type == 'LSTM':
            self.rnn = nn.LSTM(self.embed_dim, self.hidden_dim, self.num_layers,
                               batch_first=True, dropout=rnn_drop)
        elif self.cell_type == 'GRU':
            self.rnn = nn.GRU(self.embed_dim, self.hidden_dim, self.num_layers,
                              batch_first=True, dropout=rnn_drop)
        else:
            self.rnn = nn.RNN(self.embed_dim, self.hidden_dim, self.num_layers,
                              batch_first=True, dropout=rnn_drop, nonlinearity='tanh')
        self.drop = nn.Dropout(self.dropout)
        self.fc   = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)              # (B, T, E)
        out, hidden = self.rnn(emb, hidden)  # (B, T, H)
        logits = self.fc(self.drop(out))     # (B, T, V)
        return logits, hidden

    @staticmethod
    def _to_tensor(data):
        import numpy as np
        return torch.from_numpy(np.array(data, dtype=np.int64))

    def train_model(self):
        self.train()
        device = next(self.parameters()).device
        print('  Converting joke sequences to tensors...')
        X = self._to_tensor(self.data['train']['X'])
        y = self._to_tensor(self.data['train']['y'])
        loader = DataLoader(TensorDataset(X, y), batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD tokens

        self.train_losses = []

        for epoch in range(self.max_epoch):
            total_loss, total = 0.0, 0
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                logits, _ = self(X_b)               # (B, T, V)
                B, T, V = logits.shape
                loss = criterion(logits.reshape(B * T, V), y_b.reshape(B * T))
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()

                # count non-padding targets for normalisation
                n_tokens = (y_b != 0).sum().item()
                total_loss += loss.item() * n_tokens
                total      += n_tokens

            avg_loss = total_loss / max(total, 1)
            self.train_losses.append(avg_loss)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'  Epoch [{epoch+1:>2}/{self.max_epoch}]  Loss: {avg_loss:.4f}')

    def generate(self, seed_words, max_len=60, temperature=0.7, top_k=10):
        '''Autoregressively generate a joke given a list of seed word strings.

        top_k: only sample from the k most probable next words (eliminates word salad).
               Set to 0 to disable and sample from the full vocabulary.
        temperature: values < 1 make the distribution sharper (more deterministic).
        '''
        self.eval()
        device  = next(self.parameters()).device
        vocab   = self.data['vocab']
        id2word = self.data['id2word']
        eos_id  = vocab.get('<EOS>', 2)
        unk_id  = vocab.get('<UNK>', 1)

        seed_ids      = [vocab.get(w, unk_id) for w in seed_words]
        generated_ids = list(seed_ids)

        with torch.no_grad():
            x = torch.tensor([seed_ids], dtype=torch.long).to(device)
            _, hidden = self(x)

            for _ in range(max_len - len(seed_ids)):
                last_token = torch.tensor([[generated_ids[-1]]], dtype=torch.long).to(device)
                logits, hidden = self(last_token, hidden)
                logits = logits[0, 0, :].float()    # (V,)  — fp32 for softmax stability

                logits = logits / max(temperature, 1e-8)

                # top-k filtering: mask everything below the k-th largest logit
                if top_k > 0:
                    k = min(top_k, logits.size(-1))
                    threshold = torch.topk(logits, k).values[-1]
                    logits = logits.masked_fill(logits < threshold, float('-inf'))

                probs   = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()

                if next_id == eos_id:
                    break
                generated_ids.append(next_id)

        words = [id2word.get(i, '<UNK>') for i in generated_ids]
        # re-attach punctuation (remove space before ? . ! ,)
        text = ' '.join(words)
        text = re.sub(r' ([?.!,])', r'\1', text)
        return text

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
        return None
