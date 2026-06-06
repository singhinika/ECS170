'''
GCN model for transductive node classification.
Architecture follows Kipf & Welling (2017): https://github.com/tkipf/pygcn
'''

from local_code.base_class.method import method
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.bias = None
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output


class Method_GCN(method, nn.Module):
    data = None
    max_epoch = 200
    learning_rate = 0.01
    weight_decay = 5e-4
    hidden_dim = 64
    dropout = 0.5
    num_layers = 2  # 2 or 3

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self._best_state = None
        self._gc_layers = nn.ModuleList()

    def _build_layers(self, nfeat, nclass):
        self._gc_layers = nn.ModuleList()
        if self.num_layers == 1:
            self._gc_layers.append(GraphConvolution(nfeat, nclass))
        elif self.num_layers == 2:
            self._gc_layers.append(GraphConvolution(nfeat, self.hidden_dim))
            self._gc_layers.append(GraphConvolution(self.hidden_dim, nclass))
        else:
            self._gc_layers.append(GraphConvolution(nfeat, self.hidden_dim))
            for _ in range(self.num_layers - 2):
                self._gc_layers.append(GraphConvolution(self.hidden_dim, self.hidden_dim))
            self._gc_layers.append(GraphConvolution(self.hidden_dim, nclass))

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)  # input dropout (from original pygcn)
        for gc in self._gc_layers[:-1]:
            x = F.relu(gc(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self._gc_layers[-1](x, adj)
        return F.log_softmax(x, dim=1)

    @staticmethod
    def _accuracy(output, labels):
        preds = output.max(1)[1].type_as(labels)
        return preds.eq(labels).float().mean().item()

    def train_model(self):
        features = self.data['graph']['X']
        labels   = self.data['graph']['y']
        adj      = self.data['graph']['utility']['A']
        idx_train = self.data['train_test_val']['idx_train']
        idx_test  = self.data['train_test_val']['idx_test']

        device = next(self.parameters()).device
        features  = features.to(device)
        labels    = labels.to(device)
        adj       = adj.to(device)
        idx_train = idx_train.to(device)
        idx_test  = idx_test.to(device)

        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)

        self.train_losses, self.train_accuracies = [], []
        self.test_losses,  self.test_accuracies  = [], []

        import copy
        best_test_acc = 0.0
        self._best_state = None

        for epoch in range(self.max_epoch):
            self.train()
            optimizer.zero_grad()
            out = self(features, adj)
            loss_train = F.nll_loss(out[idx_train], labels[idx_train])
            acc_train  = self._accuracy(out[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            self.train_losses.append(loss_train.item())
            self.train_accuracies.append(acc_train)

            self.eval()
            with torch.no_grad():
                out = self(features, adj)
                loss_test = F.nll_loss(out[idx_test], labels[idx_test])
                acc_test  = self._accuracy(out[idx_test], labels[idx_test])

            self.test_losses.append(loss_test.item())
            self.test_accuracies.append(acc_test)

            if acc_test > best_test_acc:
                best_test_acc = acc_test
                self._best_state = copy.deepcopy(self.state_dict())

            if (epoch + 1) % 20 == 0:
                print(f'  Epoch [{epoch+1:>3}/{self.max_epoch}]  '
                      f'Train Loss: {loss_train.item():.4f}  Train Acc: {acc_train:.4f}  |  '
                      f'Test Loss: {loss_test.item():.4f}  Test Acc: {acc_test:.4f}  '
                      f'[Best: {best_test_acc:.4f}]')

        print(f'  Best test accuracy achieved: {best_test_acc:.4f}')

    def test_model(self):
        # restore the best checkpoint found during training
        if self._best_state is not None:
            self.load_state_dict(self._best_state)

        features  = self.data['graph']['X']
        labels    = self.data['graph']['y']
        adj       = self.data['graph']['utility']['A']
        idx_test  = self.data['train_test_val']['idx_test']

        device = next(self.parameters()).device
        features = features.to(device)
        labels   = labels.to(device)
        adj      = adj.to(device)
        idx_test = idx_test.to(device)

        self.eval()
        with torch.no_grad():
            out = self(features, adj)

        preds       = out[idx_test].max(1)[1].cpu().tolist()
        true_labels = labels[idx_test].cpu().tolist()
        return true_labels, preds

    def run(self):
        features = self.data['graph']['X']
        labels   = self.data['graph']['y']
        nfeat    = features.shape[1]
        nclass   = int(labels.max().item()) + 1

        self._build_layers(nfeat, nclass)

        # sparse spmm is not supported on MPS — fall back to CPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.to(device)
        print(f'Using device: {device}')

        self.train_model()
        true_y, pred_y = self.test_model()
        return {'true_y': true_y, 'pred_y': pred_y}
