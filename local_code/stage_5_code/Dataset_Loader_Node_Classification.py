'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from local_code.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp

class Dataset_Loader(dataset):
    data = None
    dataset_name = None

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(Dataset_Loader, self).__init__(dName, dDescription)

    def adj_normalize(self, mx):
        """normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = sorted(set(labels))  # sorted for deterministic class-index mapping
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        onehot_labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return onehot_labels

    def load(self):
        """Load citation network dataset"""
        print('Loading {} dataset...'.format(self.dataset_name))

        # load node data from file
        idx_features_labels = np.genfromtxt("{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        onehot_labels = self.encode_onehot(idx_features_labels[:, -1])

        # load link data from file and build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        reverse_idx_map = {i: j for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/link".format(self.dataset_source_folder_path), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        # convert to pytorch tensors
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(onehot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        if self.dataset_name == 'cora-small':
            idx_train = torch.LongTensor(range(5))
            idx_val   = torch.LongTensor(range(5, 10))
            idx_test  = torch.LongTensor(range(5, 10))
        else:
            # class-balanced random sampling as required by the assignment
            if self.dataset_name == 'cora':
                train_per_class, test_per_class = 20, 150
            elif self.dataset_name == 'citeseer':
                train_per_class, test_per_class = 20, 200
            elif self.dataset_name == 'pubmed':
                train_per_class, test_per_class = 20, 200
            else:
                train_per_class, test_per_class = 20, 150

            labels_np = np.where(onehot_labels)[1]
            n_classes = int(labels_np.max()) + 1
            class_indices = {c: np.where(labels_np == c)[0] for c in range(n_classes)}

            train_idx, test_idx, val_idx = [], [], []
            for c in range(n_classes):
                idxs = class_indices[c].copy()
                np.random.shuffle(idxs)
                n_needed = train_per_class + test_per_class
                replace = len(idxs) < n_needed
                if replace:
                    idxs = np.random.choice(idxs, size=n_needed, replace=True)
                train_idx.extend(idxs[:train_per_class])
                test_idx.extend(idxs[train_per_class:train_per_class + test_per_class])
                # use a small slice of remaining nodes for validation
                remaining = idxs[train_per_class + test_per_class:]
                val_size = min(50, len(remaining))
                if val_size > 0:
                    val_idx.extend(remaining[:val_size])

            idx_train = torch.LongTensor(train_idx)
            idx_test  = torch.LongTensor(test_idx)
            idx_val   = torch.LongTensor(val_idx) if val_idx else torch.LongTensor(test_idx[:50])
        # get the training nodes/testing nodes
        # train_x = features[idx_train]
        # val_x = features[idx_val]
        # test_x = features[idx_test]
        # print(train_x, val_x, test_x)

        train_test_val = {'idx_train': idx_train, 'idx_test': idx_test, 'idx_val': idx_val}
        graph = {'node': idx_map, 'edge': edges, 'X': features, 'y': labels, 'utility': {'A': adj, 'reverse_idx': reverse_idx_map}}
        return {'graph': graph, 'train_test_val': train_test_val}
