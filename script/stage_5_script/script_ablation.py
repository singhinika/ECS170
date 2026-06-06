'''
Ablation study: compare GCN variants across depth, hidden dimension, and dropout.

Run from project root:
    python -m script.stage_5_script.script_ablation
'''

import numpy as np
import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from local_code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from local_code.stage_5_code.Method_GCN import Method_GCN

# ---- ablation configurations -----------------------------------------------
# (label, hidden_dim, num_layers, dropout, lr, weight_decay, epochs)
ABLATIONS = [
    ('Baseline (2L-d64-dr0.5)',  64,  2, 0.5, 0.01, 5e-4, 200),
    # depth
    ('Depth: 1 layer',           64,  1, 0.5, 0.01, 5e-4, 200),
    ('Depth: 3 layers',          64,  3, 0.5, 0.01, 5e-4, 200),
    # hidden dimension
    ('Width: hidden=16',         16,  2, 0.5, 0.01, 5e-4, 200),
    ('Width: hidden=128',        128, 2, 0.5, 0.01, 5e-4, 200),
    # dropout
    ('Dropout: 0.0 (none)',      64,  2, 0.0, 0.01, 5e-4, 200),
    ('Dropout: 0.3',             64,  2, 0.3, 0.01, 5e-4, 200),
    ('Dropout: 0.7',             64,  2, 0.7, 0.01, 5e-4, 200),
]

DATASETS = {
    'cora':     './data/stage_5_data/cora',
    'citeseer': './data/stage_5_data/citeseer',
    'pubmed':   './data/stage_5_data/pubmed',
}

SEED       = 42
RESULT_DIR = './result/stage_5_result/ablation'


def run_config(data, label, hidden, layers, dropout, lr, wd, epochs):
    torch.manual_seed(SEED)
    m = Method_GCN(label, '')
    m.data         = data
    m.hidden_dim   = hidden
    m.num_layers   = layers
    m.dropout      = dropout
    m.learning_rate = lr
    m.weight_decay = wd
    m.max_epoch    = epochs
    result = m.run()
    true_y = result['true_y']
    pred_y = result['pred_y']
    return {
        'accuracy':  accuracy_score(true_y, pred_y),
        'precision': precision_score(true_y, pred_y, average='macro', zero_division=0),
        'recall':    recall_score(true_y, pred_y, average='macro', zero_division=0),
        'f1':        f1_score(true_y, pred_y, average='macro', zero_division=0),
    }


def print_table(dataset_name, rows):
    print(f'\n{"="*80}')
    print(f'ABLATION RESULTS — {dataset_name.upper()}')
    print(f'{"="*80}')
    print(f'  {"Configuration":<35} {"Acc":>7} {"Prec":>7} {"Rec":>7} {"F1":>7}')
    print(f'  {"-"*35} {"-"*7} {"-"*7} {"-"*7} {"-"*7}')
    for label, m in rows:
        print(f'  {label:<35} {m["accuracy"]:>7.4f} {m["precision"]:>7.4f} '
              f'{m["recall"]:>7.4f} {m["f1"]:>7.4f}')
    print(f'{"="*80}')


def plot_ablation(all_results, metric='accuracy'):
    labels = [r[0] for r in ABLATIONS]
    datasets = list(DATASETS.keys())
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, ds in enumerate(datasets):
        vals = [all_results[ds][lbl][metric] for lbl in labels]
        ax.bar(x + i * width, vals, width, label=ds.capitalize())

    ax.set_title(f'GCN Ablation Study — {metric.capitalize()} (macro)', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel(metric.capitalize())
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, f'ablation_{metric}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ---- main ------------------------------------------------------------------
if __name__ == '__main__':
    all_results = {ds: {} for ds in DATASETS}

    for dataset_name, data_path in DATASETS.items():
        print(f'\n{"#"*60}')
        print(f'# Dataset: {dataset_name.upper()}')
        print(f'{"#"*60}')

        # load data once per dataset
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        data_obj = Dataset_Loader(dName=dataset_name, dDescription='')
        data_obj.dataset_name = dataset_name
        data_obj.dataset_source_folder_path = data_path
        data = data_obj.load()

        rows = []
        for (label, hidden, layers, dropout, lr, wd, epochs) in ABLATIONS:
            print(f'\n--- {label} ---')
            metrics = run_config(data, label, hidden, layers, dropout, lr, wd, epochs)
            all_results[dataset_name][label] = metrics
            rows.append((label, metrics))

        print_table(dataset_name, rows)

    # ---- summary across all datasets ----
    print(f'\n{"="*80}')
    print('ABLATION SUMMARY — ACCURACY ACROSS ALL DATASETS')
    print(f'{"="*80}')
    print(f'  {"Configuration":<35} {"Cora":>8} {"Citeseer":>10} {"Pubmed":>8}')
    print(f'  {"-"*35} {"-"*8} {"-"*10} {"-"*8}')
    for (label, *_) in ABLATIONS:
        accs = [all_results[ds][label]['accuracy'] for ds in DATASETS]
        print(f'  {label:<35} {accs[0]:>8.4f} {accs[1]:>10.4f} {accs[2]:>8.4f}')
    print(f'{"="*80}')

    # ---- plots ----
    for metric in ('accuracy', 'f1'):
        plot_ablation(all_results, metric)

    print('\n************ Ablation Study Complete ************')
