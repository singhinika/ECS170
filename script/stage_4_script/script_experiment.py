'''
Stage 4-5 / 4-6  Experiment Script
====================================
Runs all three generation models (RNN, LSTM, GRU) back-to-back on the same seeds,
collects metrics, and writes a unified comparison report.

What this script does (steps 4-2 → 4-4 re-done for each cell type):
  4-2  Load & preprocess jokes data once, shared across all models.
  4-3  Train each model (RNN / LSTM / GRU) with its tuned hyperparameters.
  4-4  Generate jokes from 7 shared seeds; save individual output files.
  4-5  Produce a side-by-side comparison table and overlay loss plot.

Hyperparameter choices and rationale
--------------------------------------
Base RNN  : embed=128, hidden=256, lr=1e-3, epochs=50, temp=0.8, top_k=10
  Baseline — vanilla tanh-RNN, suffers from vanishing gradients on jokes > ~10 words.

LSTM      : embed=256, hidden=512, lr=5e-4, epochs=80, temp=0.7, top_k=15
  Adds forget/input/output gates and a separate cell state (c_t).  The cell state
  acts as a "long-term memory lane" that can carry the joke setup through many
  timesteps, making punchlines more likely to reference earlier words.
  Larger embed/hidden to exploit the extra capacity; lower LR and more epochs to
  converge stably; lower temperature so the sharper LSTM distribution doesn't
  produce redundant top-1 tokens.

GRU       : embed=256, hidden=512, lr=5e-4, epochs=75, temp=0.75, top_k=12
  Combines the reset and update gates of LSTM into two (no separate cell state).
  Fewer parameters → faster convergence on the short jokes (mean ~15 tokens).
  Slightly higher temperature than LSTM (0.75 vs 0.7) to preserve the natural
  variety that GRU's smaller parameter set tends to produce.

Run from project root:
    python -m script.stage_4_script.script_experiment
'''

import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from local_code.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation
from local_code.stage_4_code.Method_RNN_Generation     import Method_RNN_Generation

RESULT_DIR = './result/stage_4_result/Experiment_Generation'

# Shared seeds — same for all three models so comparisons are fair
FIXED_SEEDS = [
    ['what', 'did', 'the'],
    ["don't", 'you', 'hate'],
    ['two', 'artists', 'had'],
    ['why', 'did', 'the'],
    ['what', 'do', 'you'],
    ['why', 'does', 'a'],
    ['how', 'do', 'you'],
]

# Per-model hyperparameter configs
MODEL_CONFIGS = {
    'RNN': {
        'cell_type':     'RNN',
        'embed_dim':     128,
        'hidden_dim':    256,
        'num_layers':    2,
        'dropout':       0.3,
        'max_epoch':     50,
        'learning_rate': 1e-3,
        'batch_size':    128,
        'temperature':   0.8,
        'top_k':         10,
    },
    'LSTM': {
        'cell_type':     'LSTM',
        'embed_dim':     256,
        'hidden_dim':    512,
        'num_layers':    2,
        'dropout':       0.4,
        'max_epoch':     80,
        'learning_rate': 5e-4,
        'batch_size':    128,
        'temperature':   0.7,
        'top_k':         15,
    },
    'GRU': {
        'cell_type':     'GRU',
        'embed_dim':     256,
        'hidden_dim':    512,
        'num_layers':    2,
        'dropout':       0.35,
        'max_epoch':     75,
        'learning_rate': 5e-4,
        'batch_size':    128,
        'temperature':   0.75,
        'top_k':         12,
    },
}


def get_seeds(raw_jokes, vocab, n=7):
    '''Return n unique 3-word seeds verified to be in the vocab.'''
    seeds, seen = [], set()
    for s in FIXED_SEEDS:
        key = tuple(s)
        if key not in seen and all(w in vocab for w in s):
            seeds.append(s)
            seen.add(key)
        if len(seeds) >= n:
            break
    for joke in raw_jokes:
        if len(seeds) >= n:
            break
        if len(joke) >= 4:
            key = tuple(joke[:3])
            if key not in seen and all(w in vocab for w in joke[:3]):
                seeds.append(list(joke[:3]))
                seen.add(key)
    return seeds


def find_original(raw_jokes, seed):
    for joke in raw_jokes:
        if joke[:len(seed)] == seed:
            return ' '.join(w for w in joke if w != '<EOS>')
    return None


def build_and_train(model_name, cfg, loaded_data):
    '''Instantiate, configure, and train a generation model. Returns the trained object.'''
    m = Method_RNN_Generation(model_name, '')
    m.data          = loaded_data
    m.vocab_size    = len(loaded_data['vocab'])
    m.cell_type     = cfg['cell_type']
    m.embed_dim     = cfg['embed_dim']
    m.hidden_dim    = cfg['hidden_dim']
    m.num_layers    = cfg['num_layers']
    m.dropout       = cfg['dropout']
    m.max_epoch     = cfg['max_epoch']
    m.learning_rate = cfg['learning_rate']
    m.batch_size    = cfg['batch_size']
    m.run()
    return m


def plot_loss_comparison(all_losses, save_dir):
    '''Overlay training loss curves for all three models on one figure.'''
    plt.figure(figsize=(11, 5))
    styles = {'RNN': ('b-', 'RNN (baseline)'), 'LSTM': ('g-', 'LSTM'), 'GRU': ('r-', 'GRU')}
    for name, losses in all_losses.items():
        style, label = styles[name]
        plt.plot(range(1, len(losses) + 1), losses, style, linewidth=2, label=label)
    plt.title('Generation Model Comparison — Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss (token-level)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, 'loss_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Loss comparison plot saved to: {path}')


def print_comparison_table(results_table):
    '''Print a side-by-side table of generated jokes for all models.'''
    divider = '=' * 80
    print('\n' + divider)
    print('SIDE-BY-SIDE JOKE COMPARISON')
    print(divider)
    for entry in results_table:
        seed_str  = ' '.join(entry['seed'])
        original  = entry.get('original', '')
        print(f'\nSeed: "{seed_str}"')
        if original:
            print(f'  Original : {original}')
        for model_name in MODEL_CONFIGS:
            gen = entry.get(model_name, '')
            print(f'  {model_name:<6}   : {gen}')
    print(divider)


if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)

    os.makedirs(RESULT_DIR, exist_ok=True)

    # ---- load data once, shared by all models ----
    print('\n' + '=' * 65)
    print('Loading dataset...')
    print('=' * 65)
    data_obj = Dataset_Loader_Generation('jokes', 'Short Jokes Dataset')
    data_obj.dataset_source_file_path = './data/stage_4_data/text_generation/data'
    loaded_data = data_obj.load()

    vocab     = loaded_data['vocab']
    raw_jokes = loaded_data['raw_jokes']
    seeds     = get_seeds(raw_jokes, vocab, n=7)

    # ---- train all three models ----
    all_losses   = {}    # model_name -> list of per-epoch losses
    all_models   = {}    # model_name -> trained Method object
    train_times  = {}    # model_name -> seconds

    for model_name, cfg in MODEL_CONFIGS.items():
        print('\n' + '=' * 65)
        print(f'Training {model_name} ...')
        print(f'  cell_type={cfg["cell_type"]}  embed={cfg["embed_dim"]}  '
              f'hidden={cfg["hidden_dim"]}  epochs={cfg["max_epoch"]}  '
              f'lr={cfg["learning_rate"]}  dropout={cfg["dropout"]}')
        print('=' * 65)
        t0 = time.time()
        trained = build_and_train(model_name, cfg, loaded_data)
        elapsed = time.time() - t0
        all_losses[model_name]  = trained.train_losses
        all_models[model_name]  = trained
        train_times[model_name] = elapsed
        print(f'  {model_name} done in {elapsed:.0f}s  '
              f'(initial loss: {trained.train_losses[0]:.4f}, '
              f'final loss: {trained.train_losses[-1]:.4f})')

    # ---- generate jokes from shared seeds ----
    results_table = []
    for seed in seeds:
        entry = {'seed': seed, 'original': find_original(raw_jokes, seed)}
        for model_name, cfg in MODEL_CONFIGS.items():
            model = all_models[model_name]
            entry[model_name] = model.generate(
                seed,
                max_len=60,
                temperature=cfg['temperature'],
                top_k=cfg['top_k'],
            )
        results_table.append(entry)

    # ---- console output ----
    print_comparison_table(results_table)

    # ---- metrics summary ----
    print('\n' + '=' * 65)
    print('EXPERIMENT SUMMARY')
    print(f'{"Model":<8} {"Cell":<6} {"embed":<7} {"hidden":<8} {"epochs":<8} '
          f'{"lr":<8} {"init loss":<12} {"final loss":<12} {"time(s)":<10}')
    print('-' * 85)
    for model_name, cfg in MODEL_CONFIGS.items():
        losses = all_losses[model_name]
        print(f'{model_name:<8} {cfg["cell_type"]:<6} {cfg["embed_dim"]:<7} '
              f'{cfg["hidden_dim"]:<8} {cfg["max_epoch"]:<8} '
              f'{cfg["learning_rate"]:<8} {losses[0]:<12.4f} {losses[-1]:<12.4f} '
              f'{train_times[model_name]:<10.0f}')
    print('=' * 65)

    # ---- save comparison plot ----
    plot_loss_comparison(all_losses, RESULT_DIR)

    # ---- write full report to file ----
    report_lines = [
        'Experiment Report: RNN vs. LSTM vs. GRU Text Generation\n',
        '=' * 65 + '\n\n',
        'Hyperparameter summary:\n',
    ]
    for model_name, cfg in MODEL_CONFIGS.items():
        report_lines.append(
            f'  {model_name}: cell={cfg["cell_type"]}, embed={cfg["embed_dim"]}, '
            f'hidden={cfg["hidden_dim"]}, epochs={cfg["max_epoch"]}, '
            f'lr={cfg["learning_rate"]}, dropout={cfg["dropout"]}, '
            f'temp={cfg["temperature"]}, top_k={cfg["top_k"]}\n'
        )

    report_lines.append('\nTraining results:\n')
    for model_name in MODEL_CONFIGS:
        losses = all_losses[model_name]
        report_lines.append(
            f'  {model_name}: init_loss={losses[0]:.4f}, '
            f'final_loss={losses[-1]:.4f}, '
            f'reduction={100*(losses[0]-losses[-1])/losses[0]:.1f}%, '
            f'time={train_times[model_name]:.0f}s\n'
        )

    report_lines.append('\nGenerated jokes (7 shared seeds):\n\n')
    for entry in results_table:
        seed_str = ' '.join(entry['seed'])
        report_lines.append(f'Seed: "{seed_str}"\n')
        original = entry.get('original', '')
        if original:
            report_lines.append(f'  Original : {original}\n')
        for model_name in MODEL_CONFIGS:
            report_lines.append(f'  {model_name:<6}   : {entry[model_name]}\n')
        report_lines.append('\n')

    report_lines.append('\nFull loss history per model:\n')
    for model_name in MODEL_CONFIGS:
        report_lines.append(f'\n{model_name}:\n')
        for e, l in enumerate(all_losses[model_name]):
            report_lines.append(f'  Epoch {e+1:>2}: {l:.4f}\n')

    report_path = os.path.join(RESULT_DIR, 'experiment_report.txt')
    with open(report_path, 'w') as f:
        f.writelines(report_lines)

    print(f'\nFull experiment report saved to: {report_path}')
    print('************ Experiment Finished ************')
