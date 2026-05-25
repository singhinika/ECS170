'''
Stage 4-4: Train RNN language model on jokes dataset, then generate complete jokes
           from 3-word seeds that exist in the dataset.
Run from project root:
    python -m script.stage_4_script.script_rnn_generation
'''

import os
import numpy as np
import torch

from local_code.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation
from local_code.stage_4_code.Method_RNN_Generation     import Method_RNN_Generation
from local_code.stage_4_code.plot_utils                import plot_generation_loss

RESULT_DIR = './result/stage_4_result/RNN_Generation'

# Fixed 3-word seeds taken from the first few jokes in the dataset (after cleaning)
FIXED_SEEDS = [
    ['what', 'did', 'the'],        # joke 1: "What did the bartender say..."
    ["don't", 'you', 'hate'],      # joke 2: "Don't you hate jokes about German sausage..."
    ['two', 'artists', 'had'],     # joke 3: "Two artists had an art contest..."
    ['why', 'did', 'the'],         # joke 4: "Why did the chicken cross the playground..."
    ['what', 'do', 'you'],
    ['why', 'does', 'a'],
    ['how', 'do', 'you'],
]


def get_seeds(raw_jokes, vocab, n=7):
    '''Collect n unique seeds: first 3 words of real jokes that are all in vocab.'''
    seeds, seen = [], set()
    # prefer fixed seeds first (from known jokes)
    for s in FIXED_SEEDS:
        key = tuple(s)
        if key not in seen and all(w in vocab for w in s):
            seeds.append(s)
            seen.add(key)
        if len(seeds) >= n:
            break
    # fill remainder from dataset
    for joke in raw_jokes:
        if len(seeds) >= n:
            break
        if len(joke) >= 4:  # need at least 4 tokens so seed != full joke
            key = tuple(joke[:3])
            if key not in seen and all(w in vocab for w in joke[:3]):
                seeds.append(list(joke[:3]))
                seen.add(key)
    return seeds


def find_original(raw_jokes, seed):
    '''Return the first original joke that begins with the given seed words.'''
    for joke in raw_jokes:
        if joke[:len(seed)] == seed:
            return ' '.join(w for w in joke if w != '<EOS>')
    return None


if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)

    os.makedirs(RESULT_DIR, exist_ok=True)

    # ---- data ----
    data_obj = Dataset_Loader_Generation('jokes', 'Short Jokes Dataset')
    data_obj.dataset_source_file_path = './data/stage_4_data/text_generation/data'
    loaded_data = data_obj.load()

    # ---- model ----
    method_obj = Method_RNN_Generation('RNN-Generation', '')
    method_obj.data       = loaded_data
    method_obj.vocab_size = len(loaded_data['vocab'])
    method_obj.max_epoch  = 50

    # ---- train ----
    print('************ Start Training ************')
    method_obj.run()
    print(f'  Final training loss: {method_obj.train_losses[-1]:.4f}')

    # ---- save loss curve ----
    plot_generation_loss(
        method_obj.train_losses,
        save_dir=RESULT_DIR,
        model_name='RNN_Generation',
    )

    # ---- generate jokes ----
    vocab     = loaded_data['vocab']
    raw_jokes = loaded_data['raw_jokes']
    seeds     = get_seeds(raw_jokes, vocab, n=7)

    divider = '=' * 65
    output_lines = [
        'RNN Text Generation Results\n',
        f'Model: 2-layer RNN  |  Epochs: {method_obj.max_epoch}  '
        f'|  Final Loss: {method_obj.train_losses[-1]:.4f}\n',
        '\n',
    ]

    print('\n' + divider)
    print('GENERATED JOKES vs. ORIGINAL  (base RNN)')
    print(divider)

    for i, seed in enumerate(seeds, 1):
        generated = method_obj.generate(seed, max_len=60, temperature=0.8)
        original  = find_original(raw_jokes, seed)

        block = (
            f'[{i}] Seed : "{" ".join(seed)}"\n'
            f'    Generated : {generated}\n'
        )
        if original:
            block += f'    Original  : {original}\n'
        block += '\n'

        print(block, end='')
        output_lines.append(block)

    # ---- final loss summary ----
    summary = (
        f'\nTraining Loss per Epoch:\n'
        + '\n'.join(f'  Epoch {e+1:>2}: {l:.4f}'
                    for e, l in enumerate(method_obj.train_losses))
        + '\n'
    )
    output_lines.append(summary)

    # ---- save everything to file ----
    out_path = os.path.join(RESULT_DIR, 'generated_jokes.txt')
    with open(out_path, 'w') as f:
        f.writelines(output_lines)

    print(divider)
    print(f'Results saved to: {out_path}')
    print(f'Loss plot saved to: {RESULT_DIR}/RNN_Generation_loss.png')
    print('************ Finish ************')
