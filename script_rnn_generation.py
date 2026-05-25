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

# 3-word seeds taken from the first few jokes in the dataset (after lowercase cleaning)
SEED_WORDS = [
    ['what', 'did', 'the'],       # joke 1: "What did the bartender say..."
    ["don't", 'you', 'hate'],     # joke 2: "Don't you hate jokes about German sausage..."
    ['two', 'artists', 'had'],    # joke 3: "Two artists had an art contest..."
    ['why', 'did', 'the'],        # joke 4: "Why did the chicken cross the playground..."
    ['what', 'do', 'you'],
    ['why', 'does', 'a'],
    ['how', 'many', 'does'],
]


def pick_seeds_from_data(raw_jokes, n=5):
    '''Confirm seed words actually appear in the tokenized data; fall back if missing.'''
    found = []
    seen  = set()
    for joke in raw_jokes:
        if len(joke) >= 3:
            key = tuple(joke[:3])
            if key not in seen:
                seen.add(key)
                found.append(list(joke[:3]))
        if len(found) >= n:
            break
    return found


if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)

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
    print('************ Start ************')
    method_obj.run()

    # ---- loss curve ----
    os.makedirs(RESULT_DIR, exist_ok=True)
    plot_generation_loss(
        method_obj.train_losses,
        save_dir=RESULT_DIR,
        model_name='RNN_Generation',
    )

    # ---- generate jokes ----
    vocab = loaded_data['vocab']
    seeds = pick_seeds_from_data(loaded_data['raw_jokes'], n=7)
    # supplement with hardcoded seeds if needed
    for s in SEED_WORDS:
        if all(w in vocab for w in s) and s not in seeds:
            seeds.append(s)
        if len(seeds) >= 7:
            break

    print('\n' + '='*60)
    print('GENERATED JOKES (base RNN)')
    print('='*60)
    results_lines = ['=== Generated Jokes (base RNN) ===\n']

    for seed in seeds[:7]:
        generated = method_obj.generate(seed, max_len=60, temperature=0.8)
        line = f'Seed : {" ".join(seed)}\nOutput: {generated}\n'
        print(line)
        results_lines.append(line)

    # ---- compare with original jokes ----
    print('='*60)
    print('ORIGINAL JOKES (first 5 for comparison)')
    print('='*60)
    results_lines.append('\n=== Original Jokes (first 5) ===\n')
    for joke in loaded_data['raw_jokes'][:5]:
        original = ' '.join(joke)
        print(original)
        results_lines.append(original + '\n')

    with open(os.path.join(RESULT_DIR, 'generated_jokes.txt'), 'w') as f:
        f.writelines(results_lines)

    print(f'\nResults saved to {RESULT_DIR}/generated_jokes.txt')
    print('************ Finish ************')
