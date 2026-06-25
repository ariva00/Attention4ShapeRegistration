import os

from experiments.common import run_experiment

def _couples(args):
    files = [
        'test-set0.txt',
        'test-set1.txt',
        'test-set2.txt',
        'test-set3.txt',
        'test-set4.txt',
    ]
    couples = []
    for file in files:
        with open(os.path.join('dataset', 'SHREC20b_lores', 'test-sets', file), "rt") as f:
            [couples.append(tuple(reversed(model.strip().split(',')))) for model in f.readlines()]
    return couples

if __name__ == "__main__":
    # landmarks_ids = [9, 19, 48, 41, 16, 32] with test-set0.txt
    # landmarks_ids = [5, 19, 20, 48, 45, 41] without test-set0.txt
    run_experiment("shrec20", _couples, default_landmarks_ids=[9, 19, 48, 41, 16, 32])
