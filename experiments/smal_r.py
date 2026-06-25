import itertools
import os

from experiments.common import run_experiment

def _couples(args):
    with open(os.path.join(args.path_data, "SMAL_r", "test_cat.txt"), "rt") as f:
        models = [model.strip() for model in f.readlines()]
    models = sorted(models)
    return [(model_A, model_B) for (model_A, model_B) in list(itertools.product(models, models)) if model_A != model_B]

if __name__ == "__main__":
    run_experiment("smal_r", _couples, default_landmarks_ids=[3162, 1931, 3731, 1399, 1111, 1001])
