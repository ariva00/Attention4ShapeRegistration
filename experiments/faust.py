import itertools
import os

from experiments.common import run_experiment

def _couples(args):
    models = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(args.path_data, "MPI-FAUST", "training", "registrations")) if os.path.splitext(file)[1] == '.ply']
    models = [model for model in models if int(os.path.splitext(model)[0].split("_")[-1]) >= 80]
    models = sorted(models)
    return [(model_A, model_B) for (model_A, model_B) in list(itertools.product(models, models)) if model_A != model_B]

if __name__ == "__main__":
    run_experiment("faust", _couples, default_landmarks_idx=([412, 5891, 6593, 3323, 2119], [412, 5891, 6593, 3323, 2119]))
