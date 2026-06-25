import os

from experiments.common import run_experiment

def _couples(args):
    models = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(args.path_data, "TOPKIDS", "off"))]
    models = sorted(models)
    return [('kid00', model) for model in models if model != 'kid00']

if __name__ == "__main__":
    run_experiment("topkids", _couples, default_landmarks_ids=[9207, 3340, 3495, 11771, 11680])
