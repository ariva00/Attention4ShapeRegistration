import argparse
import os
import time

from attention_matching.args import get_train_args_parser, validate_train_args
from attention_matching.pipeline import main as main_train, main_test
from attention_matching.utils import CSVLogger

RESULT_COLUMNS = [
    "shape_A",
    "shape_B",
    "time",

    "mse_AB",
    "geod_AB",
    "chamfer_AB",
    "euclid_AB",
    "dirichlet_AB",
    "dispersion_AB",
    "mse_err_sym_AB",
    "geod_err_sym_AB",

    "mse_BA",
    "geod_BA",
    "chamfer_BA",
    "euclid_BA",
    "dirichlet_BA",
    "dispersion_BA",
    "mse_err_sym_BA",
    "geod_err_sym_BA",

    "mse_p2p_AB",
    "geod_p2p_AB",
    "chamfer_p2p_AB",
    "euclid_p2p_AB",
    "dirichlet_p2p_AB",
    "dispersion_p2p_AB",
    "mse_err_sym_p2p_AB",
    "geod_err_sym_p2p_AB",

    "mse_p2p_BA",
    "geod_p2p_BA",
    "chamfer_p2p_BA",
    "euclid_p2p_BA",
    "dirichlet_p2p_BA",
    "dispersion_p2p_BA",
    "mse_err_sym_p2p_BA",
    "geod_err_sym_p2p_BA",
]

def run_experiment(name, couples_fn, default_landmarks_idx=None, default_landmarks_ids=None):
    """Train (or test) the model on every couple of a dataset and log results to CSV.

    name: dataset label, used to build the "experiment_{name}.csv" output filename.
    couples_fn: callable(args) -> list of (model_A, model_B) name pairs, built by the
        dataset-specific caller (it needs args.path_data, so it runs after parsing).
    default_landmarks_idx: optional (idx_A, idx_B) explicit landmark vertex indices (e.g. FAUST).
    default_landmarks_ids: optional list of dataset-provided landmark ids (e.g. TOPKIDS/SMAL-R/SHREC'20).
    """
    parser = get_train_args_parser()

    parser.add_argument("--test", default=False, action=argparse.BooleanOptionalAction, help="run test.py on an already-trained model instead of training")

    args, unknown = parser.parse_known_args()

    if len(unknown) > 0:
        print("UNKNOWN ARGS: ", unknown)

    couples = couples_fn(args)

    if default_landmarks_idx is not None:
        if args.landmarks_idx_A is None:
            args.landmarks_idx_A = list(default_landmarks_idx[0])
        if args.landmarks_idx_B is None:
            args.landmarks_idx_B = list(default_landmarks_idx[1])
    if default_landmarks_ids is not None and args.landmarks_ids is None:
        args.landmarks_ids = list(default_landmarks_ids)

    args = validate_train_args(args)

    root = args.path_model
    if os.path.exists(args.path_model) and not args.test:
        print("EXPERIMENT ALREADY EXISTS - EXITING")
        exit(1)

    if args.test:
        filename = "test.csv"
    else:
        filename = f"experiment_{name}.csv"

    logger = CSVLogger(
        os.path.join(args.path_model, filename),
        RESULT_COLUMNS,
        append=False
    )

    print(f"{len(couples)} couples to match")
    for i, (model_A, model_B) in enumerate(couples):
        args.seed = i
        args.run_name = f"{model_B}_{model_A}"
        args.path_model = os.path.join(root, str(args.run_name))
        args.couple_names = [model_A, model_B]
        args.couple = [model_A, model_B]
        if args.test:
            tic = time.time_ns()
            (
                out_AB,
                out_BA,
                out_p2p_AB,
                out_p2p_BA,
            ) = main_test(args)
            toc = time.time_ns()
        else:
            tic = time.time_ns()
            (
                _, _,
                out_AB,
                out_BA,
                out_p2p_AB,
                out_p2p_BA,
                _,
                _,
            ) = main_train(args)
            toc = time.time_ns()
        logger.write((
            [model_A,model_B] +
            [toc-tic] +
            [o.item() for o in out_AB] +
            [o.item() for o in out_BA] +
            [o.item() for o in out_p2p_AB] +
            [o.item() for o in out_p2p_BA]
        ))
