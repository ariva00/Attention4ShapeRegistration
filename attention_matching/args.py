import argparse

import torch

def get_train_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", default="custom_trained_model", help="name of the run, determines the name of the saved model")

    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--n-epoch", type=int, default=1000, help="number of epochs")
    parser.add_argument("--pretrain-epoch", type=int, default=500, help="number of pretraining epochs")
    parser.add_argument("--path-data", default="dataset/", help="path to dir containing the dataset")
    parser.add_argument("--path-model", default="./models", help="path to dir where the model will be saved")
    parser.add_argument("--dataset", default="faust", choices=['faust', 'smal-r', 'shrec20', 'topkids', 'generic'], help="the dataset to train/test on; 'generic' loads two standalone mesh files given by --shape-a/--shape-b")
    parser.add_argument("--flip", default=False, action=argparse.BooleanOptionalAction, help="switch shape_A and shape_B")

    parser.add_argument("--n-heads", type=int, default=8, help="number of attention heads (h)")
    parser.add_argument("--n-layers", type=int, default=[4, 4], nargs="+", help="number of self-attention and cross-attention layers (l_self, l_cross)")
    parser.add_argument("--dim", type=int, default=256, help="embedding dimension (d_xi)")
    parser.add_argument("--fourier", type=int, default=8, help="number of random Fourier feature frequencies appended to the data vector (n_ff)")

    parser.add_argument("--self-only", default=False, action=argparse.BooleanOptionalAction, help="bypass the cross-attention matcher and correlate the (self-attention) embeddings directly; reproduces the Table 2/3 'E' and '-' ablation rows")
    parser.add_argument("--matcher-only", default=False, action=argparse.BooleanOptionalAction, help="bypass the self-attention feature extractor; reproduces the Table 2/3 'M' and '-' ablation rows")
    parser.add_argument("--symmetric", default=True, action=argparse.BooleanOptionalAction, help="apply the (weight-shared) cross-attention matcher in both directions and average rho_AB with rho_BA^T (Eq.4); --no-symmetric reproduces the Section 6.3 single-direction ablation")

    parser.add_argument("--device", default="auto", help="device to use for training, auto will use cuda if available, mps if available, else cpu")
    parser.add_argument("--precision", default="mixed", choices=["float", "double", "mixed"], help="float32 everywhere; double uses float64 everywhere; mixed keeps the attention-transformer matmuls in float32 for speed while keeping shape coordinates/distances/loss-metric computation in float64 for precision")
    parser.add_argument("--log-file", default="train.log", help="file to log the training process")

    parser.add_argument("--cpu-dist", default=False, action=argparse.BooleanOptionalAction, help="use cpu ram to store the distances matrix")
    parser.add_argument("--points-permutation", default=False, action=argparse.BooleanOptionalAction, help="apply random permutations to points")

    parser.add_argument("--landmarks", type=int, default=0, help="number of landmarks (n_lmk), used both as Matching Loss anchors and as the source points of the D_lmk distance feature")
    parser.add_argument("--normalize-dist", default=False, action=argparse.BooleanOptionalAction, help="normalize landmark distances")
    parser.add_argument("--coordinates", default=True, action=argparse.BooleanOptionalAction, help="include the 3D coordinates in the data vector")

    parser.add_argument("--resample", type=int, default=1000, help="number of samples for each epoch, 0 equals no resampling (Section 4.6.1)")
    parser.add_argument("--subsample", default=False, action=argparse.BooleanOptionalAction, help="perform resampling by subsampling existing vertices instead of sampling on faces, requires --resample")
    parser.add_argument("--resample-p", default=0.5, type=float, help="percentage of the resampled points that come from new random face locations as opposed to the original point set")
    parser.add_argument("--rmt", type=int, default=0, help="number of Rematching points, used when the couple doesn't fit in CPU memory (Section 4.6.2)")

    parser.add_argument("--seed", type=int, default=108, help="random seed")
    parser.add_argument("--couple", default=None, type=int, nargs=2, help="use specific couple indexes")
    parser.add_argument("--couple-names", default=None, type=str, nargs=2, help="use specific couple names, overrides --couple")
    parser.add_argument("--landmarks-idx-A", default=None, type=int, nargs='*', help="use specific landmark vertex indices for A")
    parser.add_argument("--landmarks-idx-B", default=None, type=int, nargs='*', help="use specific landmark vertex indices for B")
    parser.add_argument("--landmarks-ids", default=None, type=int, nargs='*', help="select landmarks by dataset-provided landmark id")
    parser.add_argument("--sparse", default=False, action=argparse.BooleanOptionalAction, help="the dataset only provides a sparse/quasi-dense correspondence")
    parser.add_argument("--shape-a", default=None, help="path to the first shape's mesh file (used with --dataset generic)")
    parser.add_argument("--shape-b", default=None, help="path to the second shape's mesh file (used with --dataset generic)")

    parser.add_argument("--path-features", default="features/", help="path to dir where precomputed features are stored")
    parser.add_argument("--diffusionnet", default=False, action=argparse.BooleanOptionalAction, help="use a DiffusionNet, trained jointly with the model, as input features instead of coordinates/landmark distances (Section 6.2)")
    parser.add_argument("--load-3diff-features", default=False, action=argparse.BooleanOptionalAction, help="load precomputed, frozen Diff3f features as input features instead of coordinates/landmark distances (Section 6.2)")

    return parser

def validate_train_args(args):
    if len(args.n_layers) == 1:
        args.n_layers = [args.n_layers[0], args.n_layers[0]]

    assert args.resample_p >= 0.0 and args.resample_p <= 1.0, "--resample-p must be in the interval [0.0, 1.0]"

    assert not (args.points_permutation and args.resample and not args.subsample), "--points-permutation requires --subsample when --resample is set"

    if args.couple_names is not None:
        args.couple = args.couple_names

    if args.device == "auto":
        args.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    args.dtype = torch.float32 if args.precision == "float" else torch.float64
    args.model_dtype = torch.float64 if args.precision == "double" else torch.float32

    if args.dataset in ("shrec20", "smal-r", "topkids", "generic") and not args.sparse:
        print("INCOMPATIBLE ARGS: chosen dataset does not have dense correspondences, --sparse will be set to True")
        args.sparse = True

    if args.dataset == "generic":
        assert args.shape_a is not None and args.shape_b is not None, "--dataset generic requires --shape-a and --shape-b"
        assert args.landmarks_idx_A is not None and args.landmarks_idx_B is not None, "--dataset generic requires --landmarks-idx-A and --landmarks-idx-B (paired landmark vertex indices)"
        assert len(args.landmarks_idx_A) == len(args.landmarks_idx_B), "--landmarks-idx-A and --landmarks-idx-B must have the same length"
        if args.couple is None:
            args.couple = [0, 1]

    if not args.coordinates and not args.landmarks:
        print("INCOMPATIBLE ARGS: --no-coordinates only available if --landmarks > 0, --no-coordinates will be set to False")
        args.coordinates = True

    if (args.landmarks_idx_A is not None and args.landmarks_idx_B is None) or (args.landmarks_idx_A is None and args.landmarks_idx_B is not None):
        print("INCOMPATIBLE ARGS: --landmarks-idx-A and --landmarks-idx-B must be initialized together, both will be assigned the value of the intialized one")
        if args.landmarks_idx_B is None:
            args.landmarks_idx_B = args.landmarks_idx_A
        if args.landmarks_idx_A is None:
            args.landmarks_idx_A = args.landmarks_idx_B

    if args.landmarks_idx_A is not None and not args.landmarks:
        print("INCOMPATIBLE ARGS: --landmarks-idx-A/-B given without --landmarks, --landmarks will be set to the number of provided indices")
        args.landmarks = len(args.landmarks_idx_A)

    if args.diffusionnet and args.load_3diff_features:
        print("INCOMPATIBLE ARGS: --diffusionnet and --load-3diff-features are not compatible, --diffusionnet will be set to False")
        args.diffusionnet = False

    return args
