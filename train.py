import os

from attention_matching.args import get_train_args_parser, validate_train_args
from attention_matching.pipeline import main

if __name__ == "__main__":

    parser = get_train_args_parser()

    args, unknown = parser.parse_known_args()

    if len(unknown) > 0:
        print("UNKNOWN ARGS: ", unknown)
    print("ARGS: ", args)

    args = validate_train_args(args)

    args.path_model = os.path.join(args.path_model, args.run_name)

    name_A, name_B, out_AB, out_BA, out_AB_p2p, out_BA_p2p, out_shape_BA, out_shape_AB, p2p_AB, p2p_BA, out_shape_A, out_shape_B, out_faces_A, out_faces_B = main(args)

    geod_err_AB = out_AB[1].item()
    geod_err_BA = out_BA[1].item()
    print(f"GEODESIC ERROR: {name_A} -> {name_B}: {geod_err_AB}, {name_B} -> {name_A}: {geod_err_BA}")
