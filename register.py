import os

import numpy as np
import trimesh

from attention_matching.args import get_train_args_parser, validate_train_args
from attention_matching.pipeline import main

if __name__ == "__main__":

    parser = get_train_args_parser()

    args, unknown = parser.parse_known_args()

    if len(unknown) > 0:
        print("UNKNOWN ARGS: ", unknown)

    args.dataset = "generic"
    print("ARGS: ", args)

    args = validate_train_args(args)

    args.path_model = os.path.join(args.path_model, args.run_name)

    (
        name_A, name_B,
        out_AB, out_BA, out_AB_p2p, out_BA_p2p,
        out_shape_BA, out_shape_AB,
        p2p_AB, p2p_BA,
        out_shape_A, out_shape_B,
        out_faces_A, out_faces_B,
    ) = main(args)

    trimesh.Trimesh(
        vertices=out_shape_AB.squeeze(0).detach().cpu().numpy(),
        faces=out_faces_A.squeeze(0).detach().cpu().numpy(),
        process=False,
    ).export(os.path.join(args.path_model, f"{name_A}_registered_to_{name_B}.ply"))

    trimesh.Trimesh(
        vertices=out_shape_BA.squeeze(0).detach().cpu().numpy(),
        faces=out_faces_B.squeeze(0).detach().cpu().numpy(),
        process=False,
    ).export(os.path.join(args.path_model, f"{name_B}_registered_to_{name_A}.ply"))

    np.savetxt(os.path.join(args.path_model, "p2p_AB.txt"), p2p_AB.detach().cpu().numpy(), fmt="%d")
    np.savetxt(os.path.join(args.path_model, "p2p_BA.txt"), p2p_BA.detach().cpu().numpy(), fmt="%d")

    metric_names = ["mse", "geod", "chamfer", "eucl", "dirichlet", "dispersion", "mse_sym", "geod_sym"]
    print(f"metrics {name_A} -> {name_B}: " + ", ".join(f"{n}={v.item():.4f}" for n, v in zip(metric_names, out_AB)))
    print(f"metrics {name_B} -> {name_A}: " + ", ".join(f"{n}={v.item():.4f}" for n, v in zip(metric_names, out_BA)))
    print(f"saved registration + checkpoint to {args.path_model}")
