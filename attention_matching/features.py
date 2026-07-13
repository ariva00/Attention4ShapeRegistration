import os

import numpy
import torch

from .model import AttentionMatcher

def _import_geomfum():
    try:
        from geomfum.shape import TriangleMesh
        from geomfum.descriptor.learned import FeatureExtractor
    except ImportError as e:
        raise ImportError(
            "--diffusionnet requires geomfum, which is not installed. Install it from "
            "source from https://github.com/3diglab/geomfum (see the README's "
            "Installation section)."
        ) from e
    torch.set_default_dtype(torch.float32)  # geomfum's geomstats import resets this to float64 again
    return TriangleMesh, FeatureExtractor

def get_model(args):
    in_dim = (3 if args.coordinates else 0) + args.landmarks
    if args.diffusionnet:
        in_dim = args.dim
    if args.load_3diff_features:
        in_dim = 2048
    model = AttentionMatcher(
        in_dim=in_dim,
        embed_dim=args.dim,
        num_heads=args.n_heads,
        num_layers=args.n_layers,
        fourier=args.fourier,
        self_only=args.self_only,
        matcher_only=args.matcher_only,
        symmetric=args.symmetric,
    )
    return model.to(device=args.device, dtype=args.model_dtype)

def get_diffusionnet(args):
    _, FeatureExtractor = _import_geomfum()
    return FeatureExtractor.from_registry(which="diffusionnet", descriptor=None, in_channels=3, out_channels=args.dim, device=args.device).to(args.device)

def compute_diffusionnet_features(diffusionnet, shape, faces):
    TriangleMesh, _ = _import_geomfum()
    mesh = TriangleMesh(shape[0], faces[0])
    # scipy's ARPACK solver used by find_spectrum always returns float64 eigenvalues,
    # regardless of input precision; geomfum's internal dtype checks require the
    # ambient default dtype to match, so cast to it for the duration of this call.
    torch.set_default_dtype(torch.float64)
    try:
        mesh.laplacian.find_spectrum(spectrum_size=128, set_as_basis=True)
    finally:
        torch.set_default_dtype(torch.float32)
    return diffusionnet(mesh).double().to(shape.device).unsqueeze(0)

def get_input_features(args, diffusionnet, shape_A, shape_B, faces_A, faces_B, name_A, name_B):
    if args.load_3diff_features:
        features_A = torch.from_numpy(numpy.load(os.path.join(args.path_features, f'{args.dataset}-3diff-features', f"{args.dataset}_features_{''.join([c for c in name_A if c.isnumeric()])}.npy"))).double().to(args.device).unsqueeze(0)
        features_B = torch.from_numpy(numpy.load(os.path.join(args.path_features, f'{args.dataset}-3diff-features', f"{args.dataset}_features_{''.join([c for c in name_B if c.isnumeric()])}.npy"))).double().to(args.device).unsqueeze(0)
    elif diffusionnet is not None:
        features_A = compute_diffusionnet_features(diffusionnet, shape_A, faces_A)
        features_B = compute_diffusionnet_features(diffusionnet, shape_B, faces_B)
    else:
        features_A = None
        features_B = None
    return features_A, features_B
