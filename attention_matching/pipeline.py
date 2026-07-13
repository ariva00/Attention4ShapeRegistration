import os
import time
import logging

import torch
from tqdm import tqdm

from meshtorch import sample_on_faces
from meshtorch.transforms import RandomRotateAllAxis

from .utils import set_seed, CSVLogger
from .data_loading import get_dataloader, get_shapes
from .landmarks import get_landmarks
from .features import get_model, get_diffusionnet, compute_diffusionnet_features, get_input_features
from .training import train
from .evaluation import test

def main(args):
    """Train (pretraining + training, Section 4.5) and evaluate a single A/B couple from scratch."""
    logging.basicConfig(filename=args.log_file, level=logging.DEBUG)
    logger = logging.getLogger(args.run_name)
    logger.info(f"training {args.run_name}")
    logger.info(f"args: {args}")

    set_seed(args.seed)
    os.makedirs(args.path_model, exist_ok=True)

    tic = time.time_ns()
    dataloader_train, dataset_train = get_dataloader(args)
    shape_A, shape_B, faces_A, faces_B, name_A, name_B, landmarks_id_A, landmarks_id_B, landmarks_idx_A, landmarks_idx_B, distances_A, distances_B, symmetric_map_A, symmetric_map_B, rmt_A, rmt_B = get_shapes(dataloader_train, dataset_train, args)
    print(f'{name_A} <-> {name_B}')

    landmark_idx, test_landmark_idx, test_landmark_id, landmark_distances_A, landmark_distances_B = get_landmarks(shape_A, shape_B, faces_A, faces_B, distances_A, distances_B, landmarks_id_A, landmarks_id_B, landmarks_idx_A, landmarks_idx_B, args)

    landmark_matches = landmark_idx.to(args.device) if args.landmarks else None

    # INITIALIZE MODEL
    model = get_model(args)
    diffusionnet = get_diffusionnet(args) if args.diffusionnet else None

    toc = time.time_ns()
    with open("times.txt", "at") as f:
        f.write(f"{args.run_name} preprocess: {toc-tic} | {shape_A.shape[1]} / {shape_B.shape[1]}\n")

    params = model.parameters() if diffusionnet is None else list(diffusionnet.parameters()) + list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    features_A, features_B = get_input_features(args, diffusionnet, shape_A, shape_B, faces_A, faces_B, name_A, name_B)

    print("TRAINING --------------------------------------------------------------------------------------------------")
    model = model.train()

    train_CSV_logger = CSVLogger(
        os.path.join(args.path_model, "train.csv"),
        ["loss", "time", "matching_loss", "double_permutation_loss", "chamfer_loss", "mse_err", "geod_err", "chamfer_err", "mse_err_AB", "geod_err_AB", "chamfer_err_AB", "mse_err_BA", "geod_err_BA", "chamfer_err_BA"],
        append=False
    )

    pretrain_CSV_logger = CSVLogger(
        os.path.join(args.path_model, "pretrain.csv"),
        ["loss", "time", "matching_loss", "double_permutation_loss", "chamfer_loss"],
        append=False
    )

    # PRETRAINING (Section 4.5.1): the cross-attention matcher is left untouched and only the
    # self-attention feature extractor is trained, against a random rotation of each shape, with
    # only the (dense) Matching Loss active -- every point is a landmark by construction.
    random_rotation = RandomRotateAllAxis(180)
    if args.pretrain_epoch:
        pretrain_params = model.self_parameters() if diffusionnet is None else list(diffusionnet.parameters()) + list(model.self_parameters())
        pretrain_optimizer = torch.optim.Adam(pretrain_params, lr=args.lr)
        tic = time.time_ns()
        for epoch in tqdm(range(args.pretrain_epoch)):
            augmented_shape_A = random_rotation(shape_A)
            augmented_shape_B = random_rotation(shape_B)

            if diffusionnet is not None:
                pretrain_features_A = compute_diffusionnet_features(diffusionnet, shape_A, faces_A)
                pretrain_augmented_features_A = compute_diffusionnet_features(diffusionnet, augmented_shape_A, faces_A)
                pretrain_features_B = compute_diffusionnet_features(diffusionnet, shape_B, faces_B)
                pretrain_augmented_features_B = compute_diffusionnet_features(diffusionnet, augmented_shape_B, faces_B)
            else:
                pretrain_features_A = features_A
                pretrain_augmented_features_A = features_A
                pretrain_features_B = features_B
                pretrain_augmented_features_B = features_B

            # A shape and its random rotation must be resampled at the *same* face/barycentric
            # (or vertex, if --subsample) locations, since pretraining relies on every point of
            # the resampled set trivially corresponding to itself across the rotation.
            if args.resample and args.subsample:
                sampling_idx_A = torch.randint(0, shape_A.shape[1], (1, args.resample)).expand(2, -1).to(shape_A.device)
                sampling_idx_B = torch.randint(0, shape_B.shape[1], (1, args.resample)).expand(2, -1).to(shape_B.device)
            elif args.resample:
                sampling_idx_A = sample_on_faces(shape_A, faces_A, args.resample)[1:]
                sampling_idx_B = sample_on_faces(shape_B, faces_B, args.resample)[1:]
            else:
                sampling_idx_A = None
                sampling_idx_B = None

            start = time.time()
            pretrain_optimizer.zero_grad(set_to_none=True)

            epoch_loss_A, losses_A, _ = train(
                model,
                shape_A, augmented_shape_A,
                faces_A, faces_A,
                distances_A, distances_A,
                landmark_distances_A, landmark_distances_A,
                None,
                args.resample, args.resample_p, args.subsample, sampling_idx_A,
                args.coordinates, pretrain_features_A, pretrain_augmented_features_A, args.points_permutation,
                pretraining=True,
            )
            epoch_loss_B, losses_B, _ = train(
                model,
                augmented_shape_B, shape_B,
                faces_B, faces_B,
                distances_B, distances_B,
                landmark_distances_B, landmark_distances_B,
                None,
                args.resample, args.resample_p, args.subsample, sampling_idx_B,
                args.coordinates, pretrain_augmented_features_B, pretrain_features_B, args.points_permutation,
                pretraining=True,
            )

            epoch_loss = epoch_loss_A + epoch_loss_B
            losses = [losses_A[i] + losses_B[i] for i in range(len(losses_A))]
            epoch_loss.backward()
            pretrain_optimizer.step()
            end = time.time()

            pretrain_CSV_logger.write([epoch_loss.item(), end-start] + [l.item() for l in losses])
        toc = time.time_ns()
        with open("times.txt", "at") as f:
            f.write(f"{args.run_name} pretrain: {toc-tic} | {shape_A.shape[1]} / {shape_B.shape[1]}\n")

    # TRAINING: A and B are fixed; the Matching Loss now only uses the known landmark matches.
    if args.n_epoch:
        tic = time.time_ns()
        for epoch in tqdm(range(args.n_epoch)):
            if diffusionnet is not None:
                features_A = compute_diffusionnet_features(diffusionnet, shape_A, faces_A)
                features_B = compute_diffusionnet_features(diffusionnet, shape_B, faces_B)

            start = time.time()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss, losses, shapes_out = train(
                model,
                shape_A, shape_B,
                faces_A, faces_B,
                distances_A, distances_B,
                landmark_distances_A, landmark_distances_B,
                landmark_matches,
                args.resample, args.resample_p, args.subsample, None,
                args.coordinates, features_A, features_B, args.points_permutation,
                pretraining=False,
            )
            epoch_loss.backward()
            optimizer.step()
            end = time.time()

            out_shape_A, out_shape_B, out_shape_AB, out_shape_BA, out_distances_A, out_distances_B = shapes_out
            _cdist_AB = torch.cdist(out_shape_AB, out_shape_B)
            _cdist_BA = torch.cdist(out_shape_BA, out_shape_A)
            chamfer_err_AB = _cdist_AB.min(dim=-1).values.mean() + _cdist_AB.min(dim=-2).values.mean()
            chamfer_err_BA = _cdist_BA.min(dim=-1).values.mean() + _cdist_BA.min(dim=-2).values.mean()
            chamfer_err = chamfer_err_AB + chamfer_err_BA
            mse_err_AB = torch.nn.functional.mse_loss(out_shape_AB, out_shape_B)
            mse_err_BA = torch.nn.functional.mse_loss(out_shape_BA, out_shape_A)
            mse_err = mse_err_AB + mse_err_BA
            geod_dist_AB = out_distances_B.gather(1, _cdist_AB.min(dim=-1).indices.unsqueeze(-1))
            geod_dist_BA = out_distances_A.gather(1, _cdist_BA.min(dim=-1).indices.unsqueeze(-1))
            geod_err_AB = geod_dist_AB.mean()
            geod_err_BA = geod_dist_BA.mean()
            geod_err = geod_err_AB + geod_err_BA

            errs = [mse_err.item(), geod_err.item(), chamfer_err.item(), mse_err_AB.item(), geod_err_AB.item(), chamfer_err_AB.item(), mse_err_BA.item(), geod_err_BA.item(), chamfer_err_BA.item()]

            train_CSV_logger.write([epoch_loss.item(), end-start] + [l.item() for l in losses] + errs)

        toc = time.time_ns()
        with open("times.txt", "at") as f:
            f.write(f"{args.run_name} train: {toc-tic} | {shape_A.shape[1]} / {shape_B.shape[1]}\n")

    model.eval()

    torch.save(model.state_dict(), os.path.join(args.path_model, "model.pt"))
    if diffusionnet is not None:
        torch.save(diffusionnet.state_dict(), os.path.join(args.path_model, "diffusionnet.model.pt"))

    with torch.no_grad():
        if diffusionnet is not None:
            features_A = compute_diffusionnet_features(diffusionnet, shape_A, faces_A)
            features_B = compute_diffusionnet_features(diffusionnet, shape_B, faces_B)
        P, p2p_AB, p2p_BA, out_AB, out_BA, out_AB_p2p, out_BA_p2p, geod_dist_AB, geod_dist_BA, geod_dist_p2p_AB, geod_dist_p2p_BA, eucl_dist_AB, eucl_dist_BA, eucl_dist_p2p_AB, eucl_dist_p2p_BA, shapes_out = test(model, shape_A, shape_B, faces_A, faces_B, distances_A, distances_B, landmark_distances_A, landmark_distances_B, test_landmark_idx, test_landmark_id, features_A, features_B, symmetric_map_A, symmetric_map_B, rmt_A, rmt_B, args)
    out_shape_A, out_shape_B, out_shape_AB, out_shape_BA, out_faces_A, out_faces_B = shapes_out

    return name_A, name_B, out_AB, out_BA, out_AB_p2p, out_BA_p2p, out_shape_BA, out_shape_AB, p2p_AB, p2p_BA, out_shape_A, out_shape_B, out_faces_A, out_faces_B

def main_test(args):
    """Load an already-trained model checkpoint and evaluate it on a couple."""
    logging.basicConfig(filename=args.log_file, level=logging.DEBUG)
    logger = logging.getLogger(args.run_name)
    logger.info(f"testing {args.run_name}")
    logger.info(f"args: {args}")

    set_seed(args.seed)

    dataloader_train, dataset_train = get_dataloader(args)
    shape_A, shape_B, faces_A, faces_B, name_A, name_B, landmarks_id_A, landmarks_id_B, landmarks_idx_A, landmarks_idx_B, distances_A, distances_B, symmetric_map_A, symmetric_map_B, rmt_A, rmt_B = get_shapes(dataloader_train, dataset_train, args)
    print(f'{name_A} <-> {name_B}')

    landmark_idx, test_landmark_idx, test_landmark_id, landmark_distances_A, landmark_distances_B = get_landmarks(shape_A, shape_B, faces_A, faces_B, distances_A, distances_B, landmarks_id_A, landmarks_id_B, landmarks_idx_A, landmarks_idx_B, args)

    model = get_model(args)
    model.load_state_dict(torch.load(os.path.join(args.path_model, "model.pt"), map_location=lambda storage, loc: storage))
    model.eval()

    diffusionnet = get_diffusionnet(args) if args.diffusionnet else None
    if diffusionnet is not None:
        diffusionnet.load_state_dict(torch.load(os.path.join(args.path_model, "diffusionnet.model.pt"), map_location=lambda storage, loc: storage))
        diffusionnet.eval()

    features_A, features_B = get_input_features(args, diffusionnet, shape_A, shape_B, faces_A, faces_B, name_A, name_B)

    print("TESTING --------------------------------------------------------------------------------------------------")

    with torch.no_grad():
        P, p2p_AB, p2p_BA, out_AB, out_BA, out_AB_p2p, out_BA_p2p, geod_dist_AB, geod_dist_BA, geod_dist_p2p_AB, geod_dist_p2p_BA, eucl_dist_AB, eucl_dist_BA, eucl_dist_p2p_AB, eucl_dist_p2p_BA, shapes_out = test(model, shape_A, shape_B, faces_A, faces_B, distances_A, distances_B, landmark_distances_A, landmark_distances_B, test_landmark_idx, test_landmark_id, features_A, features_B, symmetric_map_A, symmetric_map_B, rmt_A, rmt_B, args)
    out_shape_A, out_shape_B, out_shape_AB, out_shape_BA, out_faces_A, out_faces_B = shapes_out

    return out_AB, out_BA, out_AB_p2p, out_BA_p2p, out_shape_BA, out_shape_AB, p2p_AB, p2p_BA, out_shape_A, out_shape_B, out_faces_A, out_faces_B
