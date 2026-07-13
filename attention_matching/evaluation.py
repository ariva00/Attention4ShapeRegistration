import numpy
import torch

from .geometry import _geodesic_distance
from .metrics import compute_dirichlet_energy
from .utils import get_model_dtype

def test(model, shape_A, shape_B, faces_A, faces_B, distances_A, distances_B, landmark_distances_A, landmark_distances_B, test_landmark_idx, test_landmark_id, features_A, features_B, symmetric_map_A, symmetric_map_B, rmt_A, rmt_B, args):

    test_landmark_idx = test_landmark_idx.to(shape_A.device) if test_landmark_idx is not None else None

    if landmark_distances_A is not None and landmark_distances_B is not None:
        landmark_distances_A = landmark_distances_A.transpose(-1,-2)
        landmark_distances_B = landmark_distances_B.transpose(-1,-2)

    if features_A is not None and features_B is not None:
        x = features_A
        y = features_B
    elif args.landmarks:
        if args.coordinates:
            x = torch.cat((shape_A, landmark_distances_A), dim=-1)
            y = torch.cat((shape_B, landmark_distances_B), dim=-1)
        else:
            x = landmark_distances_A
            y = landmark_distances_B
    else:
        x = shape_A
        y = shape_B

    model_dtype = get_model_dtype(model)
    x = x.to(model_dtype)
    y = y.to(model_dtype)

    if args.resample and args.resample < shape_A.shape[1]:
        # Test-time resampling for shapes too large to fit in GPU memory (supplementary, Section 2):
        # relate every point of A against every point of B by tiling both into B = ceil(max(nA,nB)/n_S)
        # blocks of n_S points each, running the model on every (b_x, b_y) pair of blocks, and averaging
        # the overlapping regions of the resulting padded rho matrix.
        dim_A = shape_A.shape[1]
        permidx_A = torch.randperm(dim_A)
        x = x[:, permidx_A, :]
        shape_A = shape_A[:, permidx_A, :]
        gt_A = torch.zeros_like(permidx_A)
        gt_A[permidx_A] = torch.arange(dim_A)
        distances_A = distances_A[:, permidx_A, :][:, :, permidx_A]

        dim_B = shape_B.shape[1]
        permidx_B = torch.randperm(dim_B)
        y = y[:, permidx_B, :]
        shape_B = shape_B[:, permidx_B, :]
        gt_B = torch.zeros_like(permidx_B)
        gt_B[permidx_B] = torch.arange(dim_B)
        distances_B = distances_B[:, permidx_B, :][:, :, permidx_B]

        if landmark_distances_A is not None and landmark_distances_B is not None:
            landmark_distances_A = landmark_distances_A[:, permidx_A, :]
            landmark_distances_B = landmark_distances_B[:, permidx_B, :]

        B = int(numpy.ceil(max(dim_A, dim_B)/args.resample))
        N = args.resample * B
        idx_x = torch.arange(dim_A).unsqueeze(0)
        idx_y = torch.arange(dim_B).unsqueeze(0)
        idx_x = torch.cat([idx_x] + [idx_x for _ in range((N-dim_A)//idx_x.shape[1])] + [idx_x[:, :(N-dim_A)%idx_x.shape[1]]], dim=1).reshape((B, args.resample))
        idx_y = torch.cat([idx_y] + [idx_y for _ in range((N-dim_B)//idx_y.shape[1])] + [idx_y[:, :(N-dim_B)%idx_y.shape[1]]], dim=1).reshape((B, args.resample))
        x = torch.cat([x] + [x for _ in range((N-dim_A)//x.shape[1])] + [x[:, :(N-dim_A)%x.shape[1]]], dim=1).reshape((B, args.resample, x.shape[-1]))
        y = torch.cat([y] + [y for _ in range((N-dim_B)//y.shape[1])] + [y[:, :(N-dim_B)%y.shape[1]]], dim=1).reshape((B, args.resample, y.shape[-1]))

        P = torch.zeros((N, N), dtype=model_dtype).cpu()
        for b_x in range(B):
            for b_y in range(B):
                P_b = model(x[b_x].unsqueeze(0), y[b_y].unsqueeze(0))
                P[b_x*args.resample:(b_x+1)*args.resample, b_y*args.resample:(b_y+1)*args.resample] = P_b.cpu()
        P = torch.cat((P, torch.zeros((P.shape[0], dim_B - (N % dim_B)), dtype=model_dtype)), dim=1)
        P = torch.cat((P, torch.zeros((dim_A - (N % dim_A), P.shape[1]), dtype=model_dtype)), dim=0)
        P = P.reshape((-1, dim_A, P.shape[1])).sum(dim=0) / idx_x.cpu().unique(return_counts=True)[1].unsqueeze(-1)
        P = P.transpose(-1,-2).reshape((-1, dim_B, P.shape[0])).sum(dim=0).transpose(-1,-2) / idx_y.cpu().unique(return_counts=True)[1].unsqueeze(0)
        P = P.unsqueeze(0).to(x.device)

        shape_A = shape_A[:,gt_A,:]
        shape_B = shape_B[:,gt_B,:]

        P = P[:, gt_A, :][:, :, gt_B]

        distances_A = distances_A[:, gt_A, :][:, :, gt_A]
        distances_B = distances_B[:, gt_B, :][:, :, gt_B]

        if landmark_distances_A is not None and landmark_distances_B is not None:
            landmark_distances_A = landmark_distances_A[:, gt_A, :]
            landmark_distances_B = landmark_distances_B[:, gt_B, :]

    else:
        P = model(x, y)

    P = P.to(shape_A.dtype)

    shape_AB = P.softmax(dim=-1)@shape_B
    shape_BA = P.softmax(dim=-2).transpose(-1,-2)@shape_A

    test_landmark_idx_A = test_landmark_idx[:, 0] if test_landmark_idx is not None else None
    test_landmark_idx_B = test_landmark_idx[:, 1] if test_landmark_idx is not None else None

    if args.rmt:
        # Rematching (Section 4.6.2/Table 5): map the registration computed on the remeshed
        # discretization back onto the original discretization via the barycentric maps.
        shape_AB = torch.from_numpy(rmt_A[0].baryc_map(rmt_A[1])@(shape_AB[0].cpu().numpy())).unsqueeze(0).to(args.device)
        shape_A = torch.from_numpy(rmt_A[1]).double().unsqueeze(0).to(args.device)
        faces_A = rmt_A[2].long().unsqueeze(0).to(args.device)
        distances_A = _geodesic_distance(shape_A, faces_A)
        test_landmark_id_A = rmt_A[4][torch.isin(rmt_A[4], test_landmark_id)]
        test_landmark_idx_A = rmt_A[3][torch.isin(rmt_A[4], test_landmark_id)]
        test_landmark_idx_A = test_landmark_idx_A[test_landmark_id_A.sort().indices]
        test_landmark_id_A = test_landmark_id_A[test_landmark_id_A.sort().indices]

        shape_BA = torch.from_numpy(rmt_B[0].baryc_map(rmt_B[1])@(shape_BA[0].cpu().numpy())).unsqueeze(0).to(args.device)
        shape_B = torch.from_numpy(rmt_B[1]).double().unsqueeze(0).to(args.device)
        faces_B = rmt_B[2].long().unsqueeze(0).to(args.device)
        distances_B = _geodesic_distance(shape_B, faces_B)
        test_landmark_id_B = rmt_B[4][torch.isin(rmt_B[4], test_landmark_id)]
        test_landmark_idx_B = rmt_B[3][torch.isin(rmt_B[4], test_landmark_id)]
        test_landmark_idx_B = test_landmark_idx_B[test_landmark_id_B.sort().indices]
        test_landmark_id_B = test_landmark_id_B[test_landmark_id_B.sort().indices]

        test_landmark_id = test_landmark_id_A

    out_AB, geod_dist_AB, eucl_dist_AB = evaluate(shape_AB, shape_B, shape_A, faces_A, distances_B, args.sparse, test_landmark_idx_B, test_landmark_idx_A, symmetric_map_B)
    out_BA, geod_dist_BA, eucl_dist_BA = evaluate(shape_BA, shape_A, shape_B, faces_B, distances_A, args.sparse, test_landmark_idx_A, test_landmark_idx_B, symmetric_map_A)

    p2p_AB = torch.cdist(shape_AB, shape_B).min(dim=-1).indices.squeeze(0)
    shape_p2p_AB = shape_B[:, p2p_AB, :]
    out_AB_p2p, geod_dist_p2p_AB, eucl_dist_p2p_AB = evaluate(shape_p2p_AB, shape_B, shape_A, faces_A, distances_B, args.sparse, test_landmark_idx_B, test_landmark_idx_A, symmetric_map_B)

    p2p_BA = torch.cdist(shape_BA, shape_A).min(dim=-1).indices.squeeze(0)
    shape_p2p_BA = shape_A[:, p2p_BA, :]
    out_BA_p2p, geod_dist_p2p_BA, eucl_dist_p2p_BA = evaluate(shape_p2p_BA, shape_A, shape_B, faces_B, distances_A, args.sparse, test_landmark_idx_A, test_landmark_idx_B, symmetric_map_A)

    return P, p2p_AB, p2p_BA, out_AB, out_BA, out_AB_p2p, out_BA_p2p, geod_dist_AB, geod_dist_BA, geod_dist_p2p_AB, geod_dist_p2p_BA, eucl_dist_AB, eucl_dist_BA, eucl_dist_p2p_AB, eucl_dist_p2p_BA, [shape_A, shape_B, shape_AB, shape_BA, faces_A, faces_B]

def evaluate(shape_BA, shape_A, shape_B, faces_B, distances_A, evaluate_landmarks=False, landmark_idx_A=None, landmark_idx_B=None, symmetric_map_A=None):
    distances_BA_A = torch.cdist(shape_BA.to(distances_A.device), shape_A.to(distances_A.device))
    dispersion = torch.tensor((distances_BA_A.min(dim=-1).indices.squeeze(0).unique().size(0) / shape_A.size(1)) * (shape_A.size(1) / shape_BA.size(1)))
    chamfer_err = distances_BA_A.min(dim=-1).values.mean() + distances_BA_A.min(dim=-2).values.mean()
    dirich_err = compute_dirichlet_energy(shape_B, shape_BA, faces_B).mean()
    if evaluate_landmarks:
        mse_err = torch.nn.functional.mse_loss(shape_BA[:, landmark_idx_B,:], shape_A[:,landmark_idx_A,:])
    else:
        mse_err = torch.nn.functional.mse_loss(shape_BA, shape_A)
    if evaluate_landmarks:
        geod_dist_BA = distances_A[:, landmark_idx_A.to(distances_A.device)].gather(2, distances_BA_A[:, landmark_idx_B.to(distances_A.device)].min(dim=-1).indices.unsqueeze(-1))
    else:
        geod_dist_BA = distances_A.gather(2, distances_BA_A.min(dim=-1).indices.unsqueeze(-1))
    geod_err = geod_dist_BA.mean() / distances_A.max()
    if symmetric_map_A is not None:
        # Symmetry-invariant variant of mse/geod error: a match is also considered correct if it
        # lands on the bilaterally-symmetric counterpart of the ground-truth point.
        if evaluate_landmarks:
            mse_err_sym = torch.min(torch.nn.functional.mse_loss(shape_BA[:, landmark_idx_B,:], shape_A[:,landmark_idx_A,:]), torch.nn.functional.mse_loss(shape_BA[:, landmark_idx_B,:], shape_A[:, symmetric_map_A.cpu().squeeze(), :][:,landmark_idx_A,:]))
        else:
            mse_err_sym = torch.min(torch.nn.functional.mse_loss(shape_BA, shape_A), torch.nn.functional.mse_loss(shape_BA, shape_A[:, symmetric_map_A.cpu().squeeze(), :]))
        distances_BA_A_sym = distances_BA_A[:, :, symmetric_map_A.cpu().squeeze()]
        if evaluate_landmarks:
            geod_dist_BA_sym = distances_A[:, symmetric_map_A.cpu().squeeze(), :][:, :, symmetric_map_A.cpu().squeeze()][:, landmark_idx_A.to(distances_A.device)].gather(2, distances_BA_A_sym[:, landmark_idx_B.to(distances_A.device)].min(dim=-1).indices.unsqueeze(-1))
        else:
            geod_dist_BA_sym = distances_A[:, symmetric_map_A.cpu().squeeze(), :][:, :, symmetric_map_A.cpu().squeeze()].gather(2, distances_BA_A_sym.min(dim=-1).indices.unsqueeze(-1))
        geod_err_sym = torch.min(geod_dist_BA, geod_dist_BA_sym).mean() / distances_A.max()
    else:
        mse_err_sym, geod_err_sym = torch.tensor(0), torch.tensor(0)
    max_euclid = torch.cdist(shape_A.to(distances_A.device), shape_A.to(distances_A.device)).max()
    if evaluate_landmarks:
        eucl_dist_BA = (shape_BA[:, landmark_idx_B] - shape_A[:, landmark_idx_A]).norm(dim=-1)
    else:
        eucl_dist_BA = (shape_BA - shape_A).norm(dim=-1)
    eucl_err = eucl_dist_BA.mean() / max_euclid
    return (mse_err, geod_err, chamfer_err, eucl_err, dirich_err, dispersion, mse_err_sym, geod_err_sym), geod_dist_BA, eucl_dist_BA
