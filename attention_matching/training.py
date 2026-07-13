import numpy
import torch

from meshtorch import faces_to_edges, sample_on_faces, truncate_shape, sample_indices_random

from .utils import get_model_dtype

def train(
        model,
        shape_A, shape_B,
        faces_A, faces_B,
        distances_A, distances_B,
        landmark_distances_A, landmark_distances_B,
        landmark_matches,
        resample,
        resample_p,
        subsample,
        sampling_idx,
        coordinates_features,
        features_A, features_B,
        points_permutation,
        pretraining=False,
    ):

    # DATA PREPARATION

    if landmark_distances_A is not None and landmark_distances_B is not None:
        landmark_distances_A = landmark_distances_A.transpose(-1,-2)
        landmark_distances_B = landmark_distances_B.transpose(-1,-2)

    dim_A = shape_A.shape[1]
    dim_B = shape_B.shape[1]

    if points_permutation:
        permidx_A = torch.randperm(dim_A)
        shape_A = shape_A[:, permidx_A, :]
        gt_A = torch.zeros_like(permidx_A)
        gt_A[permidx_A] = torch.arange(dim_A)
        distances_A = distances_A[:, permidx_A, :][:, :, permidx_A]

        permidx_B = torch.randperm(dim_B)
        shape_B = shape_B[:, permidx_B, :]
        gt_B = torch.zeros_like(permidx_B)
        gt_B[permidx_B] = torch.arange(dim_B)
        distances_B = distances_B[:, permidx_B, :][:, :, permidx_B]

        if landmark_distances_A is not None and landmark_distances_B is not None:
            landmark_distances_A = landmark_distances_A[:, permidx_A, :]
            landmark_distances_B = landmark_distances_B[:, permidx_B, :]

        if features_A is not None and features_B is not None:
            features_A = features_A[:, permidx_A, :]
            features_B = features_B[:, permidx_B, :]
    else:
        gt_A = torch.arange(dim_A)
        gt_B = torch.arange(dim_B)

    if features_A is not None and features_B is not None:
        x = features_A
        y = features_B
    elif landmark_distances_A is not None and landmark_distances_B is not None:
        if coordinates_features:
            x = torch.cat((shape_A, landmark_distances_A), dim=-1)
            y = torch.cat((shape_B, landmark_distances_B), dim=-1)
        else:
            x = landmark_distances_A
            y = landmark_distances_B
    else:
        x = shape_A
        y = shape_B

    # FORWARD

    if resample:
        if subsample:
            if sampling_idx is None:
                sampled_idx_A, _ = sample_indices_random(x, resample)
                sampled_idx_B, _ = sample_indices_random(y, resample)
            else:
                sampled_idx_A = gt_A.to(sampling_idx.device)[sampling_idx[0]].unsqueeze(0)
                sampled_idx_B = gt_B.to(sampling_idx.device)[sampling_idx[1]].unsqueeze(0)
            if landmark_matches is not None:
                sampled_idx_A = torch.cat((gt_A.to(sampled_idx_A.device)[landmark_matches[:, 0]].unsqueeze(0), sampled_idx_A), dim=1)
                sampled_idx_B = torch.cat((gt_B.to(sampled_idx_B.device)[landmark_matches[:, 1]].unsqueeze(0), sampled_idx_B), dim=1)
            x, _, _ = truncate_shape(x, faces_to_edges(faces_A), idx=sampled_idx_A)
            y, _, _ = truncate_shape(y, faces_to_edges(faces_B), idx=sampled_idx_B)
            if landmark_distances_A is not None:
                landmark_distances_A = landmark_distances_A.gather(1, sampled_idx_A.unsqueeze(-1).expand(-1, -1, landmark_distances_A.size(-1)))
            if landmark_distances_B is not None:
                landmark_distances_B = landmark_distances_B.gather(1, sampled_idx_B.unsqueeze(-1).expand(-1, -1, landmark_distances_B.size(-1)))
            shape_A, _, _ = truncate_shape(shape_A, faces_to_edges(faces_A), idx=sampled_idx_A)
            shape_B, _, _ = truncate_shape(shape_B, faces_to_edges(faces_B), idx=sampled_idx_B)
            distances_A = distances_A[:, sampled_idx_A[0].to(distances_A.device), :][:, :, sampled_idx_A[0].to(distances_A.device)]
            distances_B = distances_B[:, sampled_idx_B[0].to(distances_B.device), :][:, :, sampled_idx_B[0].to(distances_B.device)]
        else:
            if sampling_idx is None:
                _, sampled_faces_A, sampled_coordinates_A = sample_on_faces(x, faces_A, int(numpy.ceil(resample*resample_p)))
                _, sampled_faces_B, sampled_coordinates_B = sample_on_faces(y, faces_B, int(numpy.ceil(resample*resample_p)))

                anchors_matches = torch.randint(shape_A.size(1), (int(numpy.floor(resample*(1-resample_p))), 2)).to(shape_A.device)

                sampled_anchors_faces_A = (anchors_matches[:, 0].unsqueeze(-1).unsqueeze(-1) == faces_A).int().argmax(dim=1).max(dim=1).values
                sampled_anchors_coordinates_A = (faces_A[:, sampled_anchors_faces_A] == anchors_matches[:,0].unsqueeze(0).unsqueeze(-1)).max(dim=-1).indices
                sampled_anchors_coordinates_A = (sampled_anchors_coordinates_A==0) * torch.tensor([0,0]).to(sampled_anchors_coordinates_A.device).unsqueeze(-1) + (sampled_anchors_coordinates_A==1) * torch.tensor([1,0]).to(sampled_anchors_coordinates_A.device).unsqueeze(-1) + (sampled_anchors_coordinates_A==2) * torch.tensor([0,1]).to(sampled_anchors_coordinates_A.device).unsqueeze(-1)
                sampled_faces_A = torch.cat((sampled_anchors_faces_A.unsqueeze(0), sampled_faces_A), dim=-1)
                sampled_coordinates_A = torch.cat((sampled_anchors_coordinates_A.transpose(-1,-2).unsqueeze(0), sampled_coordinates_A), dim=-2)

                sampled_anchors_faces_B = (anchors_matches[:, 1].unsqueeze(-1).unsqueeze(-1) == faces_B).int().argmax(dim=1).max(dim=1).values
                sampled_anchors_coordinates_B = (faces_B[:, sampled_anchors_faces_B] == anchors_matches[:,1].unsqueeze(0).unsqueeze(-1)).max(dim=-1).indices
                sampled_anchors_coordinates_B = (sampled_anchors_coordinates_B==0) * torch.tensor([0,0]).to(sampled_anchors_coordinates_B.device).unsqueeze(-1) + (sampled_anchors_coordinates_B==1) * torch.tensor([1,0]).to(sampled_anchors_coordinates_B.device).unsqueeze(-1) + (sampled_anchors_coordinates_B==2) * torch.tensor([0,1]).to(sampled_anchors_coordinates_B.device).unsqueeze(-1)
                sampled_faces_B = torch.cat((sampled_anchors_faces_B.unsqueeze(0), sampled_faces_B), dim=-1)
                sampled_coordinates_B = torch.cat((sampled_anchors_coordinates_B.transpose(-1,-2).unsqueeze(0), sampled_coordinates_B), dim=-2)
            else:
                sampled_faces_A, sampled_coordinates_A = sampling_idx
                sampled_faces_B, sampled_coordinates_B = sampling_idx
            if landmark_matches is not None:
                sampled_landmark_faces_A = (landmark_matches[:, 0].unsqueeze(-1).unsqueeze(-1) == faces_A).int().argmax(dim=1).max(dim=1).values
                sampled_landmark_coordinates_A = (faces_A[:, sampled_landmark_faces_A] == landmark_matches[:,0].unsqueeze(0).unsqueeze(-1)).max(dim=-1).indices
                sampled_landmark_coordinates_A = (sampled_landmark_coordinates_A==0) * torch.tensor([0,0]).to(sampled_landmark_coordinates_A.device).unsqueeze(-1) + (sampled_landmark_coordinates_A==1) * torch.tensor([1,0]).to(sampled_landmark_coordinates_A.device).unsqueeze(-1) + (sampled_landmark_coordinates_A==2) * torch.tensor([0,1]).to(sampled_landmark_coordinates_A.device).unsqueeze(-1)
                sampled_faces_A = torch.cat((sampled_landmark_faces_A.unsqueeze(0), sampled_faces_A), dim=-1)
                sampled_coordinates_A = torch.cat((sampled_landmark_coordinates_A.transpose(-1,-2).unsqueeze(0), sampled_coordinates_A), dim=-2)

                sampled_landmark_faces_B = (landmark_matches[:, 1].unsqueeze(-1).unsqueeze(-1) == faces_B).int().argmax(dim=1).max(dim=1).values
                sampled_landmark_coordinates_B = (faces_B[:, sampled_landmark_faces_B] == landmark_matches[:,1].unsqueeze(0).unsqueeze(-1)).max(dim=-1).indices
                sampled_landmark_coordinates_B = (sampled_landmark_coordinates_B==0) * torch.tensor([0,0]).to(sampled_landmark_coordinates_B.device).unsqueeze(-1) + (sampled_landmark_coordinates_B==1) * torch.tensor([1,0]).to(sampled_landmark_coordinates_B.device).unsqueeze(-1) + (sampled_landmark_coordinates_B==2) * torch.tensor([0,1]).to(sampled_landmark_coordinates_B.device).unsqueeze(-1)
                sampled_faces_B = torch.cat((sampled_landmark_faces_B.unsqueeze(0), sampled_faces_B), dim=-1)
                sampled_coordinates_B = torch.cat((sampled_landmark_coordinates_B.transpose(-1,-2).unsqueeze(0), sampled_coordinates_B), dim=-2)
            x = sample_on_faces(x, faces_A, resample, sampled_faces_A, sampled_coordinates_A)[0]
            y = sample_on_faces(y, faces_B, resample, sampled_faces_B, sampled_coordinates_B)[0]
            if landmark_distances_A is not None:
                landmark_distances_A = sample_on_faces(landmark_distances_A, faces_A, resample, sampled_faces_A, sampled_coordinates_A)[0]
            if landmark_distances_B is not None:
                landmark_distances_B = sample_on_faces(landmark_distances_B, faces_B, resample, sampled_faces_B, sampled_coordinates_B)[0]
            shape_A = sample_on_faces(shape_A, faces_A, resample, sampled_faces_A, sampled_coordinates_A)[0]
            shape_B = sample_on_faces(shape_B, faces_B, resample, sampled_faces_B, sampled_coordinates_B)[0]
            distances_A = sample_on_faces(sample_on_faces(distances_A, faces_A.to(distances_A.device), resample, sampled_faces_A.to(distances_A.device), sampled_coordinates_A.to(distances_A.device))[0].transpose(-1,-2), faces_A.to(distances_A.device), resample, sampled_faces_A.to(distances_A.device), sampled_coordinates_A.to(distances_A.device))[0]
            distances_B = sample_on_faces(sample_on_faces(distances_B, faces_B.to(distances_B.device), resample, sampled_faces_B.to(distances_B.device), sampled_coordinates_B.to(distances_B.device))[0].transpose(-1,-2), faces_B.to(distances_B.device), resample, sampled_faces_B.to(distances_B.device), sampled_coordinates_B.to(distances_B.device))[0]

        landmark_matches = torch.arange(landmark_matches.shape[0]).unsqueeze(1).expand(-1,2).long().to(landmark_matches.device) if landmark_matches is not None else None
        dim_A = shape_A.shape[1]
        dim_B = shape_B.shape[1]

    distances_A = distances_A.to(shape_A.device)
    distances_B = distances_B.to(shape_B.device)

    model_dtype = get_model_dtype(model)
    attn_matchings, hiddens = model(x.to(model_dtype), y.to(model_dtype), return_hiddens=True)
    attn_matchings = attn_matchings.to(shape_A.dtype)

    # LOSS COMPUTATION
    loss = torch.tensor(0.0, device=shape_A.device, dtype=shape_A.dtype)

    attn_matchings_AB = attn_matchings.softmax(dim=-1)
    attn_matchings_BA = attn_matchings.softmax(dim=-2)

    if points_permutation and not resample:
        shape_A = shape_A[:,gt_A,:]
        shape_B = shape_B[:,gt_B,:]

        distances_A = distances_A[:, gt_A, :][:, :, gt_A]
        distances_B = distances_B[:, gt_B, :][:, :, gt_B]

        attn_matchings_AB = attn_matchings_AB[:, gt_A, :][:, :, gt_B]
        attn_matchings_BA = attn_matchings_BA[:, gt_A, :][:, :, gt_B]

        if landmark_distances_A is not None and landmark_distances_B is not None:
            landmark_distances_A = landmark_distances_A[:, gt_A, :]
            landmark_distances_B = landmark_distances_B[:, gt_B, :]

    shape_AB = attn_matchings_AB@shape_B
    shape_BA = attn_matchings_BA.transpose(-1,-2)@shape_A

    # MATCHING LOSS (Eq.7). During pretraining every point is a landmark by construction
    # (landmark_matches is None and pretraining=True activates the dense variant).
    matching_loss = torch.tensor(0.0, device=shape_A.device, dtype=shape_A.dtype)

    if landmark_matches is not None:
        landmark_shape_A = shape_A[:,landmark_matches[:,0]]
        landmark_shape_B = shape_B[:,landmark_matches[:,1]]

        landmark_shape_AB = shape_AB[:,landmark_matches[:,0]]
        landmark_shape_BA = shape_BA[:,landmark_matches[:,1]]

        matching_loss += (landmark_shape_AB - landmark_shape_B).norm(dim=-1).mean()
        matching_loss += (landmark_shape_BA - landmark_shape_A).norm(dim=-1).mean()
    elif pretraining:
        matching_loss += (shape_AB - shape_B).norm(dim=-1).mean()
        matching_loss += (shape_BA - shape_A).norm(dim=-1).mean()

    loss += matching_loss

    # DOUBLE PERMUTATION LOSS (Eq.8) and CHAMFER LOSS (Eq.9): only active outside pretraining,
    # per Section 4.5.1 ("the only active loss [during pretraining] is the Matching Loss").
    double_permutation_loss = torch.tensor(0.0, device=shape_A.device, dtype=shape_A.dtype)
    chamfer_loss = torch.tensor(0.0, device=shape_A.device, dtype=shape_A.dtype)

    if not pretraining:
        double_permutation_loss += (attn_matchings_BA.transpose(-1,-2)@shape_AB - shape_B).norm(dim=-1).mean() + (attn_matchings_AB@shape_BA - shape_A).norm(dim=-1).mean()
        loss += double_permutation_loss

        chamfer_A = torch.cdist(shape_BA, shape_A).min(dim=-1).values.mean() + torch.cdist(shape_BA, shape_A).min(dim=-2).values.mean()
        chamfer_B = torch.cdist(shape_AB, shape_B).min(dim=-1).values.mean() + torch.cdist(shape_AB, shape_B).min(dim=-2).values.mean()
        chamfer_loss += chamfer_A + chamfer_B
        loss += chamfer_loss

    return loss, [matching_loss, double_permutation_loss, chamfer_loss], [shape_A, shape_B, shape_AB, shape_BA, distances_A, distances_B]
