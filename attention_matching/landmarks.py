import torch

from meshtorch import sample_indices_farthest_point_sampling

def get_landmarks(shape_A, shape_B, faces_A, faces_B, distances_A, distances_B, landmarks_id_A, landmarks_id_B, landmarks_idx_A, landmarks_idx_B, args):

    landmarks_idx_A = landmarks_idx_A[:, (landmarks_id_A.unique(sorted=True).unsqueeze(-1) == landmarks_id_A).int().argmax(dim=1)]
    landmarks_id_A = landmarks_id_A[:, (landmarks_id_A.unique(sorted=True).unsqueeze(-1) == landmarks_id_A).int().argmax(dim=1)]
    landmarks_idx_B = landmarks_idx_B[:, (landmarks_id_B.unique(sorted=True).unsqueeze(-1) == landmarks_id_B).int().argmax(dim=1)]
    landmarks_id_B = landmarks_id_B[:, (landmarks_id_B.unique(sorted=True).unsqueeze(-1) == landmarks_id_B).int().argmax(dim=1)]

    # The subset of landmarks known on both A and B, used only for test-time evaluation.
    test_landmarks_idx_A = landmarks_idx_A[torch.isin(landmarks_id_A, landmarks_id_B)].squeeze(0)
    test_landmarks_id_A = landmarks_id_A[torch.isin(landmarks_id_A, landmarks_id_B)].squeeze(0)

    test_landmarks_idx_B = landmarks_idx_B[torch.isin(landmarks_id_B, test_landmarks_id_A)].squeeze(0)
    test_landmarks_id_B = landmarks_id_B[torch.isin(landmarks_id_B, test_landmarks_id_A)].squeeze(0)

    test_landmarks_idx_A = test_landmarks_idx_A[torch.sort(test_landmarks_id_A).indices]
    test_landmarks_idx_B = test_landmarks_idx_B[torch.sort(test_landmarks_id_B).indices]
    test_landmarks_id_A = test_landmarks_id_A[torch.sort(test_landmarks_id_A).indices]
    test_landmarks_id_B = test_landmarks_id_B[torch.sort(test_landmarks_id_B).indices]

    test_landmarks_idx = torch.vstack((test_landmarks_idx_A, test_landmarks_idx_B)).transpose(-1,-2)
    test_landmarks_id = test_landmarks_id_A

    # The n_lmk training landmarks (Section 4.1): used both as the Matching Loss anchors (Eq.7)
    # and as the points from which the D_lmk geodesic-distance features are computed.
    if args.landmarks:
        n_landmarks = args.landmarks

        if args.landmarks_idx_A is not None and args.landmarks_idx_B is not None:
            landmarks_idx = torch.zeros((len(args.landmarks_idx_A), 2), dtype=torch.long)
            landmarks_idx[:, 0] = torch.tensor(args.landmarks_idx_A, dtype=landmarks_idx.dtype)
            landmarks_id_A = landmarks_id_A.squeeze(-1)[torch.isin(landmarks_idx_A.squeeze(-1), landmarks_idx[:, 0])].to(args.device) if args.sparse else landmarks_idx[:, 0].to(args.device)
            landmarks_idx_A = landmarks_idx[:, 0].to(args.device)
            landmarks_idx[:, 1] = torch.tensor(args.landmarks_idx_B, dtype=landmarks_idx.dtype)
            landmarks_id_B = landmarks_id_B.squeeze(-1)[torch.isin(landmarks_idx_B.squeeze(-1), landmarks_idx[:, 1])].to(args.device) if args.sparse else landmarks_idx[:, 1].to(args.device)
            landmarks_idx_B = landmarks_idx[:, 1].to(args.device)
        elif args.landmarks_ids is not None:
            landmarks_id = torch.tensor(args.landmarks_ids).to(args.device)
            landmarks_idx_A = landmarks_idx_A.squeeze(0)[torch.isin(landmarks_id_A.squeeze(0), landmarks_id)]
            landmarks_idx_B = landmarks_idx_B.squeeze(0)[torch.isin(landmarks_id_B.squeeze(0), landmarks_id)]
            landmarks_id_A = landmarks_id_A.squeeze(0)[torch.isin(landmarks_id_A.squeeze(0), landmarks_id)]
            landmarks_id_B = landmarks_id_B.squeeze(0)[torch.isin(landmarks_id_B.squeeze(0), landmarks_id)]
        else:
            landmarks_idx_A = landmarks_idx_A.squeeze(0)[torch.isin(landmarks_id_A.squeeze(0), landmarks_id_B.squeeze(0))]
            landmarks_id_A = landmarks_id_A.squeeze(0)[torch.isin(landmarks_id_A.squeeze(0), landmarks_id_B.squeeze(0))]
            perm = sample_indices_farthest_point_sampling(shape_A[:, landmarks_idx_A.to(shape_A.device)], landmarks_idx_A.shape[0], distances=distances_A[:, landmarks_idx_A.to(distances_A.device)][:,:,landmarks_idx_A.to(distances_A.device)])[0][0].to(landmarks_idx_A.device)
            landmarks_idx_A = landmarks_idx_A[perm][:n_landmarks]
            landmarks_id_A = landmarks_id_A[perm][:n_landmarks]
            landmarks_idx_B = landmarks_idx_B.squeeze(0)[torch.isin(landmarks_id_B.squeeze(0), landmarks_id_A)]
            landmarks_id_B = landmarks_id_B.squeeze(0)[torch.isin(landmarks_id_B.squeeze(0), landmarks_id_A)]

        landmarks_idx_A = landmarks_idx_A[torch.sort(landmarks_id_A).indices]
        landmarks_idx_B = landmarks_idx_B[torch.sort(landmarks_id_B).indices]
        landmarks_id_A = landmarks_id_A[torch.sort(landmarks_id_A).indices]
        landmarks_id_B = landmarks_id_B[torch.sort(landmarks_id_B).indices]
        landmarks_idx = torch.vstack((landmarks_idx_A, landmarks_idx_B)).transpose(-1,-2)

        if args.normalize_dist:
            landmark_distances_A = torch.nn.functional.normalize(distances_A[:, landmarks_idx[:, 0].to(distances_A.device)], dim=-1, p=1).to(shape_A.device)
            landmark_distances_B = torch.nn.functional.normalize(distances_B[:, landmarks_idx[:, 1].to(distances_B.device)], dim=-1, p=1).to(shape_B.device)
        else:
            landmark_distances_A = distances_A[:, landmarks_idx[:, 0].to(distances_A.device)].to(shape_A.device)
            landmark_distances_B = distances_B[:, landmarks_idx[:, 1].to(distances_B.device)].to(shape_B.device)
    else:
        landmarks_idx = None
        landmark_distances_A = None
        landmark_distances_B = None

    return landmarks_idx, test_landmarks_idx, test_landmarks_id, landmark_distances_A, landmark_distances_B
