import warnings

import numpy as np
import scipy.sparse as sparse
import torch

def _cotangent_laplacian(vertices, faces):
    """Cotangent-weight mesh Laplacian (stiffness matrix), following the standard
    construction also used as geomfum's default mesh Laplacian (pyFM's
    `cotangent_weights`): off-diagonal L[i,j] = -cot-weight of edge (i,j), diagonal
    L[i,i] = sum of the row's cot-weights, so rows sum to zero and v^T L v is the
    discrete Dirichlet energy of v."""
    n = vertices.shape[0]

    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]

    u1 = v3 - v2
    u2 = v1 - v3
    u3 = v2 - v1

    l1 = np.linalg.norm(u1, axis=1)
    l2 = np.linalg.norm(u2, axis=1)
    l3 = np.linalg.norm(u3, axis=1)

    cos1 = np.einsum("ij,ij->i", -u2, u3) / (l2 * l3)
    cos2 = np.einsum("ij,ij->i", u1, -u3) / (l1 * l3)
    cos3 = np.einsum("ij,ij->i", -u1, u2) / (l1 * l2)

    cos = np.concatenate([cos3, cos1, cos2])
    cot = 0.5 * cos / np.sqrt(1 - cos**2)

    I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    J = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])

    rows = np.concatenate([I, J, I, J])
    cols = np.concatenate([J, I, I, J])
    vals = np.concatenate([-cot, -cot, cot, cot])

    return sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()

def compute_dirichlet_energy(shape_B, shape_BA, faces_B=None):
    # Original code by Lorenzo Olearo (https://github.com/LorenzoOlearo)
    # and Giulio Viganò (https://github.com/gviga)

    # Create map from source to target vertices as vectors

    if shape_B.dim == 2:
        shape_B = shape_B.unsqueeze(0)
        faces_B = faces_B.unsqueeze(0)
        shape_BA = shape_BA.unsqueeze(0)
    
    total_energy = torch.zeros((len(shape_B),))

    for i in range(len(shape_B)):
        if faces_B is None:
            warnings.warn("compute_dirichlet_energy: no face connectivity available (faces_B is None), Dirichlet energy is not supported for point clouds; returning 0.")
            total_energy[i] = 0
            return total_energy / shape_B.shape[1]

        L = _cotangent_laplacian(shape_B[i].cpu().numpy(), faces_B[i].cpu().numpy())
        v = shape_BA[i].float().cpu()  # [N, 3]
        v_np = v.double().numpy()
        Lv = torch.tensor(np.column_stack([L @ v_np[:, c] for c in range(3)]), dtype=torch.float32)
        vLv = (v * Lv).sum(dim=0)  # [3]: v_i^T L v_i for each coordinate
        energy_x, energy_y, energy_z = vLv[0], vLv[1], vLv[2]

        total_energy[i] = energy_x + energy_y + energy_z

        return total_energy / shape_B.shape[1]
