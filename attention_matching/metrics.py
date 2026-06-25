from geomfum.shape import TriangleMesh, PointCloud
from geomfum.laplacian import LaplacianFinder
import torch

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
        if faces_B is not None:

            source_shape = TriangleMesh(shape_B[i].cpu().numpy(), faces_B[i].cpu().numpy())
            L, A = source_shape.laplacian.find()

        else:
            source_shape = PointCloud(shape_B[i].cpu().numpy())
            L, A = source_shape.laplacian.find(
                laplacian_finder=LaplacianFinder.from_registry(mesh=False, which="robust")
            )
        v = shape_BA[i].float().cpu()  # [N, 3]
        if isinstance(L, torch.Tensor):
            L = L.float()
            Lv = torch.sparse.mm(L.to_sparse_coo().coalesce(), v)  # [N, 3]
        else:
            import numpy as np
            v_np = v.double().numpy()
            Lv = torch.tensor(np.column_stack([L @ v_np[:, c] for c in range(3)]), dtype=torch.float32)
        vLv = (v * Lv).sum(dim=0)  # [3]: v_i^T L v_i for each coordinate
        energy_x, energy_y, energy_z = vLv[0], vLv[1], vLv[2]

        total_energy[i] = energy_x + energy_y + energy_z

        return total_energy / shape_B.shape[1]
