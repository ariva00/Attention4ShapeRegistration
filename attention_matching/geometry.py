import numpy

import torch

from meshtorch import faces_to_edges, edges_to_adj
from scipy.sparse.csgraph import dijkstra

def _geodesic_distance(points:torch.Tensor, faces:torch.Tensor, source:torch.Tensor=None):
    edges = faces_to_edges(faces)
    distance = torch.cdist(points.cpu(), points.cpu())
    adj = edges_to_adj(edges, num_vertices=points.size(1)).cpu().to_dense()
    adj = adj*distance
    geodesic = torch.cat([torch.tensor(dijkstra(a.cpu().numpy(), indices=source.cpu().numpy() if source is not None else source)).to(points.dtype).unsqueeze(0) for a in adj.unbind(dim=0)], dim=0)
    distance = distance[:, source.cpu()] if source is not None else distance
    geodesic = geodesic.where(geodesic != torch.inf, distance)
    return geodesic

def compute_shape_info(shape, faces, args):
    shape = shape.to(args.dtype)
    distances = _geodesic_distance(shape, faces)
    if not args.cpu_dist:
        distances = distances.to(args.device)
    return distances

def compute_rematching(shape, faces, landmarks_idx, landmarks_id, args):
    """Rematching (Section 4.6.2/Table 5): remesh a shape down to args.rmt points so the couple
    fits in CPU memory, returning the data needed to later map a registration back onto the
    original discretization via barycentric coordinates."""
    try:
        from PyRMT import RMTMesh
    except ImportError as e:
        raise ImportError(
            "--rmt requires PyRMT, which is not installed. Install it from the "
            "`python-binding` branch of https://github.com/filthynobleman/rematching "
            "(see the README's Installation section)."
        ) from e

    rmt_v = numpy.asfortranarray(shape[0].cpu().numpy())
    rmt_f = numpy.asfortranarray(faces[0].cpu().int().numpy())
    rmt = RMTMesh(rmt_v, rmt_f)
    rmt.make_manifold()
    rmt = rmt.remesh(args.rmt)
    rmt.clean_up()
    rmt_shape = torch.from_numpy(rmt.vertices.copy()).double().unsqueeze(0).to(args.device)
    rmt_faces = torch.from_numpy(rmt.triangles.copy()).long().unsqueeze(0).to(args.device)
    rmt_landmarks_idx = torch.cdist(shape[:, landmarks_idx], rmt_shape).min(dim=-1).indices.squeeze(1) if landmarks_idx is not None else None

    return rmt_shape, rmt_faces, rmt_landmarks_idx, [rmt, rmt_v, faces[0].cpu(), landmarks_idx, landmarks_id]
