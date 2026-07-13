import numpy as np
import os
import torch
import trimesh
from torch.utils.data import Dataset
from scipy.io import loadmat

# Original transmatching repository https://github.com/GiovanniTRA/transmatching
# The orignial code is distributed under the MIT license reported in the license folder

class FaustDataset(Dataset):

    def __init__(self, in_path, dataset="MPI-FAUST", transform=None, dtype=torch.float32):
        self.in_path = os.path.join(in_path, dataset, "training", "registrations")
        self.models = [os.path.splitext(file)[0] for file in os.listdir(self.in_path) if os.path.splitext(file)[1] == '.ply']
        self.transform = transform
        self.dtype = dtype
        self.symmetric_map = torch.from_numpy(np.loadtxt(os.path.join(in_path, dataset, "original_to_sym_map.txt"), dtype=int)).long()

    def __len__(self):
        return len(self.models)

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.models.index(index)
        model = self.models[index]
        mesh = trimesh.load_mesh(os.path.join(self.in_path, model + ".ply"), process=False)
        faces = torch.from_numpy(mesh.faces).long()
        shape = torch.from_numpy(mesh.vertices).to(self.dtype)
        if self.transform:
            shape = self.transform(shape)

        landmarks_id = torch.arange(shape.shape[0])
        landmarks_idx = torch.arange(shape.shape[0])

        symmetric_map = self.symmetric_map.clone()

        return {'x':shape, 'faces':faces, 'landmarks_id':landmarks_id, 'landmarks_idx':landmarks_idx, 'shape_name':model, 'symmetric_map':symmetric_map}


class SHREC20bDataset(Dataset):

    def __init__(self, in_path, dataset="SHREC20b_lores", transform=None, dtype=torch.float32):
        self.in_path = os.path.join(in_path, dataset, "models")
        self.gt_path = os.path.join(in_path, dataset + "_gts")
        self.models = [os.path.splitext(file)[0] for file in os.listdir(self.in_path)]
        self.transform = transform
        self.dtype = dtype

    def __len__(self):
        return len(self.models)

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.models.index(index)
        model = self.models[index]
        mesh = trimesh.load_mesh(os.path.join(self.in_path, model) + ".obj", process=False)
        faces = torch.from_numpy(mesh.faces).long()
        shape = torch.from_numpy(mesh.vertices).to(self.dtype)
        gt = loadmat(os.path.join(self.gt_path, model) + ".mat")
        landmarks_id = torch.from_numpy(gt['points']).long().squeeze() - 1
        landmarks_idx = torch.from_numpy(gt['verts']).long().squeeze() - 1

        if self.transform:
            shape = self.transform(shape)

        return {'x':shape, 'faces':faces, 'landmarks_id':landmarks_id, 'landmarks_idx':landmarks_idx, 'shape_name':model}


class SMAL_RDataset(Dataset):
    def __init__(self, in_path, dataset="SMAL_r", transform=None, dtype=torch.float32):
        self.in_path = os.path.join(in_path, dataset, "off")
        self.gt_path = os.path.join(in_path, dataset, "corres")
        self.models = [os.path.splitext(file)[0] for file in os.listdir(self.in_path)]
        self.transform = transform
        self.dtype = dtype

    def __len__(self):
        return len(self.models)

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.models.index(index)
        model = self.models[index]
        mesh = trimesh.load_mesh(os.path.join(self.in_path, model) + ".off", process=False)
        faces = torch.from_numpy(mesh.faces).long()
        shape = torch.from_numpy(mesh.vertices).to(self.dtype)
        landmarks_idx = torch.from_numpy(np.loadtxt(os.path.join(self.gt_path, model) + ".vts", dtype=int)).long() - 1
        landmarks_id = torch.arange(landmarks_idx.size(0)).long()

        if self.transform:
            shape = self.transform(shape)

        return {'x':shape, 'faces':faces, 'landmarks_id':landmarks_id, 'landmarks_idx':landmarks_idx, 'shape_name':model}

class GenericPairDataset(Dataset):
    """Wraps two standalone mesh files (not part of a registered benchmark) as a 2-item
    dataset ("A", "B"), so an arbitrary pair of shapes can flow through the same pipeline
    as the bundled datasets. Landmarks are the ones supplied by the caller (e.g. via
    --landmarks-idx-A/--landmarks-idx-B), paired by position: the k-th index of A
    corresponds to the k-th index of B."""

    def __init__(self, shape_a_path, shape_b_path, landmarks_idx_a, landmarks_idx_b, transform=None, dtype=torch.float32):
        assert len(landmarks_idx_a) == len(landmarks_idx_b), "landmarks_idx_a and landmarks_idx_b must have the same length (they are paired by position)"
        self.models = ["A", "B"]
        self.paths = {"A": shape_a_path, "B": shape_b_path}
        self.landmarks_idx = {
            "A": torch.tensor(landmarks_idx_a, dtype=torch.long),
            "B": torch.tensor(landmarks_idx_b, dtype=torch.long),
        }
        self.transform = transform
        self.dtype = dtype

    def __len__(self):
        return len(self.models)

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self.models[index]
        mesh = trimesh.load_mesh(self.paths[index], process=False)
        faces = torch.from_numpy(mesh.faces).long()
        shape = torch.from_numpy(mesh.vertices).to(self.dtype)
        if self.transform:
            shape = self.transform(shape)

        landmarks_idx = self.landmarks_idx[index]
        landmarks_id = torch.arange(landmarks_idx.shape[0])

        return {'x':shape, 'faces':faces, 'landmarks_id':landmarks_id, 'landmarks_idx':landmarks_idx, 'shape_name':index}


class TOPKIDSDataset(Dataset):
    def __init__(self, in_path, dataset="TOPKIDS", transform=None, dtype=torch.float32):
        self.in_path = os.path.join(in_path, dataset, "off")
        self.gt_path = os.path.join(in_path, dataset, "corres")
        self.models = [os.path.splitext(file)[0] for file in os.listdir(self.in_path)]
        self.transform = transform
        self.dtype = dtype

    def __len__(self):
        return len(self.models)

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.models.index(index)
        model = self.models[index]
        mesh = trimesh.load_mesh(os.path.join(self.in_path, model) + ".off", process=False)
        faces = torch.from_numpy(mesh.faces).long()
        shape = torch.from_numpy(mesh.vertices).to(self.dtype)
        if model == 'kid00':
            landmarks_id = torch.arange(shape.shape[0]).long()
        else:
            landmarks_id = torch.from_numpy(np.loadtxt(os.path.join(self.gt_path, f"{model}_ref.vts"), dtype=int)).long() - 1
        landmarks_idx = torch.arange(landmarks_id.shape[0]).long()

        if self.transform:
            shape = self.transform(shape)

        return {'x':shape, 'faces':faces, 'landmarks_id':landmarks_id, 'landmarks_idx':landmarks_idx, 'shape_name':model}
