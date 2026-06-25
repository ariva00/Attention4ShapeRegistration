import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from meshtorch.transforms import RandomRotateOneOrAllAxis, NormalizeShapeAreaWeighted

from .datasets import FaustDataset, SHREC20bDataset, SMAL_RDataset, TOPKIDSDataset
from .geometry import compute_rematching, compute_shape_info

def get_dataloader(args):
    transform_train = transforms.Compose([
        RandomRotateOneOrAllAxis(360),
        NormalizeShapeAreaWeighted(),
    ])

    if args.dataset == "shrec20":
        data_train = SHREC20bDataset(args.path_data, transform=transform_train)
    elif args.dataset == "faust":
        data_train = FaustDataset(args.path_data, transform=transform_train)
    elif args.dataset == "smal-r":
        data_train = SMAL_RDataset(args.path_data, transform=transform_train)
    elif args.dataset == "topkids":
        data_train = TOPKIDSDataset(args.path_data, transform=transform_train)

    dataloader_train = DataLoader(data_train, batch_size=2 if not args.sparse else None, shuffle=True, drop_last=True if not args.sparse else False)

    return dataloader_train, data_train

def get_couple(dataloader_train, args, iterator = None):
    if iterator is None:
        iterator = iter(dataloader_train)
    if args.sparse:
        item_A = next(iterator)
        item_B = next(iterator)
        if args.flip:
            item_A, item_B = item_B, item_A
        shape_A = item_A["x"].unsqueeze(0).to(args.device)
        shape_B = item_B["x"].unsqueeze(0).to(args.device)
        faces_A = item_A["faces"].unsqueeze(0).to(args.device)
        faces_B = item_B["faces"].unsqueeze(0).to(args.device)
        name_A = item_A["shape_name"]
        name_B = item_B["shape_name"]
        landmarks_idx_A = item_A['landmarks_idx'].unsqueeze(0).to(args.device)
        landmarks_id_A = item_A['landmarks_id'].unsqueeze(0).to(args.device)
        landmarks_idx_B = item_B['landmarks_idx'].unsqueeze(0).to(args.device)
        landmarks_id_B = item_B['landmarks_id'].unsqueeze(0).to(args.device)
        symmetric_map_A = item_A['symmetric_map'].unsqueeze(0).to(args.device) if 'symmetric_map' in item_A.keys() else None
        symmetric_map_B = item_B['symmetric_map'].unsqueeze(0).to(args.device) if 'symmetric_map' in item_B.keys() else None

    else:
        item = next(iterator)
        shapes = item["x"].to(args.device)
        shape_A = shapes[:1, :, :]
        shape_B = shapes[1:, :, :]
        faces_A = item["faces"][:1,:,:].to(args.device)
        faces_B = item["faces"][1:,:,:].to(args.device)
        name_A = item["shape_name"][0]
        name_B = item["shape_name"][1]
        if args.flip:
            shape_A, shape_B = shape_B, shape_A
            faces_A, faces_B = faces_B, faces_A
            name_A, name_B = name_B, name_A
        landmarks_idx_A = item['landmarks_idx'][:1].to(args.device)
        landmarks_id_A = item['landmarks_id'][:1].to(args.device)
        landmarks_idx_B = item['landmarks_idx'][1:].to(args.device)
        landmarks_id_B = item['landmarks_id'][1:].to(args.device)
        symmetric_map_A = item['symmetric_map'][:1].to(args.device) if 'symmetric_map' in item.keys() else None
        symmetric_map_B = item['symmetric_map'][1:].to(args.device) if 'symmetric_map' in item.keys() else None

    if args.rmt:
        shape_A, faces_A, landmarks_idx_A, rmt_A = compute_rematching(shape_A, faces_A, landmarks_idx_A, landmarks_id_A, args)
        shape_B, faces_B, landmarks_idx_B, rmt_B = compute_rematching(shape_B, faces_B, landmarks_idx_B, landmarks_id_B, args)
    else:
        rmt_A = None
        rmt_B = None

    distances_A = compute_shape_info(shape_A, faces_A, args)
    distances_B = compute_shape_info(shape_B, faces_B, args)

    return shape_A, shape_B, faces_A, faces_B, name_A, name_B, landmarks_id_A, landmarks_id_B, landmarks_idx_A, landmarks_idx_B, distances_A, distances_B, symmetric_map_A, symmetric_map_B, rmt_A, rmt_B

def get_shape_by_idx(dataset_train, idx, args):
    item = dataset_train[idx]
    shape = item["x"].unsqueeze(0).to(args.device)
    faces = item["faces"].unsqueeze(0).to(args.device)
    name = item["shape_name"]
    landmarks_idx = item['landmarks_idx'].unsqueeze(0).to(args.device)
    landmarks_id = item['landmarks_id'].unsqueeze(0).to(args.device)

    symmetric_map = item['symmetric_map'].unsqueeze(0).to(args.device) if 'symmetric_map' in item.keys() else None

    if args.rmt:
        shape, faces, landmarks_idx, rmt = compute_rematching(shape, faces, landmarks_idx, landmarks_id, args)
    else:
        rmt = None

    distances = compute_shape_info(shape, faces, args)

    return shape, faces, name, landmarks_id, landmarks_idx, distances, symmetric_map, rmt

def get_shapes(dataloader_train, dataset_train, args):
    if args.couple is None:
        shape_A, shape_B, faces_A, faces_B, name_A, name_B, landmarks_id_A, landmarks_id_B, landmarks_idx_A, landmarks_idx_B, distances_A, distances_B, symmetric_map_A, symmetric_map_B, rmt_A, rmt_B = get_couple(dataloader_train, args)
    else:
        if not args.flip:
            shape_A, faces_A, name_A, landmarks_id_A, landmarks_idx_A, distances_A, symmetric_map_A, rmt_A = get_shape_by_idx(dataset_train, args.couple[0], args)
            shape_B, faces_B, name_B, landmarks_id_B, landmarks_idx_B, distances_B, symmetric_map_B, rmt_B = get_shape_by_idx(dataset_train, args.couple[1], args)
        else:
            shape_A, faces_A, name_A, landmarks_id_A, landmarks_idx_A, distances_A, symmetric_map_A, rmt_A = get_shape_by_idx(dataset_train, args.couple[1], args)
            shape_B, faces_B, name_B, landmarks_id_B, landmarks_idx_B, distances_B, symmetric_map_B, rmt_B = get_shape_by_idx(dataset_train, args.couple[0], args)

    return shape_A, shape_B, faces_A, faces_B, name_A, name_B, landmarks_id_A, landmarks_id_B, landmarks_idx_A, landmarks_idx_B, distances_A, distances_B, symmetric_map_A, symmetric_map_B, rmt_A, rmt_B
