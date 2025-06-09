import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import os

import datautil.actdata.util as actutil
from datautil.util import combindataset, subdataset
import datautil.actdata.cross_people as cross_people

task_act = {'cross_people': cross_people}


def get_dataloader(args, tr, val, tar):
    train_loader = DataLoader(dataset=tr, batch_size=args.batch_size,
                              num_workers=args.N_WORKERS, drop_last=False, shuffle=True)
    train_loader_noshuffle = DataLoader(
        dataset=tr, batch_size=args.batch_size, num_workers=args.N_WORKERS, drop_last=False, shuffle=False)
    valid_loader = DataLoader(dataset=val, batch_size=args.batch_size,
                              num_workers=args.N_WORKERS, drop_last=False, shuffle=False)
    target_loader = DataLoader(dataset=tar, batch_size=args.batch_size,
                               num_workers=args.N_WORKERS, drop_last=False, shuffle=False)
    return train_loader, train_loader_noshuffle, valid_loader, target_loader


def get_act_dataloader(args):
    if args.dataset == "uci_har":
        return get_uci_har_dataloader(args)

    source_datasetlist = []
    target_datalist = []
    pcross_act = task_act[args.task]

    tmpp = args.act_people[args.dataset]
    args.domain_num = len(tmpp)
    for i, item in enumerate(tmpp):
        tdata = pcross_act.ActList(
            args, args.dataset, args.data_dir, item, i, transform=actutil.act_train())
        if i in args.test_envs:
            target_datalist.append(tdata)
        else:
            source_datasetlist.append(tdata)
            if len(tdata) / args.batch_size < args.steps_per_epoch:
                args.steps_per_epoch = len(tdata) / args.batch_size

    rate = 0.2
    args.steps_per_epoch = int(args.steps_per_epoch * (1 - rate))
    tdata = combindataset(args, source_datasetlist)
    l = len(tdata.labels)
    indexall = np.arange(l)
    np.random.seed(args.seed)
    np.random.shuffle(indexall)
    ted = int(l * rate)
    indextr, indexval = indexall[ted:], indexall[:ted]
    tr = subdataset(args, tdata, indextr)
    val = subdataset(args, tdata, indexval)
    targetdata = combindataset(args, target_datalist)
    return get_dataloader(args, tr, val, targetdata) + (tr, val, targetdata)


# -----------------------------
# ✅ UCI-HAR Specific Loader
# -----------------------------
def get_uci_har_dataloader(args):
    print("[INFO] Using UCI HAR dataset loader")

    # Load raw features and labels
    X_train, y_train, s_train = load_group(os.path.join(args.data_dir, 'train'), args)
    X_test, y_test, s_test = load_group(os.path.join(args.data_dir, 'test'), args)

    # Remap subject IDs to domain indices
    all_subjects = sorted(set(s_train.tolist() + s_test.tolist()))
    sid2domain = {sid: i for i, sid in enumerate(all_subjects)}

    s_train = torch.tensor([sid2domain[int(s)] for s in s_train], dtype=torch.long)
    s_test = torch.tensor([sid2domain[int(s)] for s in s_test], dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train, s_train, torch.zeros_like(s_train), s_train)
    test_dataset = TensorDataset(X_test, y_test, s_test, torch.zeros_like(s_test), s_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.N_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.N_WORKERS)

    return train_loader, train_loader, test_loader, test_loader, train_dataset, test_dataset, test_dataset


def load_group(folder, args):
    split = 'train' if 'train' in folder else 'test'
    X = fuse_signals(args, split, folder=folder)

    y = load_file(os.path.join(folder, f'y_{split}.txt')).astype(int).flatten()
    if y.min() == 1:
        y -= 1  # convert 1-indexed to 0-indexed
    s = load_file(os.path.join(folder, f'subject_{split}.txt')).astype(int).flatten()

    return X, torch.tensor(y, dtype=torch.long), torch.tensor(s, dtype=torch.long)


def fuse_signals(args, split, folder=None):
    if folder is None:
        folder = os.path.join(args.data_dir, split)

    # Main feature vector: (N, 561)
    X_flat = load_file(os.path.join(folder, f'X_{split}.txt'))
    X_flat = torch.tensor(X_flat, dtype=torch.float32).unsqueeze(2).unsqueeze(2)  # (N, 561, 1, 1)

    # Inertial Signals: each (N, 128)
    inertial_signals = []
    for name in [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z']:
        fpath = os.path.join(folder, 'Inertial Signals', f"{name}_{split}.txt")
        sig = load_file(fpath).reshape(-1, 128, 1)
        inertial_signals.append(sig)

    # Normalize and reshape inertial data
    X_inertial = np.concatenate(inertial_signals, axis=2)  # (N, 128, 9)
    X_inertial = (X_inertial - X_inertial.mean()) / (X_inertial.std() + 1e-8)
    X_inertial = torch.tensor(X_inertial.transpose(0, 2, 1), dtype=torch.float32).unsqueeze(2)  # (N, 9, 1, 128)

    # Expand flat to match time steps (N, 561, 1, 128)
    X_flat = X_flat.expand(-1, -1, 1, 128)

    # Final fused shape: (N, 570, 1, 128)
    return torch.cat([X_flat, X_inertial], dim=1)


def load_file(filepath):
    return np.loadtxt(filepath)
