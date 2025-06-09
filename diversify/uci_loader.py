import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import os

import datautil.actdata.util as actutil
from datautil.util import combindataset, subdataset
import datautil.actdata.cross_people as cross_people

task_act = {'cross_people': cross_people}


def get_dataloader(args, tr, val, tar):
    train_loader = DataLoader(tr, batch_size=args.batch_size,
                              num_workers=args.N_WORKERS, shuffle=True, drop_last=False)
    train_loader_noshuffle = DataLoader(tr, batch_size=args.batch_size,
                                        num_workers=args.N_WORKERS, shuffle=False, drop_last=False)
    valid_loader = DataLoader(val, batch_size=args.batch_size,
                              num_workers=args.N_WORKERS, shuffle=False, drop_last=False)
    target_loader = DataLoader(tar, batch_size=args.batch_size,
                               num_workers=args.N_WORKERS, shuffle=False, drop_last=False)
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
        tdata = pcross_act.ActList(args, args.dataset, args.data_dir, item, i, transform=actutil.act_train())
        if i in args.test_envs:
            target_datalist.append(tdata)
        else:
            source_datasetlist.append(tdata)
            if len(tdata)/args.batch_size < args.steps_per_epoch:
                args.steps_per_epoch = len(tdata)/args.batch_size

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


def get_uci_har_dataloader(args):
    print("[INFO] Using UCI HAR dataset loader")

    X_train, y_train, s_train = load_group(os.path.join(args.data_dir, 'train'), args)
    X_test, y_test, s_test = load_group(os.path.join(args.data_dir, 'test'), args)

    # Remap subject IDs to domain indices
    all_subjects = sorted(set(s_train.tolist() + s_test.tolist()))
    sid_to_domain = {sid: idx for idx, sid in enumerate(all_subjects)}
    s_train = torch.tensor([sid_to_domain[int(s)] for s in s_train], dtype=torch.long)
    s_test = torch.tensor([sid_to_domain[int(s)] for s in s_test], dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train, s_train, torch.zeros_like(s_train), s_train)
    test_dataset = TensorDataset(X_test, y_test, s_test, torch.zeros_like(s_test), s_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.N_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.N_WORKERS)

    return train_loader, train_loader, test_loader, test_loader, train_dataset, test_dataset, test_dataset


def load_group(folder, args):
    split = 'train' if 'train' in folder else 'test'
    X = fuse_signals(args, split, folder=folder)

    y = load_file(os.path.join(folder, f'y_{split}.txt')).astype(int).flatten()
    y = y - 1 if y.min() == 1 else y  # Ensure 0-indexed activity labels

    s = load_file(os.path.join(folder, f'subject_{split}.txt')).astype(int).flatten()
    return X, torch.tensor(y, dtype=torch.long), torch.tensor(s, dtype=torch.long)


def fuse_signals(args, split, folder):
    # X.txt (flattened features)
    X_flat = load_file(os.path.join(folder, f'X_{split}.txt'))  # (N, 561)
    X_flat = torch.tensor(X_flat, dtype=torch.float32).unsqueeze(2).expand(-1, -1, 128).unsqueeze(2)  # (N, 561, 1, 128)

    # Inertial signals (9 channels)
    signals = []
    for name in ['body_acc_x', 'body_acc_y', 'body_acc_z',
                 'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
                 'total_acc_x', 'total_acc_y', 'total_acc_z']:
        path = os.path.join(folder, 'Inertial Signals', f"{name}_{split}.txt")
        data = load_file(path).reshape(-1, 128, 1)  # (N, 128, 1)
        signals.append(data)

    X_inertial = np.concatenate(signals, axis=2)  # (N, 128, 9)
    X_inertial = (X_inertial - X_inertial.mean()) / (X_inertial.std() + 1e-8)
    X_inertial = torch.tensor(X_inertial.transpose(0, 2, 1), dtype=torch.float32).unsqueeze(2)  # (N, 9, 1, 128)

    return torch.cat([X_flat, X_inertial], dim=1)  # (N, 570, 1, 128)


def load_file(filepath):
    return np.loadtxt(filepath)
