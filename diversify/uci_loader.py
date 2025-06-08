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
            if len(tdata)/args.batch_size < args.steps_per_epoch:
                args.steps_per_epoch = len(tdata)/args.batch_size
    rate = 0.2
    args.steps_per_epoch = int(args.steps_per_epoch*(1-rate))
    tdata = combindataset(args, source_datasetlist)
    l = len(tdata.labels)
    indexall = np.arange(l)
    np.random.seed(args.seed)
    np.random.shuffle(indexall)
    ted = int(l*rate)
    indextr, indexval = indexall[ted:], indexall[:ted]
    tr = subdataset(args, tdata, indextr)
    val = subdataset(args, tdata, indexval)
    targetdata = combindataset(args, target_datalist)
    return get_dataloader(args, tr, val, targetdata) + (tr, val, targetdata)


# -----------------------------
# ✅ UCI-HAR Specific Loader
# -----------------------------
def get_uci_har_dataloader(args):
    X_train, y_train, s_train = load_group(os.path.join(args.data_dir, 'train'))
    X_test, y_test, s_test = load_group(os.path.join(args.data_dir, 'test'))

    # Normalize globally
    mean = X_train.mean()
    std = X_train.std()
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    # Reshape to (N, C=1, H=1, W=561)
    X_train = X_train.view(X_train.size(0), 1, 1, -1)
    X_test = X_test.view(X_test.size(0), 1, 1, -1)

    train_dataset = TensorDataset(X_train, y_train, s_train, torch.zeros_like(s_train), s_train)
    test_dataset = TensorDataset(X_test, y_test, s_test, torch.zeros_like(s_test), s_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.N_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.N_WORKERS)

    return train_loader, train_loader, test_loader, test_loader, train_dataset, test_dataset, test_dataset


def load_group(folder):
    if 'train' in folder:
        X = load_file(os.path.join(folder, 'X_train.txt'))
        y = load_file(os.path.join(folder, 'y_train.txt'))
        subjects = load_file(os.path.join(folder, 'subject_train.txt'))
    else:
        X = load_file(os.path.join(folder, 'X_test.txt'))
        y = load_file(os.path.join(folder, 'y_test.txt'))
        subjects = load_file(os.path.join(folder, 'subject_test.txt'))

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y.flatten() - 1, dtype=torch.long),       # Make classes 0-based
        torch.tensor(subjects.flatten(), dtype=torch.long)
    )


def load_file(filepath):
    return np.loadtxt(filepath)
