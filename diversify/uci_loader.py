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
    train_loader, train_loader_noshuffle, valid_loader, target_loader = get_dataloader(
        args, tr, val, targetdata)
    return train_loader, train_loader_noshuffle, valid_loader, target_loader, tr, val, targetdata

def get_uci_har_dataloader(args):
    X_train, y_train, s_train = load_group(os.path.join(args.data_dir, 'train'))
    X_test, y_test, s_test = load_group(os.path.join(args.data_dir, 'test'))

    # normalize
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_test = (X_test - X_test.mean()) / X_test.std()

    # reshape to (N, C, 1, T) for CNN
    N_train, T = X_train.shape
    N_test, _ = X_test.shape
    X_train = X_train.view(N_train, T, 1).permute(0, 2, 1).unsqueeze(2)
    X_test = X_test.view(N_test, T, 1).permute(0, 2, 1).unsqueeze(2)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train, s_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test, s_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.N_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.N_WORKERS)

    # for compatibility
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

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y.flatten(), dtype=torch.long), torch.tensor(subjects.flatten(), dtype=torch.long)


    X_train, y_train = load_group(os.path.join(args.data_dir, 'train'))
    X_test, y_test = load_group(os.path.join(args.data_dir, 'test'))

    n_features = 9
    n_timesteps = 128
    X_train = X_train.reshape(-1, n_timesteps, n_features).transpose(0, 2, 1)
    X_test = X_test.reshape(-1, n_timesteps, n_features).transpose(0, 2, 1)

    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    return train_loader, train_loader, test_loader, test_loader, None, None, None
