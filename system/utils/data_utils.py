import numpy as np
import os
import torch

data_path = '/home/duong/data'


def read_data(dataset, idx, niid, balance, alpha, is_train=True, num_clients=None):
    if is_train:
        train_data_dir = os.path.join(data_path, dataset, str(num_clients), f"{balance}_{niid}_{alpha}", 'train/')
        # train_data_dir = os.path.join(data_path, dataset, "100", f"{balance}_{niid}_{alpha}", 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join(data_path, dataset, str(num_clients), f"{balance}_{niid}_{alpha}", 'test/')
        # test_data_dir = os.path.join(data_path, dataset, "100", f"{balance}_{niid}_{alpha}", 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, niid, balance, alpha, is_train=True, num_clients=None):
    if num_clients is None:
        raise ValueError("num_clients cannot be None")
    if dataset[:2] == "ag" or dataset[:2] == "SS":
        return read_client_data_text(dataset, idx, niid, balance, alpha, is_train, num_clients)
    elif dataset[:2] == "sh":
        return read_client_data_shakespeare(dataset, idx, niid, balance, alpha, is_train, num_clients)

    if is_train:
        train_data = read_data(dataset, idx, niid, balance, alpha, is_train, num_clients)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, niid, balance, alpha, is_train, num_clients)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def read_client_data_text(dataset, idx, niid, balance, alpha, is_train=True, num_clients=None):
    if is_train:
        train_data = read_data(dataset, idx, niid, balance, alpha, is_train, num_clients)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, niid, balance, alpha, is_train, num_clients)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data


def read_client_data_shakespeare(dataset, idx, niid, balance, alpha, is_train=True, num_clients=None):
    if is_train:
        train_data = read_data(dataset, idx, niid, balance, alpha, is_train, num_clients)
        X_train = torch.Tensor(train_data['x']).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, niid, balance, alpha, is_train, num_clients)
        X_test = torch.Tensor(test_data['x']).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data
