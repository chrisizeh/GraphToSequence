import os.path as osp
from glob import glob
import random

import tqdm as tqdm
from itertools import chain

import numpy as np

import torch
from torch_geometric.data import Dataset, Data


class RandomGraphDataset(Dataset):

    def __init__(self, root, nodes=5, data_count=10, area=[[0, 100], [0, 100], [0, 100]], transform=None, pre_transform=None, pre_filter=None):
        self.nodes = nodes
        self.data_count = data_count
        self.area = np.array(area)
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return glob(f"{self.raw_dir}/*")

    @property
    def processed_file_names(self):
        return glob(f"{self.processed_dir}/data_*.pt")

    def download(self):
        # data = np.zeros((self.nodes, len(self.area) + 2))
        data = np.random.rand(self.nodes, self.area.shape[0] + 2)
        data[:, :3] = data[:, :3] * (self.area[:, 1] - self.area[:, 0]) + self.area[:, 0]
        data[:, 3:] = data[:, 3:] * 2 * np.pi

        torch.save(data, osp.join(self.raw_dir, f'data.pt'))

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            run = torch.load(raw_path)
            nAllEdges = int(self.nodes * (self.nodes - 1) / 2)
            num = 0

            all_edges = np.zeros((2, nAllEdges))
            for i in range(self.nodes):
                for j in range(i + 1, self.nodes):
                    all_edges[0, num] = i
                    all_edges[1, num] = j
                    num += 1

            for nodes in range(self.data_count):
                numEdges = random.randrange(self.nodes * (self.nodes - 1) / 2)
                edges = random.sample(range(nAllEdges), numEdges)

                data = Data(x=torch.from_numpy(run), num_nodes=self.nodes, edge_index=torch.from_numpy(all_edges[:, edges]))

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
                idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
