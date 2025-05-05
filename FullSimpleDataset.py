import os.path as osp
from glob import glob
import random

import tqdm as tqdm
from itertools import chain

import numpy as np

import torch
from torch_geometric.data import Dataset, Data

from GraphToSequence import graphToSequence


class FullSimpleGraph(Dataset):
    def __init__(self, root, nodes=5, area=[[0, 100], [0, 100], [0, 100]], test=False, transform=None, pre_transform=None, pre_filter=None):
        self.nodes = nodes
        self.area = np.array(area)
        self.test = test
        self.test_idx = [45, 1, 87, 43, 10, 47,  12, 2, 105]
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return glob(f"{self.raw_dir}/*")

    @property
    def processed_file_names(self):
        return glob(f"{self.processed_dir}/data_*.pt")

    def download(self):
        nAllEdges = int(self.nodes * (self.nodes - 1) / 2)
        all_edges = np.zeros((2, nAllEdges), dtype=np.int64)
        num = 0
        for i in range(1, self.nodes+1):
            for j in range(i + 1, self.nodes+1):
                all_edges[0, num] = i
                all_edges[1, num] = j
                num += 1

        i = 0
        for x in range(nAllEdges):
            for y in range(x+1, nAllEdges):
                for z in range(y+1, nAllEdges):
                    edges = np.array([[all_edges[0, x], all_edges[0, y], all_edges[0, z]], [all_edges[1, x], all_edges[1, y], all_edges[1, z]]])

                    torch.save(edges, osp.join(self.raw_dir, f'edge_{i}.pt'))
                    i += 1

    def process(self):
        nodes = np.arange(0, self.nodes, dtype=int)
        idx = 0
        for i, raw_path in enumerate(self.raw_paths):
            edges = torch.load(raw_path)

            if (not self.test and i in self.test_idx):
                continue
            elif (self.test and i not in self.test_idx):
                continue

            data = Data(x=torch.from_numpy(nodes), num_nodes=self.nodes, edge_index=torch.from_numpy(edges), y=graphToSequence(edges))

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
