import argparse
import os.path as osp

import torch

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DimeNet#, DimeNetPlusPlus

import sys
#parser = argparse.ArgumentParser()
#parser.add_argument('--use_dimenet_plus_plus', action='store_true')
#args = parser.parse_args()

from torch_geometric.nn import radius_graph

def debug_code(data, z, pos, batch):
    #print(data)
    #print(data.pos)

    edge_index = radius_graph(pos, r=5.0, batch=batch, max_num_neighbors=32)

    print(edge_index.shape)

    i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
        edge_index, num_nodes=z.size(0))

    ## Calculate distances.
    dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
    print("hi, the end")
    sys.exit()



#Model = DimeNetPlusPlus if args.use_dimenet_plus_plus else DimeNet
Model = DimeNet#PlusPlus if args.use_dimenet_plus_plus else DimeNet

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
dataset = QM9(path)

# DimeNet uses the atomization energy for targets U0, U, H, and G, i.e.:
# 7 -> 12, 8 -> 13, 9 -> 14, 10 -> 15
idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11]) #12 no.
dataset.data.y = dataset.data.y[:, idx]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for target in range(12):
    # Skip target \delta\epsilon, since it can be computed via
    # \epsilon_{LUMO} - \epsilon_{HOMO}:
    if target == 4:
        continue

    model, datasets = Model.from_qm9_pretrained(path, dataset, target)
    train_dataset, val_dataset, test_dataset = datasets

    test_dd = test_dataset.data
    #print(test_dd.x.shape, test_dd.y.shape, test_dd.z.shape)
    #print(test_dd.z.shape, test_dd.pos.shape)#, test_dd.batch.shape)
    #sys.exit()

    model = model.to(device)
    loader = DataLoader(test_dataset, batch_size=256)

    maes = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():

            debug_code(data, data.z, data.pos, data.batch)

            pred = model(data.z, data.pos, data.batch)
        mae = (pred.view(-1) - data.y[:, target]).abs()
        maes.append(mae)

    mae = torch.cat(maes, dim=0)

    # Report meV instead of eV:
    mae = 1000 * mae if target in [2, 3, 4, 6, 7, 8, 9, 10] else mae

    print(f'Target: {target:02d}, MAE: {mae.mean():.5f} ± {mae.std():.5f}')
