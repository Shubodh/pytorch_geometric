import argparse
import os.path as osp
import csv

import torch
import numpy as np

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

import sys
from pathlib import Path
from torch_geometric.nn import DimeNet#, DimeNetPlusPlus

import torch_geometric.nn as expt_imp
#print(expt_imp.__file__)
#sys.exit()
#sys.path.append(str(Path(__file__).parent / '../torch_geometric/nn/models/'))
##from torch_geometric.nn.models.dimenet import DimeNet
#from dimenet import DimeNet

#parser = argparse.ArgumentParser()
#parser.add_argument('--use_dimenet_plus_plus', action='store_true')
#args = parser.parse_args()

#custom
from torch_geometric.nn import radius_graph
from torch_sparse import SparseTensor

def triplets(edge_index, num_nodes):

    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value,
			 sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji


def debug_code(data, z, pos, batch):
    #print(data.pos)

    edge_index = radius_graph(pos, r=2.0, batch=batch, max_num_neighbors=32)

    ## Understanding radius_graph: START
    #If I reduce r from 5 to 2.0, edge_index.shape and dist.shape reduces from 70000 to 12000. Also edge_index is different from given edge_index, see below, i.e. you are calculating your own edges using z and  pos alone, NOT using input edge_index or even x for that matter.
    #data=DataBatch(x=[4565, 11], edge_index=[2, 9482], edge_attr=[9482, 4], y=[256, 12], pos=[4565, 3], idx=[256], name=[256], z=[4565], batch=[4565], ptr=[257])
    #edge_index.shape=torch.Size([2, 12008])
    #dist.shape=torch.Size([12008])
    ## Understanding radius_graph END



    i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
        edge_index, num_nodes=z.size(0))

    ## Calculate distances.
    print(i.shape, j.shape, idx_i.shape, idx_j.shape, idx_k.shape, idx_kj.shape, idx_ji.shape) # 12008, 12008, rest all 25062 #didn't understand shape of latter, but since it's angle of a triplet, it has to be different than "edge" or dist.shape
    dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
    pos_i = pos[idx_i]
    pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i
    a = (pos_ji * pos_ki).sum(dim=-1)
    b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
    angle = torch.atan2(b, a)

    print(angle)
    print(f"{z=}")
    print(f"{data=}")
    #print(f"{data.has_isolated_nodes()=}")
    #print(f"{data.num_graphs=}")
    #print(f"{data.batch=}")
    print(f"{edge_index.shape=}")
    print(f"{dist.shape=}")
    print(f"{angle.shape=}")
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

paper_results = {'0': ['mu', 0.0286], 
                 '1': ['alpha', 0.0469],
                 '2' : ['epsilon_HOMO', 27.8],
                 '3' : ['epsilon_LUMO', 19.7],
                 '4' : ['Delta epsilon', 34.8],
                 '5' : ['<R^2>', 0.331],
                 '6' : ['ZPVE', 1.29],
                 '7' : ['U_0', 8.02],
                 '8' : ['U', 7.89],
                 '9' : ['H', 8.11],
                 '10' : ['G', 8.98],
                 '11' : ['c_v', 0.0249]
                }

csv_fields = ['property', 'reported result', 'pyg result']
csv_main = []

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

            #debug_code(data, data.z, data.pos, data.batch)

            pred = model(data.z, data.pos, data.batch)

            #print(pred.shape)
            #sys.exit()
        mae = (pred.view(-1) - data.y[:, target]).abs()
        maes.append(mae)

    mae = torch.cat(maes, dim=0)

    # Report meV instead of eV:
    mae = 1000 * mae if target in [2, 3, 4, 6, 7, 8, 9, 10] else mae

    #print(f'Target: {target:02d}, MAE: {mae.mean():.5f} ± {mae.std():.5f}')
    current = paper_results[str(target)]
    print(f'Target: {target:02d}, MAE: {mae.mean():.5f} ± {mae.std():.5f}, paper result: {current[1]}')
    mae_final = str("{:.5f}".format(mae.mean()))  + '+-' + str("{:.5f}".format(mae.std()))

    csv_main_row = [paper_results[str(target)][0], str(paper_results[str(target)][1]), mae_final]
    csv_main.append(csv_main_row) #['property', 'reported result', 'pyg result']
    print(csv_main_row)

print(csv_fields)
print(csv_main)
filename = "saved_results.csv"
with open(filename, 'w') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerow(csv_fields) 
    csvwriter.writerows(csv_main)
