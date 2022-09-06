__date__ = "2022/09/03 01:23:54"

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rc('font', size = 16)
mpl.rc('axes', titlesize = 'large', labelsize = 'large')
mpl.rc('xtick', labelsize = 'large')
mpl.rc('ytick', labelsize = 'large')
import mdtraj
import os
from sys import exit
import scipy
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform

prmtop = mdtraj.load_prmtop('./data/cln025.prmtop')
alpha_carbon_atom_indices = []
for atom in prmtop.atoms:
    if atom.name == 'CA':
        alpha_carbon_atom_indices.append(atom.index)
traj = mdtraj.load_dcd('./data/cln025_all_atom.dcd', prmtop, stride = 10)
distances = np.empty((traj.n_frames, traj.n_frames))
for i in range(traj.n_frames):
    distances[i] = mdtraj.rmsd(traj, traj, i, atom_indices = alpha_carbon_atom_indices)
    if (i + 1) % 200 == 0:
        print(i)
        
print('Max pairwise rmsd: %f nm' % np.max(distances))
reduced_distances = squareform(distances, checks=False)

linkage = scipy.cluster.hierarchy.linkage(reduced_distances, method='average')
n_cluster = 50
cluster_label = scipy.cluster.hierarchy.fcluster(linkage, t = n_cluster, criterion = 'maxclust')
cluster_size = []
for i in range(1, n_cluster+1):
    cluster_size.append(np.sum(cluster_label == i))

popular_cluster_label = np.argmax(cluster_size) + 1
flag = cluster_label == popular_cluster_label
traj = traj[flag]

dist = np.mean(distances[flag,:][:,flag], -1)
idx = np.argmin(dist)
ref_traj = traj[idx]
ref_traj.save_pdb('./data/cln025_reference.pdb')
exit()
