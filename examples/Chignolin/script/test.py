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
import torch
import math
import sys
sys.path.append('/home/xqding/my_projects_on_github/PC/')
import PC
import openmm
import openmm.unit as unit
import openmm.app as ommapp
import time

#### convert the all atom trajectory into a coarse-grained one
top_aa = mdtraj.load_prmtop('./data/cln025.prmtop')
traj_aa_list = []
for idx_chunk in range(3):
    traj_aa = mdtraj.load_dcd(f'/home/xqding/projects/MMCD/output/cln025/traj_at_temperature/traj_temperature_chunk_{idx_chunk}_366.14.dcd', top_aa)
    traj_aa_list.append(traj_aa)
traj_aa = mdtraj.join(traj_aa_list)    

#traj_aa.save_dcd('./data/cln025_all_atom.dcd')

alpha_carbon_atom_indices = []
for atom in top_aa.atoms:
    if atom.name == 'CA':
        alpha_carbon_atom_indices.append(atom.index)

top_cg = top_aa.subset(alpha_carbon_atom_indices)
for i in range(top_cg.n_atoms - 1):
    top_cg.add_bond(top_cg.atom(i), top_cg.atom(i+1))
    
traj_cg = traj_aa.atom_slice(alpha_carbon_atom_indices)

pdb = mdtraj.load_pdb('./data/cln025_reference.pdb')
pdb = pdb.atom_slice(alpha_carbon_atom_indices)
ref_traj = mdtraj.Trajectory(pdb.xyz, topology = top_cg)

n_atoms = top_cg.n_atoms

bonded_prm = {
    'bond': {'indices': np.array([[i,i+1] for i in range(n_atoms - 1)])},
    'angle': {'indices': np.array([[i,i+1,i+2] for i in range(n_atoms - 2)])},
    'dihedral': {'indices': np.array([[i,i+1,i+2,i+3] for i in range(n_atoms - 3)])}    
}

bond_cg = mdtraj.compute_distances(traj_cg, bonded_prm['bond']['indices'])
angle_cg = mdtraj.compute_angles(traj_cg, bonded_prm['angle']['indices'])
dihedral_cg = mdtraj.compute_dihedrals(traj_cg, bonded_prm['dihedral']['indices'])

fig = plt.figure(figsize = (6.4*3, 4.8*8))
fig.clf()
idx_plot = 1
for j in range(bond_cg.shape[1]):
    plt.subplot(8,3,idx_plot)
    plt.hist(bond_cg[:,j], bins = 30, density = True, color = 'C0', alpha = 0.5, label = 'All atom')
    plt.legend()
    idx_plot += 1
    
for j in range(angle_cg.shape[1]):
    plt.subplot(8,3,idx_plot)
    plt.hist(angle_cg[:,j], bins = 30, density = True, color = 'C0', alpha = 0.5, label = 'All atom')
    plt.legend()
    idx_plot += 1

for j in range(dihedral_cg.shape[1]):
    plt.subplot(8,3,idx_plot)
    plt.hist(dihedral_cg[:,j], bins = 30, density = True, color = 'C0', alpha = 0.5, label = 'All atom')
    plt.legend()
    idx_plot += 1
    
plt.savefig('./output/test.pdf')

rmsd = mdtraj.rmsd(traj_cg, ref_traj)
fig = plt.figure()
fig.clf()
plt.hist(rmsd, bins = 30, density = True, log = True)
plt.savefig('./output/rmsd_test.pdf')
