import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mdtraj
import os
import scipy
import scipy.optimize as optimize
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform
import torch
import math
import openmm
import openmm.unit as unit
import openmm.app as ommapp
import time
from sys import exit
import pccg

#### convert the all atom trajectory into a coarse-grained one
top_aa = mdtraj.load_prmtop('./data/cln025_all_atom.prmtop')
traj_aa = mdtraj.load_dcd('./data/cln025_all_atom.dcd', top_aa, stride = 10)

alpha_carbon_atom_indices = []
for atom in top_aa.atoms:
    if atom.name == 'CA':
        alpha_carbon_atom_indices.append(atom.index)
traj_cg = traj_aa.atom_slice(alpha_carbon_atom_indices)

os.makedirs('./output/', exist_ok = True)
traj_cg.save_dcd('./output/cln025_cg.dcd')

# top_cg = top_aa.subset(alpha_carbon_atom_indices)
# for i in range(top_cg.n_atoms - 1):
#     top_cg.add_bond(top_cg.atom(i), top_cg.atom(i+1))

top_cg = mdtraj.load_psf('./data/cln025_cg.psf')
traj_cg = mdtraj.load_dcd('./output/cln025_cg.dcd', top_cg)

ref_pdb = mdtraj.load_pdb('./data/cln025_reference.pdb')
ref_pdb = ref_pdb.atom_slice(alpha_carbon_atom_indices)
ref_traj = mdtraj.Trajectory(ref_pdb.xyz, topology = top_cg)
rmsd = mdtraj.rmsd(traj_cg, ref_traj)

fig = plt.figure()
fig.clf()
plt.hist(rmsd, bins = 30, density = True, range = (0, 0.8), color = 'C1', label = 'All atom')
plt.legend()
plt.xlabel('RMSD (nm)')
plt.ylabel('Probablity density')
plt.tight_layout()
plt.savefig('./output/rmsd_hist_all_atom.png')
plt.close()

n_atoms = top_cg.n_atoms
bonded_terms = {
    'bond': {'indices': np.array([[i,i+1] for i in range(n_atoms - 1)])},
    'angle': {'indices': np.array([[i,i+1,i+2] for i in range(n_atoms - 2)])},
    'dihedral': {'indices': np.array([[i,i+1,i+2,i+3] for i in range(n_atoms - 3)])}    
}

#### fit parameters for bonds

bonded_terms['bond']['b0'] = []
bonded_terms['bond']['kb'] = []
for i in range(bonded_terms['bond']['indices'].shape[0]):
    atom_pair = bonded_terms['bond']['indices'][i]
    dist = mdtraj.compute_distances(traj_cg, atom_pairs = [atom_pair])
    bonded_terms['bond']['b0'].append(dist.mean())
    bonded_terms['bond']['kb'].append(1./dist.var())

#### fit parameters for angles    
angle_knots = torch.linspace(0, math.pi, 10)[1:-1]
angle_boundary_knots = torch.tensor([0.0, math.pi])

bonded_terms['angle']['grid'] = []
bonded_terms['angle']['basis_grid'] = []
bonded_terms['angle']['theta'] = []
bonded_terms['angle']['energy_grid'] = []

for i in range(bonded_terms['angle']['indices'].shape[0]):
    angle_atom_indices = bonded_terms['angle']['indices'][i]
    angle_data = mdtraj.compute_angles(traj_cg, [angle_atom_indices])
    angle_data = np.squeeze(angle_data).astype(np.float64)
    angle_data = torch.from_numpy(angle_data)
    angle_data.clamp_(0, math.pi)
    
    angle_noise = torch.rand(len(angle_data))*math.pi

    basis_data = pccg.spline.bs(angle_data, angle_knots, angle_boundary_knots)
    basis_noise = pccg.spline.bs(angle_noise, angle_knots, angle_boundary_knots)

    log_q_data = torch.ones_like(angle_data)*math.log(1./math.pi)
    log_q_noise = torch.ones_like(angle_noise)*math.log(1./math.pi)    

    theta, dF = pccg.NCE(log_q_noise, log_q_data,
                       basis_noise, basis_data,
                       verbose = False)
    
    angle_grid = torch.linspace(0, math.pi, 100)
    basis_grid = pccg.spline.bs(angle_grid, angle_knots, angle_boundary_knots)
    energy_grid = torch.matmul(basis_grid, theta)

    bonded_terms['angle']['grid'].append(angle_grid)
    bonded_terms['angle']['basis_grid'].append(basis_grid)
    bonded_terms['angle']['theta'].append(theta)        
    bonded_terms['angle']['energy_grid'].append(energy_grid)
    
#### fit parameters for dihedrals    
dihedral_knots = torch.linspace(-math.pi, math.pi, 12)[1:-1]
dihedral_boundary_knots = torch.tensor([-math.pi, math.pi])    

bonded_terms['dihedral']['grid'] = []
bonded_terms['dihedral']['basis_grid'] = []
bonded_terms['dihedral']['theta'] = []
bonded_terms['dihedral']['energy_grid'] = []
for i in range(bonded_terms['dihedral']['indices'].shape[0]):
    dihedral_atom_indices = bonded_terms['dihedral']['indices'][i]
    dihedral_data = mdtraj.compute_dihedrals(traj_cg, [dihedral_atom_indices])
    dihedral_data = np.squeeze(dihedral_data).astype(np.float64)
    dihedral_data = torch.from_numpy(dihedral_data)
    dihedral_data.clamp_(-math.pi, math.pi)    

    dihedral_noise = torch.distributions.Uniform(-math.pi, math.pi).sample((len(dihedral_data),))    

    basis_data = pccg.spline.pbs(dihedral_data, dihedral_knots, dihedral_boundary_knots)
    basis_noise = pccg.spline.pbs(dihedral_noise, dihedral_knots, dihedral_boundary_knots)

    log_q_data = torch.ones_like(dihedral_data)*math.log(1./(2*math.pi))
    log_q_noise = torch.ones_like(dihedral_noise)*math.log(1./(2*math.pi))    

    theta, dF = pccg.NCE(log_q_noise, log_q_data,
                       basis_noise, basis_data,
                       verbose = False)
    
    dihedral_grid = torch.linspace(-math.pi, math.pi, 100)
    basis_grid = pccg.spline.pbs(dihedral_grid, dihedral_knots, dihedral_boundary_knots)
    energy_grid = torch.matmul(basis_grid, theta)

    bonded_terms['dihedral']['grid'].append(dihedral_grid)
    bonded_terms['dihedral']['basis_grid'].append(basis_grid)
    bonded_terms['dihedral']['theta'].append(theta)
    bonded_terms['dihedral']['energy_grid'].append(energy_grid)

#### make an openmm system with bonded parameters
#### add particles
aa_mass = {'ALA': 71.08, 'ARG': 156.2, 'ASN': 114.1, 'ASP': 115.1,
           'CYS': 103.1, 'GLN': 128.1, 'GLU': 129.1, 'GLY': 57.05,
           'HIS': 137.1, 'ILE': 113.2, 'LEU': 113.2, 'LYS': 128.2,
           'MET': 131.2, 'PHE': 147.2, 'PRO': 97.12, 'SER': 87.08,
           'THR': 101.1, 'TRP': 186.2, 'TYR': 163.2, 'VAL': 99.07}

sys_im = openmm.System()
for res in top_cg.residues:
    print(res.name, aa_mass[res.name])
    sys_im.addParticle(aa_mass[res.name])

k = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
T = 360.47 * unit.kelvin
kT = (k*T).value_in_unit(unit.kilojoule_per_mole)
    
#### add bond force
bond_force = openmm.CustomBondForce("(0.5*kb*(r - b0)^2 + alpha*log(r))*kT")
bond_force.addGlobalParameter('kT', kT)
bond_force.addPerBondParameter('b0')
bond_force.addPerBondParameter('kb')
bond_force.addPerBondParameter('alpha')
for k in range(bonded_terms['bond']['indices'].shape[0]):
    p1,p2 = bonded_terms['bond']['indices'][k]
    b0 = bonded_terms['bond']['b0'][k]
    kb = bonded_terms['bond']['kb'][k]
    if (p1, p2) in [(0, 1), (1, 0)]:
        alpha = 0
    elif (p1, p2) in [(1, 2), (2, 1)]:
        alpha = 1
    else:
        alpha = 2
    bond_force.addBond(p1,p2, [b0, kb, alpha])
bond_force.setForceGroup(0)
sys_im.addForce(bond_force)

#### add angle force
ua = torch.stack(bonded_terms['angle']['energy_grid']).numpy()
func = openmm.Continuous2DFunction(xsize = ua.shape[0],
                                   ysize = ua.shape[1],
                                   values = ua.reshape(-1, order = 'F'),
                                   xmin = 0.0, xmax = float(ua.shape[0] - 1),
                                   ymin = 0.0, ymax = math.pi,
                                   periodic = False)
angle_force = openmm.CustomCompoundBondForce(
    3,
    f"(ua(idx, angle(p1, p2, p3)) + alpha*log(sin(pi - angle(p1, p2, p3))) )*kT"
)
angle_force.addGlobalParameter('pi', math.pi)
angle_force.addGlobalParameter('kT', kT)
angle_force.addTabulatedFunction("ua", func)
angle_force.addPerBondParameter('idx')
angle_force.addPerBondParameter('alpha')

for k in range(bonded_terms['angle']['indices'].shape[0]):
    p1, p2, p3 = bonded_terms['angle']['indices'][k]
    if (p1, p2, p3) == (0, 1, 2):
        alpha = 0.
    else:
        alpha = 1.
    angle_force.addBond([p1, p2, p3], [float(k), alpha])
angle_force.setForceGroup(0)
sys_im.addForce(angle_force)

#### add dihedral force
ud = torch.stack(bonded_terms['dihedral']['energy_grid']).numpy()
ud = np.concatenate([ud, ud[[0]]])
tmp = (ud[:,0] + ud[:,-1])/2
ud[:,0] = tmp
ud[:,-1] = tmp
func = openmm.Continuous2DFunction(xsize = ud.shape[0],
                                   ysize = ud.shape[1],
                                   values = ud.reshape(-1, order = 'F'),
                                   xmin = 0.0, xmax = float(ud.shape[0] - 1),
                                   ymin = -math.pi, ymax = math.pi,
                                   periodic = True)
dihedral_force = openmm.CustomCompoundBondForce(
    4,
    f"ud(idx, dihedral(p1, p2, p3, p4))*kT"
)
dihedral_force.addGlobalParameter('kT', kT)
dihedral_force.addTabulatedFunction("ud", func)
dihedral_force.addPerBondParameter('idx')

for k in range(bonded_terms['dihedral']['indices'].shape[0]):
    p1, p2, p3, p4 = bonded_terms['dihedral']['indices'][k]
    dihedral_force.addBond([p1, p2, p3, p4], [float(k)])
dihedral_force.setForceGroup(0)
sys_im.addForce(dihedral_force)

sys_im.addForce(openmm.CMMotionRemover())

#### run simulation with the system that only contains
#### bonded interaction energy terms
fricCoef = 1./unit.picoseconds
stepsize = 5. * unit.femtoseconds
integrator = openmm.LangevinMiddleIntegrator(T, fricCoef, stepsize)

pdb = mdtraj.load_pdb('./data/cln025_reference.pdb')
pdb = pdb.atom_slice(alpha_carbon_atom_indices)
xyz_init = pdb.xyz[0]

platform = openmm.Platform.getPlatformByName('Reference')
context = openmm.Context(sys_im, integrator, platform)
context.setPositions(xyz_init)

os.makedirs(f"./output/", exist_ok = True)
file_handle = open(f"./output/traj_im.dcd", 'wb')
dcd_file = ommapp.DCDFile(file_handle, top_cg.to_openmm(), dt = 200*unit.femtoseconds)

num_frames = len(traj_aa)
for i in range(num_frames):
    integrator.step(100)
    state = context.getState(getPositions = True)
    pos = state.getPositions()
    dcd_file.writeModel(pos)
    if (i + 1) % 1000 == 0:
        print(i, flush = True)
file_handle.close()

#### compare traj_im and traj_cg
bond_cg = mdtraj.compute_distances(traj_cg, bonded_terms['bond']['indices'])
angle_cg = mdtraj.compute_angles(traj_cg, bonded_terms['angle']['indices'])
dihedral_cg = mdtraj.compute_dihedrals(traj_cg, bonded_terms['dihedral']['indices'])

traj_im = mdtraj.load_dcd('./output/traj_im.dcd', top_cg)
bond_im = mdtraj.compute_distances(traj_im, bonded_terms['bond']['indices'])
angle_im = mdtraj.compute_angles(traj_im, bonded_terms['angle']['indices'])
dihedral_im = mdtraj.compute_dihedrals(traj_im, bonded_terms['dihedral']['indices'])

fig = plt.figure(figsize = (6.4*4, 4.8*6))
fig.clf()
idx_plot = 1
for j in range(bond_cg.shape[1]):
    plt.subplot(6,4,idx_plot)
    bond_min = min(bond_cg[:,j].min(), bond_im[:,j].min())
    bond_max = max(bond_cg[:,j].max(), bond_im[:,j].max())    
    plt.hist(bond_cg[:,j], bins = 30, range = (bond_min, bond_max),
             density = True, color = 'C0', alpha = 0.5, label = 'All atom')
    plt.hist(bond_im[:,j], bins = 30, range = (bond_min, bond_max),
             density = True, color = 'C1', alpha = 0.5, label = 'CG_im')
    plt.xlabel('Bond length (nm)')
    plt.ylabel('Probability density')    
    plt.title(f'Bond {j}-{j+1}')
    plt.legend()
    plt.tight_layout()
    idx_plot += 1
    
for j in range(angle_cg.shape[1]):
    plt.subplot(6,4,idx_plot)
    plt.hist(angle_cg[:,j], bins = 30, range = (0, math.pi),
             density = True, color = 'C0', alpha = 0.5, label = 'All atom')
    plt.hist(angle_im[:,j], bins = 30, range = (0, math.pi),
             density = True, color = 'C1', alpha = 0.5, label = 'CG_im')
    plt.title(f'Angle {j}-{j+1}-{j+2}')
    plt.xlabel('Angle (radian)')
    plt.ylabel('Probability density')        
    plt.legend()
    plt.tight_layout()    
    idx_plot += 1

for j in range(dihedral_cg.shape[1]):
    plt.subplot(6,4,idx_plot)
    plt.hist(dihedral_cg[:,j], bins = 30, range = (-math.pi, math.pi),
             density = True, color = 'C0', alpha = 0.5, label = 'All atom')
    plt.hist(dihedral_im[:,j], bins = 30, range = (-math.pi, math.pi),
             density = True, color = 'C1', alpha = 0.5, label = 'CG_im')
    plt.title(f'Dihedral {j}-{j+1}-{j+2}-{j+3}')
    plt.xlabel('Dihedral (radian)')
    plt.ylabel('Probability density')            
    plt.legend()
    plt.tight_layout()    
    idx_plot += 1
    
plt.savefig('./output/bad_hist_im.png')
plt.close()

rmsd_data = mdtraj.rmsd(traj_cg, ref_traj)
rmsd_noise = mdtraj.rmsd(traj_im, ref_traj)

fig = plt.figure()
fig.clf()
plt.hist(rmsd_data, bins = 30, density = True, range = (0, 0.8), color = 'C1', label = 'All atom', alpha = 0.5)
plt.hist(rmsd_noise, bins = 30, density = True, range = (0, 0.8), color = 'C0', label = 'CG_im', alpha = 0.5)
plt.legend()
plt.xlabel('RMSD (nm)')
plt.ylabel('Probablity density')
plt.tight_layout()
plt.savefig('./output/rmsd_hist_aa_vs_im.png')
plt.close()

exit()

traj = mdtraj.join([traj_cg, traj_im])
n_data = traj_cg.n_frames
n_noise = traj_im.n_frames
target = torch.cat([torch.ones(n_data),
                    torch.zeros(n_noise)])

#### compute energies on bonds
bond_energy = np.zeros(traj.n_frames)
for i in range(bonded_terms['bond']['indices'].shape[0]):
    atom_pair = bonded_terms['bond']['indices'][i]
    dist = mdtraj.compute_distances(traj, atom_pairs = [atom_pair])
    dist = np.squeeze(dist)
    b0 = bonded_terms['bond']['b0'][i]
    kb = bonded_terms['bond']['kb'][i]
    bond_energy += 0.5*kb*(dist - b0)**2
bond_energy = torch.from_numpy(bond_energy)
    
#### compute basis for angles and dihedrals
bonded_terms['angle']['basis'] = []
for i in range(bonded_terms['angle']['indices'].shape[0]):
    angle_atom_indices = bonded_terms['angle']['indices'][i]
    angle = mdtraj.compute_angles(traj, [angle_atom_indices])
    angle = np.squeeze(angle).astype(np.float64)
    angle = torch.from_numpy(angle)
    angle.clamp_(0, math.pi)

    basis = pccg.spline.bs(angle, angle_knots, angle_boundary_knots)
    bonded_terms['angle']['basis'].append(basis)
    

bonded_terms['dihedral']['basis'] = []
for i in range(bonded_terms['dihedral']['indices'].shape[0]):
    dihedral_atom_indices = bonded_terms['dihedral']['indices'][i]
    dihedral = mdtraj.compute_dihedrals(traj, [dihedral_atom_indices])
    dihedral = np.squeeze(dihedral).astype(np.float64)
    dihedral = torch.from_numpy(dihedral)
    dihedral.clamp_(-math.pi, math.pi)

    basis = pccg.spline.pbs(dihedral, dihedral_knots, dihedral_boundary_knots)
    bonded_terms['dihedral']['basis'].append(basis)

#### compute basis for nonbonded interactions
nonbonded_atom_indices = np.array([(i,j) for i in range(n_atoms) for j in range(i+4, n_atoms)])
nonbonded_terms = {
    'indices': nonbonded_atom_indices,
    'r_rep_on': np.ones(nonbonded_atom_indices.shape[0]),
    'r_off': np.ones(nonbonded_atom_indices.shape[0]),
    'basis': [],
    'basis_grid': [],
    'theta':[],
    'omega': [],
}

for i in range(nonbonded_terms['indices'].shape[0]):
    atom_indices = nonbonded_terms['indices'][i]

    distance_data = mdtraj.compute_distances(traj_cg, [atom_indices])
    distance_data = np.squeeze(distance_data).astype(np.float64)
    distance_data = torch.from_numpy(distance_data)
    distance_data.clamp_min_(0.0)    
    nonbonded_terms['r_rep_on'][i] = torch.min(distance_data).item()
    nonbonded_terms['r_off'][i] = 1.5
    
    distance = mdtraj.compute_distances(traj, [atom_indices])
    distance = np.squeeze(distance).astype(np.float64)
    distance = torch.from_numpy(distance)
    distance.clamp_min_(0.0)    

    basis = pccg.spline.bs_lj(distance,
                            nonbonded_terms['r_rep_on'][i],
                            nonbonded_terms['r_off'][i],
                            num_of_basis = 12)    

    distance_grid = torch.linspace(0.2, nonbonded_terms['r_off'][i], 100)
    basis_grid, omega = pccg.spline.bs_lj(distance_grid,
                                        nonbonded_terms['r_rep_on'][i],
                                        nonbonded_terms['r_off'][i],
                                        num_of_basis = 12,
                                        omega = True)
    
    nonbonded_terms['basis'].append(basis)
    nonbonded_terms['theta'].append(torch.zeros(basis.shape[1], dtype = torch.float64))
    nonbonded_terms['basis_grid'].append(basis_grid)
    nonbonded_terms['omega'].append(omega)
    
 #### compute log_q
log_q = []
for k in range(traj.n_frames):
    xyz = traj.xyz[k]
    context.setPositions(xyz)
    state = context.getState(getEnergy = True)
    log_q.append(-state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)/kT)
log_q = torch.tensor(log_q)

dF = torch.zeros(1, requires_grad = True, dtype = torch.float64)
def _gather_param():
    thetas = [ p for p in bonded_terms['angle']['theta'] ] + \
             [ p for p in bonded_terms['dihedral']['theta'] ] + \
             [ p for p in nonbonded_terms['theta'] ] + \
             [ dF ]
    return thetas

thetas = _gather_param()
for p in thetas:
    p.requires_grad_()

def _set_param(x):
    if type(x) is np.ndarray:
        x = torch.from_numpy(x)

    offset = 0
    for p in _gather_param():
        numel = p.numel()
        p.data.copy_(x[offset:offset + numel].view_as(p))
        offset += numel

def _zero_grad():
    for p in _gather_param():
        if p.grad is not None:
            p.grad.zero_()

x_init = torch.cat([p.data.detach().clone() for p in _gather_param()]).numpy()

def compute_NCE_loss_and_grad(x, weight_decay):
    _zero_grad()
    _set_param(x)

    u = bond_energy
    for i in range(len(bonded_terms['angle']['theta'])):
        basis = bonded_terms['angle']['basis'][i]
        theta = bonded_terms['angle']['theta'][i]
        u = u + torch.mv(basis, theta)
        
    for i in range(len(bonded_terms['dihedral']['theta'])):
        basis = bonded_terms['dihedral']['basis'][i]
        theta = bonded_terms['dihedral']['theta'][i]
        u = u + torch.mv(basis, theta)

    for i in range(len(nonbonded_terms['theta'])):
        basis = nonbonded_terms['basis'][i]
        theta = nonbonded_terms['theta'][i]
        u = u + torch.mv(basis, theta)

    log_p = - (u - dF) + u.new_tensor(n_data/n_noise).log()
    logit = log_p - log_q
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logit, target)

    for i in range(len(nonbonded_terms['theta'])):
        theta = nonbonded_terms['theta'][i]        
        omega = nonbonded_terms['omega'][i]
        loss = loss + weight_decay*torch.sum(theta*torch.mv(omega, theta))

    loss.backward()
    
    grad = torch.cat([p.grad.clone().data for p in _gather_param()])
    return loss.item(), grad.numpy()

options={"disp": True, "gtol": 1e-6}
weight_decay = 1e-8
results = optimize.minimize(
    compute_NCE_loss_and_grad, x_init,
    args = (weight_decay), jac=True,
    method="L-BFGS-B", options=options
)
x = results["x"]


exit()


ref_traj = mdtraj.Trajectory(ref_pdb.xyz, prmtop)
rmsd = mdtraj.rmsd(traj, ref_traj, atom_indices = alpha_carbon_atom_indices)
fig = plt.figure()
fig.clf()
plt.hist(rmsd, bins = 50, range = (0, 0.83), density = True, log = False)
plt.savefig('./output/rmsd_hist.pdf')
plt.close()
exit()


distances = np.empty((traj.n_frames, traj.n_frames))
for i in range(traj.n_frames):
    distances[i] = mdtraj.rmsd(traj, traj, i)
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

#ref_traj = mdtraj.Trajectory(pdb.xyz, prmtop)

rmsd = {}
rmsd_max = 0
for T in temperatures:
    print(T)
    rmsd[T] = []
    for idx_chunk in range(240):
        traj = mdtraj.load_dcd(f'./data/cln025/train/chunk_{idx_chunk}/traj_temperature_{T:.2f}.dcd', prmtop, stride = 10)
        rmsd[T].append(mdtraj.rmsd(traj, ref_traj, atom_indices = alpha_carbon_indices))
    rmsd[T] = np.concatenate(rmsd[T])

    rmsd_max = max(rmsd_max, rmsd[T].max())

fig = plt.figure(figsize = (6.4*4, 4.8*8))
fig.clf()
for i in range(len(temperatures)):
    plt.subplot(8, 4, i + 1)
    plt.hist(rmsd[temperatures[i]], bins = 50, range = (0, rmsd_max), density = True, log = True)
    plt.title(f"T = {temperatures[i]:.2f}")
plt.savefig('./output/rmsd_hist.pdf')
plt.close()
    


traj = []
T = 360.47
for idx_chunk in range(240):
    traj.append(mdtraj.load_dcd(f'./data/cln025/train/chunk_{idx_chunk}/traj_temperature_{T:.2f}.dcd', prmtop, stride = 10))
traj = mdtraj.join(traj)
rmsd = mdtraj.rmsd(traj, ref_traj, atom_indices = alpha_carbon_atom_indices)
fig = plt.figure()
fig.clf()
plt.hist(rmsd, bins = 50, range = (0, 0.83), density = True, log = False)
plt.savefig('./output/rmsd_hist_{T:.2f}.pdf')
pplt.close()

exit()
