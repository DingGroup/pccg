import numpy as np
import simtk.openmm as omm
import simtk.unit as unit
import simtk.openmm.app as ommapp
import math
import mdtraj
import pickle
from sys import exit
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/MMFlow")
from MMFlow import utils

def make_system(masses, coor_transformer, bonded_parameters, T):
    
    ## add particles
    system = omm.System()
    for m in masses:
        system.addParticle(m)

    Kb = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
    Kb = Kb.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)

    ## bond force
    bond_force = omm.CustomBondForce("0.5*kb*(r - b0)^2*Kb*T")
    bond_force.addGlobalParameter("Kb", Kb)
    bond_force.addGlobalParameter("T", T)
    bond_force.addPerBondParameter('b0')
    bond_force.addPerBondParameter('kb')
    bond_force.addBond(
        coor_transformer.ref_particle_1,
        coor_transformer.ref_particle_2,
        [bonded_parameters['reference_particle_2_bond']['b0'],
         bonded_parameters['reference_particle_2_bond']['kb']]
    )

    bond_force.addBond(
        coor_transformer.ref_particle_1,
        coor_transformer.ref_particle_3,
        [bonded_parameters['reference_particle_3_bond']['b0'],
         bonded_parameters['reference_particle_3_bond']['kb']]
    )

    for i in range(len(coor_transformer.particle_visited_in_order)):
        p = coor_transformer.particle_visited_in_order[i]
        p1 = coor_transformer.bond_particle_idx[p][0]
        p2 = coor_transformer.bond_particle_idx[p][1]
        bond_force.addBond(
            p1,
            p2,
            [float(bonded_parameters['bond']['b0'][i]),
             float(bonded_parameters['bond']['kb'][i])]
        )
    bond_force.setForceGroup(0)
    system.addForce(bond_force)
    
    ## custom angle force
    angle_force_between_reference_particles = omm.CustomAngleForce("0.5*ka*(theta - a0)^2*Kb*T")
    angle_force_between_reference_particles.addGlobalParameter('Kb', Kb)
    angle_force_between_reference_particles.addGlobalParameter('T', T)
    angle_force_between_reference_particles.addPerAngleParameter('a0')
    angle_force_between_reference_particles.addPerAngleParameter('ka')
    angle_force_between_reference_particles.addAngle(
        coor_transformer.ref_particle_2,
        coor_transformer.ref_particle_1,
        coor_transformer.ref_particle_3,
        [bonded_parameters['reference_particle_3_angle']['a0'],
         bonded_parameters['reference_particle_3_angle']['ka']]
    )

    angle_force_between_reference_particles.setForceGroup(0)
    system.addForce(angle_force_between_reference_particles)

    angle_force = omm.CustomAngleForce("(0.5*ka*(theta - a0)^2 + log(sin(pi - theta)))*Kb*T")
    angle_force.addGlobalParameter('Kb', Kb)
    angle_force.addGlobalParameter('T', T)
    angle_force.addGlobalParameter('pi', math.pi)
    angle_force.addPerAngleParameter('a0')
    angle_force.addPerAngleParameter('ka')

    for i in range(len(coor_transformer.particle_visited_in_order)):
        p = coor_transformer.particle_visited_in_order[i]
        p1 = coor_transformer.angle_particle_idx[p][0]
        p2 = coor_transformer.angle_particle_idx[p][1]
        p3 = coor_transformer.angle_particle_idx[p][2]    
        angle_force.addAngle(
            p1,
            p2,
            p3,
            [float(bonded_parameters['angle']['a0'][i]),
             float(bonded_parameters['angle']['ka'][i])]
        )
    angle_force.setForceGroup(0)        
    system.addForce(angle_force)

    ## torsion force
    dihedral_parameters = bonded_parameters['dihedral']
    for i in range(len(coor_transformer.particle_visited_in_order)):
        p = coor_transformer.particle_visited_in_order[i]
        p1, p2, p3, p4 = coor_transformer.dihedral_particle_idx[p]
        f = omm.Continuous1DFunction(dihedral_parameters[i]['U'], -np.pi, np.pi, periodic = True)
        torsion = omm.CustomCompoundBondForce(4, f"ud_{i}(dihedral(p1, p2, p3, p4))*Kb*T")
        torsion.addGlobalParameter('Kb', Kb)
        torsion.addGlobalParameter('T', T)    
        torsion.addTabulatedFunction(f"ud_{i}", f)
        torsion.addBond([p1, p2, p3, p4])
        torsion.setForceGroup(0)
        system.addForce(torsion)

    system.addForce(omm.CMMotionRemover())
    
    return system
