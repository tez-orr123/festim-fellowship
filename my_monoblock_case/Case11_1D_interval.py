import festim as F
import numpy as np

avo = 6.022e23
W_D_0_D = 4.1e-7
W_D_0_T = 4.1e-7

W_E_D_D = 0.38
W_E_D_T = 0.39

Cu_D_0_D = 6.6e-7
Cu_D_0_T = 6.6e-7

Cu_E_D_D = 0.377
Cu_E_D_T = 0.387

CuCrZr_D_0_D = 3.92e-7
CuCrZr_D_0_T = 3.92e-7

CuCrZr_E_D_D = 0.408
CuCrZr_E_D_T = 0.418

tungsten = F.Material(
    D_0={"D": float(W_D_0_D), "T": (W_D_0_T)},
    E_D={"D": float(W_E_D_D), "T": (W_E_D_T)}, 
    K_S_0=1.87e24 / avo,
    E_K_S=1.04,
    thermal_conductivity=100,
)

copper = F.Material(
    D_0={"D": float(Cu_D_0_D), "T": (Cu_D_0_T)},
    E_D={"D": float(Cu_E_D_D), "T": (Cu_E_D_T)},
    K_S_0=3.14e24 / avo,
    E_K_S=0.572,
    thermal_conductivity=350,
)

cucrzr = F.Material(
    D_0={"D": float(CuCrZr_D_0_D), "T": (CuCrZr_D_0_T)},
    E_D={"D": float(CuCrZr_E_D_D), "T": (CuCrZr_E_D_T)},
    K_S_0=4.28e23 / avo, 
    E_K_S=0.387, 
    thermal_conductivity=350,
)

# 1D mesh

# NEED to do the following so that the volumes are stitched together properly
# Define segment end point:
x0= 0.0
x1 = 5e-3
x2 = 6e-3
x3 = 8e-3
# Define points per region (the same relative to their size, 1000 per mm)
n_w = 5000
n_cu = 1000
n_cucrzr = 2000

# Make linspace meshes of the three volumes then stitch them together with concatenate
x_w = np.linspace(x0, x1, n_w)
x_cu = np.linspace(x1, x2, n_cu)
x_cucrzr = np.linspace(x2, x3, n_cucrzr)

# THIS IS VERY IMPORTANT ##################################################################
mesh = np.concatenate([x_w, x_cu, x_cucrzr])
shared_mesh = F.Mesh1D(mesh) # THIS ensuring that both problem use the SAME mesh
###########################################################################################

# Subdomains
W_volume = F.VolumeSubdomain1D(id=6, borders=[0, 5e-3], material=tungsten)
Cu_volume = F.VolumeSubdomain1D(id=7, borders=[5e-3, 6e-3], material=copper)
CuCrZr_volume = F.VolumeSubdomain1D(id=8, borders=[6e-3, 8e-3], material=cucrzr)

plasma_facing_side = F.SurfaceSubdomain1D(id=9, x=0)
coolant_facing_side = F.SurfaceSubdomain1D(id=10, x=8e-3)
W_Cu_interface = F.SurfaceSubdomain1D(id=11, x=5e-3)
Cu_CuCrZr_interface = F.SurfaceSubdomain1D(id=12, x=6e-3)

all_subdomains = [
    W_volume,
    Cu_volume,
    CuCrZr_volume,
    plasma_facing_side,
    coolant_facing_side,
    W_Cu_interface,
    Cu_CuCrZr_interface,
]

##### HEAT TRANSFER PROBLEM #####
heat_transfer_problem = F.HeatTransferProblem()

heat_transfer_problem.subdomains = all_subdomains

heat_transfer_problem.mesh = shared_mesh

PF_temp = F.FixedTemperatureBC(subdomain=plasma_facing_side, value=1173)
coolant_temp = F.FixedTemperatureBC(subdomain=coolant_facing_side, value=773)

heat_transfer_problem.boundary_conditions = [
    PF_temp,
    coolant_temp
]

heat_transfer_problem.exports = [F.VTXTemperatureExport("monoblock_exports/temp.bp")]

heat_transfer_problem.settings = F.Settings(
    atol=1e-10,
    rtol=1e-10,
    transient=False,
)

heat_transfer_problem.initialise()
heat_transfer_problem.run()

##### Hydrogen Transport Discontinuous Problem #####

# Problem
my_model = F.HydrogenTransportProblemDiscontinuous()

# Penalty #1
my_model.method_interface = "penalty"

# Subdomains
my_model.subdomains = all_subdomains

# Species, try all explicit species
Deuterium = F.Species("D", subdomains=my_model.volume_subdomains)
trapped_D = F.Species("D_trapped", mobile=False, subdomains=my_model.volume_subdomains)
Tritium = F.Species("T", subdomains=my_model.volume_subdomains)
trapped_T = F.Species("T_trapped", mobile=False, subdomains=my_model.volume_subdomains)
empty_traps = F.Species("empty_traps", mobile=False, subdomains=my_model.volume_subdomains)
my_model.species = [Deuterium, Tritium, trapped_D, trapped_T, empty_traps]
my_model.initial_conditions = [F.InitialConcentration(value = 1E10, volume = W_volume, species=empty_traps)]
# w_density = 6.3e28 / avo
# trap_density = 1e17
# empty_traps = F.ImplicitSpecies(n = trap_density, others = [trapped_D, trapped_T])
# my_model.species = [Deuterium, trapped_D, Tritium, trapped_T]
# So implicit species of empty traps results in error of:
# ValueError: Cannot compute concentration of None because T_trapped has no solution.

# But produces good results when empty_traps is an explicit species
# Which is preferred?

# Subdomains
my_model.subdomains = all_subdomains

# Mesh
my_model.mesh = shared_mesh


my_model.surface_to_volume = {
    plasma_facing_side: W_volume,
    coolant_facing_side: CuCrZr_volume,
}

# Penalty #2
penalty_term = 1e-5
my_model.interfaces = [
    F.Interface(
        id=11, subdomains=(W_volume, Cu_volume), penalty_term=penalty_term
        ),
    F.Interface(id=12, subdomains=(Cu_volume, CuCrZr_volume), penalty_term=penalty_term)
]

# Trapping reactions
lattice_length = 1.1e-10  # m
n_solute_per_site = 6
my_model.reactions = [
    F.Reaction(
        reactant=[Deuterium, empty_traps],
        product=[trapped_D],
        # the test will be k_0 being 1e13 and then 1e-10
        # doesn't run with 1e13, mumps solver crashes
        k_0=1e-10,
        #k_0=((W_D_0_D/((lattice_length)**2 * n_solute_per_site))/avo), # trapping pre-exponential factor k_0 = (1/6) * 1e13 / rho <- from sanjeet task
        E_k=0.265, # trapping activation energy
        p_0=1.2397e13, # detrapping pre-exponential factor
        E_p = 0.83, # detrapping activation energy, p = p_0 exp( - E_p/kT )
        volume=W_volume,
    ),
    F.Reaction(
        reactant=[Tritium, empty_traps],
        product=[trapped_T],
        k_0=1e-10,
        #k_0=(((W_D_0_T)/((lattice_length)**2 * n_solute_per_site))/avo), # trapping pre-exponential factor k_0 = (1/6) * 1e13 / rho <- from sanjeet task
        E_k=0.265, # trapping activation energy
        p_0=1.2397e13, # detrapping pre-exponential factor
        E_p = 0.83, # detrapping activation energy, p = p_0 exp( - E_p/kT )
        volume=W_volume,
    ),
]

# BCs
import ufl
phi = ((0.23e24) / 2)/avo
R_p = 1.1e-9 
my_model.boundary_conditions = [
    F.FixedConcentrationBC(
        subdomain=plasma_facing_side,
        value=lambda T: phi * R_p / (W_D_0_D * ufl.exp(-W_E_D_D / F.k_B / T)),
        species=Deuterium
    ),
    F.FixedConcentrationBC(
        subdomain=coolant_facing_side, 
        value=0, 
        species=Deuterium
    ),
    F.FixedConcentrationBC(
        subdomain=plasma_facing_side,
        value=lambda T: phi * R_p / (W_D_0_T * ufl.exp(-W_E_D_T / F.k_B / T)),
        species=Tritium
    ),
    F.FixedConcentrationBC(
        subdomain=coolant_facing_side, 
        value=0, 
        species=Tritium
    ),
]

# Temperature field from heat transfer problem
my_model.temperature = heat_transfer_problem.u

# Settings
my_model.settings = F.Settings(
    transient=True,
    atol=1e-17, # lower tolerance if we solving in zero iterations
    rtol=1e-10,
    final_time=3.2e7,
)
my_model.settings.stepsize = F.Stepsize(
    initial_value=1e6,
    growth_factor=1.1,
    cutback_factor=0.9,
    target_nb_iterations=4,
)

# Exports
my_model.exports = [
        F.VTXSpeciesExport(filename=f"monoblock_exports/{spe.name}_{subdomain.id}.bp", field=spe, subdomain=subdomain)
        for spe in my_model.species
        for subdomain in my_model.volume_subdomains
        ]

# SHOW THAT LOG
from dolfinx.log import LogLevel, set_log_level
# need
set_log_level(LogLevel.INFO)

my_model.initialise()
my_model.run()