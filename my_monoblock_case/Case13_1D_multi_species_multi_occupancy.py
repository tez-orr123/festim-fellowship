# CASE 13
# Multi Occupancy Traps
# Multi species
# D + [] <-> [D]
# D + [D] <-> [2D]
# T + [] <-> [T]
# T + [T] <-> [2T]
# D + [T] <-> [DT]
# T + [D] <-> [DT]


# From the tutorial of multi-occupancy trapping:
# defined problem, then mesh, then mobile H, then three trapped H species of which are non mobile
# had empty traps as implicit species but I will have to try as explicit
# defined models species, surface subdomains, material, volume subdomains, model subdomains,
# reactions -> three reactions of mobile + empty trap <-> trap 1H, mobile + trap 1H <-> trap 2H, mobile + trap 2H <-> trap 3H
# defined boundary conditions, temperature, settings, stepsize and then GO
# I will want to compile the trapped concentrations to see total amount trapped
# Do not care about the concentrations of each individual level
# Now, what values of the trapping levels should I use?
# The ones from Sanjeets paper, E_p_values = [1.49, 1.46, 1.32, 1.21, 1.12, 0.53]
# Can just take the first three traps from this set and see with them.
#
# trap_1 = F.Trap(
#     k_0 = 2.6413e-17,
#     E_k = 0.21,
#     p_0 = 1e13,
#     E_p = 1.49,
#     density = metal_density * 0.01 * 6/21,  #
#     materials = [metal],
# )
#
# trap_2 = F.Trap(
#     k_0 = 2.6413e-17,
#     E_k = 0.21,  # E_D
#     p_0 = 1e13,  # attempt frequency
#     E_p = 1.46,  # binding energy + migration energy
#     density = metal_density * 0.01 * 12/21,
#     materials = [metal],
# )
#
# trap_3 = F.Trap(
#     k_0 = 2.6413e-17,
#     E_k = 0.21,  # E_D
#     p_0 = 1e13,  # attempt frequency
#     E_p = 1.32,  # binding energy + migration energy
#     density = metal_density * 0.01 * 18/21,
#     materials = [metal],
# )
#
# all_traps = [trap_1, trap_2, trap_3]
#
# k_0 will have to be /avo to be in mol/m3
# density of traps... may have to set this inital concentration separately?
# Let's just get going and see what happens.
#
# Sanjeets tungsten parameters were so different, this is at 3500!! very hot!
#


import festim as F
import numpy as np
import ufl
from dolfinx.log import LogLevel, set_log_level

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

x0 = 0.0
x1 = 5e-3
x2 = 6e-3
x3 = 8e-3

n_w = 5000
n_cu = 1000
n_cucrzr = 2000

x_w = np.linspace(x0, x1, n_w)
x_cu = np.linspace(x1, x2, n_cu)
x_cucrzr = np.linspace(x2, x3, n_cucrzr)

mesh = np.concatenate([x_w, x_cu, x_cucrzr])
shared_mesh = F.Mesh1D(mesh)  #

# Subdomains
W_volume = F.VolumeSubdomain1D(id=6, borders=[x0, x1], material=tungsten)
Cu_volume = F.VolumeSubdomain1D(id=7, borders=[x1, x2], material=copper)
CuCrZr_volume = F.VolumeSubdomain1D(id=8, borders=[x2, x3], material=cucrzr)

plasma_facing_side = F.SurfaceSubdomain1D(id=9, x=x0)
coolant_facing_side = F.SurfaceSubdomain1D(id=10, x=x3)
W_Cu_interface = F.SurfaceSubdomain1D(id=11, x=x1)
Cu_CuCrZr_interface = F.SurfaceSubdomain1D(id=12, x=x2)

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

heat_transfer_problem.boundary_conditions = [PF_temp, coolant_temp]

heat_transfer_problem.exports = [F.VTXTemperatureExport("monoblock_exports/temp.bp")]

heat_transfer_problem.settings = F.Settings(
    atol=1e-8,
    rtol=1e-8,
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

my_model.surface_to_volume = {
    plasma_facing_side: W_volume,
    coolant_facing_side: CuCrZr_volume,
}

W_density = 6.3e28 / avo
empty_traps_density = W_density * 0.003

Deuterium = F.Species("D", subdomains=my_model.volume_subdomains)
Tritium = F.Species("T", subdomains=my_model.volume_subdomains)

trapped_1D = F.Species(
    "D_1_trapped", mobile=False, subdomains=my_model.volume_subdomains
)
trapped_2D = F.Species(
    "D_2_trapped", mobile=False, subdomains=my_model.volume_subdomains
)
trapped_1T = F.Species(
    "T_1_trapped", mobile=False, subdomains=my_model.volume_subdomains
)
trapped_2T = F.Species(
    "T_2_trapped", mobile=False, subdomains=my_model.volume_subdomains
)
empty_traps = F.Species(
    "empty_traps", mobile=False, subdomains=my_model.volume_subdomains
)
trapped_DT = F.Species(
    "D_T_trapped", mobile=False, subdomains=my_model.volume_subdomains
)

empty_traps = F.ImplicitSpecies(
    n=empty_traps_density, others=[trapped_1D, trapped_1T, trapped_2D, trapped_2T, trapped_DT]
)

my_model.species = [
    Deuterium,
    Tritium,
    trapped_1D,
    trapped_2D,
    trapped_1T,
    trapped_2T,
    trapped_DT,
]

my_model.mesh = shared_mesh

# Penalty #2
penalty_term = 1e-5
my_model.interfaces = [
    F.Interface(id=11, subdomains=(W_volume, Cu_volume), penalty_term=penalty_term),
    F.Interface(
        id=12, subdomains=(Cu_volume, CuCrZr_volume), penalty_term=penalty_term
    ),
]

my_model.reactions = [
    F.Reaction(
        reactant=[Deuterium, empty_traps],
        product=[trapped_1D],
        k_0=1e-3,
        E_k=0.21,
        p_0=1e11,
        E_p=1.49,
        volume=W_volume,
    ),
    F.Reaction(
        reactant=[Deuterium, trapped_1D],
        product=[trapped_2D],
        k_0=1e-3,
        E_k=0.21,
        p_0=2 * 1e11,
        E_p=1.46,
        volume=W_volume,
    ),
    F.Reaction(
        reactant=[Tritium, empty_traps],
        product=[trapped_1T],
        k_0=1e-3,
        E_k=0.21,
        p_0=1e11,
        E_p=1.49,
        volume=W_volume,
    ),
    F.Reaction(
        reactant=[Tritium, trapped_1T],
        product=[trapped_2T],
        k_0=1e-3,
        E_k=0.21,
        p_0=2 * 1e11,
        E_p=1.46,
        volume=W_volume,
    ),
    F.Reaction(
        reactant=[Deuterium, trapped_1T],
        product=[trapped_DT],
        k_0=1e-3,
        E_k=0.21,
        p_0=2* 1e11,
        E_p=1.46,
        volume=W_volume,
    ),
    F.Reaction(
        reactant=[Tritium, trapped_1D],
        product=[trapped_DT],
        k_0=1e-3,
        E_k=0.21,
        p_0=2 * 1e11,
        E_p=1.46,
        volume=W_volume,
    ),
]

phi = (0.23e28) / avo
R_p = 1.1e-9
my_model.boundary_conditions = [
    F.FixedConcentrationBC(
        subdomain=plasma_facing_side,
        value=lambda T: phi * R_p / (W_D_0_D * ufl.exp(-W_E_D_D / F.k_B / T)),
        species=Deuterium,
    ),
    F.FixedConcentrationBC(subdomain=coolant_facing_side, value=0, species=Deuterium),
    F.FixedConcentrationBC(
        subdomain=plasma_facing_side,
        value=lambda T: phi * R_p / (W_D_0_T * ufl.exp(-W_E_D_T / F.k_B / T)),
        species=Tritium,
    ),
    F.FixedConcentrationBC(subdomain=coolant_facing_side, value=0, species=Tritium),
]

my_model.temperature = heat_transfer_problem.u

my_model.settings = F.Settings(
    transient=True,
    atol=1e-26, 
    rtol=1e-10,
    final_time=3.2e7,
)
my_model.settings.stepsize = F.Stepsize(
    initial_value=1e2,
    growth_factor=1.1,
    cutback_factor=0.9,
    target_nb_iterations=4,
    # milestones=[100, 1000, 1E6, 1E7, 3.2E7],
)



my_model.exports = [
    F.VTXSpeciesExport(
        filename=f"monoblock_exports/case13/multi_spe_implicit_multi_occ/implicit_tot_conc_{subdomain.id}.bp",
        field=my_model.species,
        subdomain=subdomain,
        #times=[100, 1000, 1E6, 1E7, 3.2E7],
    )  
    for subdomain in my_model.volume_subdomains
]

# need
set_log_level(LogLevel.INFO)

my_model.initialise()
my_model.run()
