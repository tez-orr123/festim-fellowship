
import festim as F
import numpy as np

avo = 6.022e23
W_D_0_D = 1
W_D_0_T = 1

W_E_D_D = 0
W_E_D_T = 0

Cu_D_0_D = 1
Cu_D_0_T = 1

Cu_E_D_D = 0
Cu_E_D_T = 0

CuCrZr_D_0_D = 1
CuCrZr_D_0_T = 1

CuCrZr_E_D_D = 0
CuCrZr_E_D_T = 0

tungsten = F.Material(
    D_0={"D": float(W_D_0_D), "T": (W_D_0_T)},
    E_D={"D": float(W_E_D_D), "T": (W_E_D_T)}, 
    K_S_0=1,
    E_K_S=0,
    thermal_conductivity=100,
)

copper = F.Material(
    D_0={"D": float(Cu_D_0_D), "T": (Cu_D_0_T)},
    E_D={"D": float(Cu_E_D_D), "T": (Cu_E_D_T)},
    K_S_0=1,
    E_K_S=0,
    thermal_conductivity=350,
)

cucrzr = F.Material(
    D_0={"D": float(CuCrZr_D_0_D), "T": (CuCrZr_D_0_T)},
    E_D={"D": float(CuCrZr_E_D_D), "T": (CuCrZr_E_D_T)},
    K_S_0=1, 
    E_K_S=0, 
    thermal_conductivity=350,
)

# 1D mesh

x0= 0.0
x1 = 5e-3

n_w = 5000

x_w = np.linspace(x0, x1, n_w)

mesh = np.concatenate([x_w])
shared_mesh = F.Mesh1D(mesh) # 

# Subdomains
W_volume = F.VolumeSubdomain1D(id=6, borders=[x0, x1], material=tungsten)

plasma_facing_side = F.SurfaceSubdomain1D(id=9, x=x0)
back = F.SurfaceSubdomain1D(id=11, x=x1)


all_subdomains = [
    W_volume,
    plasma_facing_side,
    back
]
##### Hydrogen Transport Discontinuous Problem #####

# Problem
my_model = F.HydrogenTransportProblemDiscontinuous()

# Penalty #1
my_model.method_interface = "penalty"

# Subdomains
my_model.subdomains = all_subdomains

w_density = 6.3e28
trap_density = (w_density * 0.00118) /avo

Deuterium = F.Species("D", subdomains=my_model.volume_subdomains)
trapped_1D = F.Species("D_1_trapped", mobile=False, subdomains=my_model.volume_subdomains)
trapped_2D = F.Species("D_2_trapped", mobile=False, subdomains=my_model.volume_subdomains)
Tritium = F.Species("T", subdomains=my_model.volume_subdomains)
trapped_1T = F.Species("T_1_trapped", mobile=False, subdomains=my_model.volume_subdomains)
trapped_2T = F.Species("T_2_trapped", mobile=False, subdomains=my_model.volume_subdomains)
empty_traps = F.Species("empty_traps", mobile=False, subdomains=my_model.volume_subdomains)
trapped_DT = F.Species("D_T_trapped", mobile=False, subdomains=my_model.volume_subdomains)

my_model.species = [Deuterium, Tritium, trapped_1D, trapped_2D, trapped_1T, trapped_2T, empty_traps]

my_model.initial_conditions = [F.InitialConcentration(value = trap_density, volume = W_volume, species=empty_traps)]

# Mesh
my_model.mesh = shared_mesh


my_model.surface_to_volume = {
    plasma_facing_side: W_volume,
    back: W_volume,
}

lattice_length = 1.1e-10  # m
n_solute_per_site = 6
my_model.reactions = [
    F.Reaction(
        reactant=[Deuterium, empty_traps],
        product=[trapped_1D],
        k_0 = 1, 
        E_k = 0, 
        p_0 = 1, 
        E_p = 0, 
        volume=W_volume,
    ),
    F.Reaction(
        reactant=[Deuterium, trapped_1D],
        product=[trapped_2D],
        k_0 = 1, 
        E_k = 0, 
        p_0 = 1,
        E_p = 0,
        volume=W_volume,
    ),
    F.Reaction(
        reactant=[Tritium, empty_traps],
        product=[trapped_1T],
        k_0 = 1, 
        E_k = 0, 
        p_0 = 1,
        E_p = 0,
        volume=W_volume,
    ),
        F.Reaction(
        reactant=[Tritium, trapped_1T],
        product=[trapped_2T],
        k_0 = 1, 
        E_k = 0, 
        p_0 = 1,
        E_p = 0,
        volume=W_volume,
    ),
    #     F.Reaction( # Making [DT] seems to be the problem here.
    #     reactant=[Deuterium, trapped_1T],
    #     product=[trapped_DT],
    #     k_0 = 1, 
    #     E_k = 0, 
    #     p_0 = 1,
    #     E_p = 0,
    #     volume=W_volume,
    # ),
        F.Reaction(
        reactant=[Tritium, trapped_1D],
        product=[trapped_DT],
        k_0 = 1, 
        E_k = 0, 
        p_0 = 1,
        E_p = 0,
        volume=W_volume,
    ),
]

# BCs
# import ufl
# phi = ((0.23e24)) /avo
# R_p = 1.1e-9 
# my_model.boundary_conditions = [
#     F.FixedConcentrationBC(
#         subdomain=plasma_facing_side,
#         value=lambda T: phi * R_p / (W_D_0_D * ufl.exp(-W_E_D_D / F.k_B / T)),
#         species=Deuterium
#     ),
#     F.FixedConcentrationBC(
#         subdomain=coolant_facing_side, 
#         value=0, 
#         species=Deuterium
#     ),
#     F.FixedConcentrationBC(
#         subdomain=plasma_facing_side,
#         value=lambda T: phi * R_p / (W_D_0_T * ufl.exp(-W_E_D_T / F.k_B / T)),
#         species=Tritium
#     ),
#     F.FixedConcentrationBC(
#         subdomain=coolant_facing_side, 
#         value=0, 
#         species=Tritium
#     ),
# ]

# Temperature field from heat transfer problem
my_model.temperature = 600

# Settings
my_model.settings = F.Settings(
    transient=False,
    atol=1e-10, 
    rtol=1e-10,
)




# SHOW THAT LOG
from dolfinx.log import LogLevel, set_log_level
# need
set_log_level(LogLevel.INFO)

my_model.initialise()
my_model.run()



