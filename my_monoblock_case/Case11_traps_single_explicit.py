# Case 11.2
# Set-up:
# -	Discontinuous
# -	Transient
# -	Single species
# -	1 intrinsic W trap, EXPLICIT
# - Multi-occupancy trap, 3 levels
# -	Temperature from heat transfer problem

import festim as F
import numpy as np
import ufl
from dolfinx.log import LogLevel, set_log_level

# materials
avo = 6.022e23

W_D_0_H = 4.1e-7

W_E_D_H = 0.38

Cu_D_0_H = 6.6e-7

Cu_E_D_H = 0.377

CuCrZr_D_0_H = 3.92e-7

CuCrZr_E_D_H = 0.408

tungsten = F.Material(
    D_0=W_D_0_H,
    E_D=W_E_D_H,
    K_S_0=1.87e24
    / avo,  # in the monoblock tutorial, they don't have the divide by avo, OOM is e24 hmmmm
    # but then tolerances would have to change to 1e10 or something too right... NOOOOOO WRONG DONT DO THAT
    E_K_S=1.04,
    thermal_conductivity=100,
    density=19300,  # kg/m3
    heat_capacity=134,  # J/kg/K
)

copper = F.Material(
    D_0=Cu_D_0_H,
    E_D=Cu_E_D_H,
    K_S_0=3.14e24 / avo,
    E_K_S=0.572,
    thermal_conductivity=350,
    density=8900,
    heat_capacity=390,  # at around 900 celsius
)

cucrzr = F.Material(
    D_0=CuCrZr_D_0_H,
    E_D=CuCrZr_E_D_H,
    K_S_0=4.28e23 / avo,
    E_K_S=0.387,
    thermal_conductivity=350,
    density=8960,
    heat_capacity=383,
)
# ------------------------------------------------


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
W_volume = F.VolumeSubdomain1D(id=1, borders=[x0, x1], material=tungsten)
Cu_volume = F.VolumeSubdomain1D(id=2, borders=[x1, x2], material=copper)
CuCrZr_volume = F.VolumeSubdomain1D(id=3, borders=[x2, x3], material=cucrzr)

plasma_facing_side = F.SurfaceSubdomain1D(id=4, x=x0)
coolant_facing_side = F.SurfaceSubdomain1D(id=5, x=x3)
W_Cu_interface = F.SurfaceSubdomain1D(id=6, x=x1)
Cu_CuCrZr_interface = F.SurfaceSubdomain1D(id=7, x=x2)

all_subdomains = [
    W_volume,
    Cu_volume,
    CuCrZr_volume,
    plasma_facing_side,
    coolant_facing_side,
    W_Cu_interface,
    Cu_CuCrZr_interface,
]

# Heat transfer problem
heat_transfer_problem = F.HeatTransferProblem()

heat_transfer_problem.subdomains = all_subdomains

heat_transfer_problem.mesh = shared_mesh

heat_flux_PF = F.FixedTemperatureBC(subdomain=plasma_facing_side, value=1173)
coolant_temp = F.FixedTemperatureBC(subdomain=coolant_facing_side, value=773)

heat_transfer_problem.boundary_conditions = [heat_flux_PF, coolant_temp]

heat_transfer_problem.exports = [F.VTXTemperatureExport("monoblock_exports/temp.bp")]

heat_transfer_problem.settings = F.Settings(
    atol=1e-10,
    rtol=1e-10,
    transient=False,
)

heat_transfer_problem.initialise()
heat_transfer_problem.run()


# H transport problem
my_model = F.HydrogenTransportProblemDiscontinuous()

my_model.method_interface = "penalty"

my_model.subdomains = all_subdomains

my_model.surface_to_volume = {
    plasma_facing_side: W_volume,
    coolant_facing_side: CuCrZr_volume,
}

mobile_H = F.Species("mobile_H", subdomains=my_model.volume_subdomains)
trapped_1H = F.Species(
    "1H_trapped", mobile=False, subdomains=my_model.volume_subdomains
)
trapped_2H = F.Species(
    "2H_trapped", mobile=False, subdomains=my_model.volume_subdomains
)
trapped_3H = F.Species(
    "3H_trapped", mobile=False, subdomains=my_model.volume_subdomains
)
empty_traps = F.Species(
    "empty_traps", mobile=False, subdomains=my_model.volume_subdomains
)

my_model.species = [mobile_H, trapped_1H, trapped_2H, trapped_3H, empty_traps]

W_density = 6.3e28 / avo
empty_traps_density = W_density * 0.003
my_model.initial_conditions = [
    F.InitialConcentration(
        value=empty_traps_density, volume=W_volume, species=empty_traps
    ),
]

my_model.mesh = shared_mesh

penalty_term = 1e-5
my_model.interfaces = [
    F.Interface(id=15, subdomains=(W_volume, Cu_volume), penalty_term=penalty_term),
    F.Interface(
        id=16, subdomains=(Cu_volume, CuCrZr_volume), penalty_term=penalty_term
    ),
]

my_model.reactions = [
    F.Reaction(
        reactant=[mobile_H, empty_traps],
        product=[trapped_1H],
        k_0=1e-3,  # MAke super HIGH??
        E_k=0.21,
        p_0=1.0e11,
        E_p=1.49,
        volume=W_volume,
    ),
    F.Reaction(
        reactant=[mobile_H, trapped_1H],
        product=[trapped_2H],
        k_0=1e-3,
        E_k=0.21,
        p_0=2 * 1e11,
        E_p=1.46,
        volume=W_volume,
    ),
    F.Reaction(
        reactant=[mobile_H, trapped_2H],
        product=[trapped_3H],
        k_0=1e-3,
        E_k=0.21,
        p_0=3 * 1e11,
        E_p=1.39,
        volume=W_volume,
    ),
]

phi = (0.23e28) / avo
R_p = 1.1e-9
my_model.boundary_conditions = [
    F.FixedConcentrationBC(
        subdomain=plasma_facing_side,
        value=lambda T: phi * R_p / (W_D_0_H * ufl.exp(-W_E_D_H / F.k_B / T)),
        species=mobile_H,
    ),
    F.FixedConcentrationBC(subdomain=coolant_facing_side, value=0, species=mobile_H),
]

my_model.temperature = heat_transfer_problem.u

my_model.settings = F.Settings(
    transient=True,
    atol=1e-16,
    rtol=1e-10,
    final_time=3.2e7,
)
my_model.settings.stepsize = F.Stepsize(
    initial_value=1e2,
    growth_factor=1.1,
    cutback_factor=0.9,
    target_nb_iterations=4,
    milestones=[100, 1000, 1E6, 1E7, 3.2E7],
)


my_model.exports = [
    F.VTXSpeciesExport(
        filename=f"monoblock_exports/single_explicit_multi_occ/explicit_tot_conc_{subdomain.id}.bp",
        field=my_model.species,
        subdomain=subdomain,
        times=[100, 1000, 1E6, 1E7, 3.2E7],
    )
    for subdomain in my_model.volume_subdomains
]

# need
set_log_level(LogLevel.INFO)

my_model.initialise()
my_model.run()
