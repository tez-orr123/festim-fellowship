# Case 10
# Set-up:
# -	Discontinuous
# -	Transient
# -	Multispecies
# -	No traps
# -	Temperature from a heat transfer problem
# -	Thinking I can look at the monoblock case in the tutorial… OH YEAH IT’S JUST THE SAME AS MINE COOL!!
# - Taking the case 8 problem and changing the temperature so that it's from a heat transfer problem

# Looking at the monoblock case:
# converted .med to .xdmf - don't need
# import festim - done
# materials - done
# define mesh using .xdmf files - done
# subdomain defining - done
# heat transfer problem, subdomains, mesh, BCs, exports, settings, run - done
# then H transport problem discontinuous
# method interface, subdomains, species, penalty term stuff, BCs, heat_transfer_problem.u for temperature, settings, run

import festim as F
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

# materials
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
    K_S_0=1.87e24/avo,
    E_K_S=1.04,
    thermal_conductivity=100,
    density = 19300, # kg/m3
    heat_capacity=134 # J/kg/K
)

copper = F.Material(
    D_0={"D": float(Cu_D_0_D), "T": (Cu_D_0_T)},
    E_D={"D": float(Cu_E_D_D), "T": (Cu_E_D_T)},
    K_S_0=3.14e24/avo,
    E_K_S=0.572,
    thermal_conductivity=350,
    density=8900,
    heat_capacity=390 # at around 900 celsius
)

cucrzr = F.Material(
    D_0={"D": float(CuCrZr_D_0_D), "T": (CuCrZr_D_0_T)},
    E_D={"D": float(CuCrZr_E_D_D), "T": (CuCrZr_E_D_T)},
    K_S_0=4.28e23/avo, 
    E_K_S=0.387, 
    thermal_conductivity=350,
    density = 8960,
    heat_capacity=383 
)
# ------------------------------------------------


# Define mesh from xdmf files
mesh = F.MeshFromXDMF("SALOME_meshes/my_monoblock_mesh_domains.xdmf", "SALOME_meshes/my_monoblock_mesh_boundaries.xdmf")
# -------------------------------------------------




# Subdomains 
W_volume = F.VolumeSubdomain(id=6, material=tungsten)
Cu_volume = F.VolumeSubdomain(id=7, material=copper)
CuCrZr_volume = F.VolumeSubdomain(id=8, material=cucrzr)

top = F.SurfaceSubdomain(id=9,)
bottom = F.SurfaceSubdomain(id=11,)
W_sides = F.SurfaceSubdomain(id=10,)
Cu_sides = F.SurfaceSubdomain(id=12,)
CuCrZr_sides = F.SurfaceSubdomain(id=13,)
W_Cu_interlayer = F.SurfaceSubdomain(id=15,)
Cu_CuCrZr_interlayer = F.SurfaceSubdomain(id=16,)
coolant_face = F.SurfaceSubdomain(id=14,)

all_subdomains = [top, bottom, W_sides, Cu_sides, CuCrZr_sides, W_Cu_interlayer, Cu_CuCrZr_interlayer, coolant_face, W_volume, Cu_volume, CuCrZr_volume]
# ----------------------------------------------------------

# Heat transfer problem
heat_transfer_problem = F.HeatTransferProblem()

heat_transfer_problem.subdomains = all_subdomains

heat_transfer_problem.mesh = mesh

heat_flux_PF = F.FixedTemperatureBC(subdomain=top, value=1173)
coolant_temp = F.FixedTemperatureBC(subdomain=coolant_face, value=773)

heat_transfer_problem.boundary_conditions = [
    heat_flux_PF,
    coolant_temp
]

heat_transfer_problem.exports = [F.VTXTemperatureExport("temp.bp")]

heat_transfer_problem.settings = F.Settings(
    atol=1e-10,
    rtol=1e-10,
    transient=False,
)

heat_transfer_problem.initialise()
heat_transfer_problem.run()
# --------------------------------------------------------



# H transport problem
# method interface, subdomains, species, penalty term stuff, BCs, heat_transfer_problem.u for temperature, settings, run
my_model = F.HydrogenTransportProblemDiscontinuous()

my_model.method_interface = "penalty"

my_model.subdomains = all_subdomains

Deuterium = F.Species("D", subdomains=my_model.volume_subdomains)
Tritium = F.Species("T", subdomains=my_model.volume_subdomains)
my_model.species = [Deuterium, Tritium]

my_model.mesh = mesh

my_model.surface_to_volume = {
    top: W_volume,
    coolant_face: CuCrZr_volume,
    W_sides: W_volume,
    Cu_sides: Cu_volume,
    CuCrZr_sides: CuCrZr_volume,
    bottom: W_volume
}

penalty_term = 1e-5 # Go up when struggling
my_model.interfaces = [
    F.Interface(
        id=15, subdomains=(W_volume, Cu_volume), penalty_term=penalty_term
        ),
    F.Interface(id=16, subdomains=(Cu_volume, CuCrZr_volume), penalty_term=penalty_term)
]

import ufl
phi = ((0.23e24) / 2)/avo
R_p = 1.1e-9 
my_model.boundary_conditions = [
    F.FixedConcentrationBC(
        subdomain=top,
        value=lambda T: phi * R_p / (W_D_0_D * ufl.exp(-W_E_D_D / F.k_B / T)),
        species=Deuterium
    ),
    F.FixedConcentrationBC(
        subdomain=coolant_face, 
        value=0, 
        species=Deuterium
    ),
    F.FixedConcentrationBC(
        subdomain=top,
        value=lambda T: phi * R_p / (W_D_0_T * ufl.exp(-W_E_D_T / F.k_B / T)),
        species=Tritium
    ),
    F.FixedConcentrationBC(
        subdomain=coolant_face, 
        value=0, 
        species=Tritium
    ),
]

my_model.temperature = heat_transfer_problem.u # Should take the temperature from the heat transfer problem

# Steady-state for now
my_model.settings = F.Settings(
    transient=False,
    atol=1e-10,
    rtol=1e-10,
    #final_time=3.2e7,
)
# my_model.settings.stepsize = F.Stepsize(
#     initial_value=5e6,
#     growth_factor=1.1,
#     cutback_factor=0.9,
#     target_nb_iterations=4,
# )


my_model.exports = [
        F.VTXSpeciesExport(filename=f"monoblock_exports/{spe.name}_{subdomain.id}.bp", field=spe, subdomain=subdomain)
        for spe in my_model.species
        for subdomain in my_model.volume_subdomains
        ]

from dolfinx.log import LogLevel, set_log_level

set_log_level(LogLevel.INFO)

my_model.initialise()
my_model.run()
