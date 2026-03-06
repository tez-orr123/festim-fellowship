# Case 11.3
# Set-up:
# -	Discontinuous
# -	Transient
# -	Multispecies
# -	1 intrinsic W trap
# -	Temperature from a heat transfer problem

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
    K_S_0=1.87e24 / avo, # in the monoblock tutorial, they don't have the divide by avo, OOM is e24 hmmmm
    # but then tolerances would have to change to 1e10 or something too right... NOOOOOO WRONG DONT DO THAT
    E_K_S=1.04,
    thermal_conductivity=100,
    density = 19300, # kg/m3
    heat_capacity=134 # J/kg/K
)

copper = F.Material(
    D_0={"D": float(Cu_D_0_D), "T": (Cu_D_0_T)},
    E_D={"D": float(Cu_E_D_D), "T": (Cu_E_D_T)},
    K_S_0=3.14e24 / avo,
    E_K_S=0.572,
    thermal_conductivity=350,
    density=8900,
    heat_capacity=390 # at around 900 celsius
)

cucrzr = F.Material(
    D_0={"D": float(CuCrZr_D_0_D), "T": (CuCrZr_D_0_T)},
    E_D={"D": float(CuCrZr_E_D_D), "T": (CuCrZr_E_D_T)},
    K_S_0=4.28e23 / avo, 
    E_K_S=0.387, 
    thermal_conductivity=350,
    density = 8960,
    heat_capacity=383 
)
# ------------------------------------------------


# Define mesh from xdmf files
mesh = F.MeshFromXDMF("SALOME_meshes/monoblock_mesh_domains.xdmf", "SALOME_meshes/monoblock_mesh_boundaries.xdmf")

mesh.mesh.geometry.x[:] *= 1e-3
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

heat_transfer_problem.exports = [F.VTXTemperatureExport("monoblock_exports/temp.bp")]

heat_transfer_problem.settings = F.Settings(
    atol=1e-10,
    rtol=1e-10,
    transient=False,
    #final_time=3.2e7,
)
# heat_transfer_problem.settings.stepsize = F.Stepsize(
#     initial_value=1e4,
#     growth_factor=1.1,
#     cutback_factor=0.9,
#     target_nb_iterations=4,
# )

heat_transfer_problem.initialise()
heat_transfer_problem.run()
# --------------------------------------------------------



# H transport problem
# method interface, subdomains, species, penalty term stuff, BCs, heat_transfer_problem.u for temperature, settings, run
my_model = F.HydrogenTransportProblemDiscontinuous()

my_model.method_interface = "penalty"

my_model.subdomains = all_subdomains


Deuterium = F.Species("D", subdomains=my_model.volume_subdomains)
trapped_D = F.Species("D_trapped", mobile=False, subdomains=my_model.volume_subdomains)
Tritium = F.Species("T", subdomains=my_model.volume_subdomains)
trapped_T = F.Species("T_trapped", mobile=False, subdomains=my_model.volume_subdomains)
w_density = 6.3e28 # NOT DIVIDED BY AVO HERE
trap_density = (w_density * 0.00118)
empty_traps = F.Species("empty_traps", mobile=False, subdomains=my_model.volume_subdomains)
my_model.species = [Deuterium, trapped_D, Tritium, trapped_T, empty_traps]
my_model.initial_conditions = [F.InitialConcentration(value=trap_density, volume=W_volume, species=empty_traps)]

my_model.mesh = mesh

my_model.surface_to_volume = {
    top: W_volume,
    coolant_face: CuCrZr_volume,
    W_sides: W_volume,
    Cu_sides: Cu_volume,
    CuCrZr_sides: CuCrZr_volume,
    bottom: W_volume
}

penalty_term = 1e-5
my_model.interfaces = [
    F.Interface(
        id=15, subdomains=(W_volume, Cu_volume), penalty_term=penalty_term
        ),
    F.Interface(id=16, subdomains=(Cu_volume, CuCrZr_volume), penalty_term=penalty_term)
]

lattice_length = 1.1e-10  # m
n_solute_per_site = 6
my_model.reactions = [
    F.Reaction(
        reactant=[Deuterium, empty_traps],
        product=[trapped_D],
        k_0=((W_D_0_D/((lattice_length)**2 * n_solute_per_site))/avo), # trapping pre-exponential factor k_0 = (1/6) * 1e13 / rho <- from sanjeet task
        E_k=0.265, # trapping activation energy
        p_0=1.2397e11, # detrapping pre-exponential factor
        E_p = 1.3, # detrapping activation energy, p = p_0 exp( - E_p/kT )
        volume=W_volume,
    ),
    F.Reaction(
        reactant=[Tritium, empty_traps],
        product=[trapped_T],
        k_0=(((W_D_0_T)/((lattice_length)**2 * n_solute_per_site))/avo), # trapping pre-exponential factor k_0 = (1/6) * 1e13 / rho <- from sanjeet task
        E_k=0.265, # trapping activation energy
        p_0=1.2397e11, # detrapping pre-exponential factor
        E_p = 1.3, # detrapping activation energy, p = p_0 exp( - E_p/kT )
        volume=W_volume,
    ),
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

#This transient but temp not, will it mansion
my_model.settings = F.Settings(
    transient=True,
    atol=1e-8, # lower tolerance if we solving in zero iterations
    rtol=1e-10,
    final_time=3.2e7,
)
my_model.settings.stepsize = F.Stepsize(
    initial_value=1e6,
    growth_factor=1.1,
    cutback_factor=0.9,
    target_nb_iterations=4,
)


my_model.exports = [
        F.VTXSpeciesExport(filename=f"monoblock_exports/{spe.name}_{subdomain.id}.bp", field=spe, subdomain=subdomain)
        for spe in my_model.species
        for subdomain in my_model.volume_subdomains
        ]

from dolfinx.log import LogLevel, set_log_level
# need
set_log_level(LogLevel.INFO)

my_model.initialise()
my_model.run()

# I'd like a quick visualisation if pos before putting it into Paraview as it's time consuming and a faff

import pyvista

pyvista.set_jupyter_backend("html")
from dolfinx import plot

T = heat_transfer_problem.u

topology, cell_types, geometry = plot.vtk_mesh(T.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["T"] = T.x.array.real
u_grid.set_active_scalars("T")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, cmap="inferno", show_edges=False)
u_plotter.add_mesh(u_grid, style="wireframe", color="white", opacity=0.2)

contours = u_grid.contour(9)
u_plotter.add_mesh(contours, color="white")

u_plotter.view_xy()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("temperature.png")


u_plotter = pyvista.Plotter()

for vol in my_model.volume_subdomains:
    sol = Deuterium.subdomain_to_post_processing_solution[vol]

    topology, cell_types, geometry = plot.vtk_mesh(sol.function_space)
    u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    u_grid.point_data["c"] = sol.x.array.real
    u_grid.set_active_scalars("c")
    u_plotter.add_mesh(u_grid, cmap="viridis", show_edges=False)
    u_plotter.add_mesh(u_grid, style="wireframe", color="white", opacity=0.2)

u_plotter.view_xy(negative=True)


if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")


# WILL try and get single species trap case with heat transfer
# problem running first and then try the multi species way.

# slow at first but did indeed run, just pyvista that didn't want
# to run as there's no notebook package
# get trapped D density of 0 in tungsten.... wrong...