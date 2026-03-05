# Case 2
# Set-up:
# -	Continuous hydrogen transport problem
# -	Transient, over a FPY
# -	Multispecies, D and T
# -	No traps
# -	No temperature gradient

import festim as F
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

avo = 6.022e23

mesh_data = gmshio.read_from_msh(
    "gmsh_files/testing_DIVMON.msh", MPI.COMM_WORLD, 0, gdim=3
)

mesh = mesh_data.mesh
assert mesh_data.facet_tags is not None
facet_tags = mesh_data.facet_tags
facet_tags.name = "Facet markers"

assert mesh_data.cell_tags is not None
cell_tags = mesh_data.cell_tags
cell_tags.name = "Cell markers"

my_model = F.HydrogenTransportProblem()

Deuterium = F.Species("D")
Tritium = F.Species("T")
my_model.species = [Deuterium, Tritium]

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
)

copper = F.Material(
    D_0={"D": float(Cu_D_0_D), "T": (Cu_D_0_T)},
    E_D={"D": float(Cu_E_D_D), "T": (Cu_E_D_T)},
    K_S_0=3.14e24/avo,
    E_K_S=0.572,
    thermal_conductivity=350,
)

cucrzr = F.Material(
    D_0={"D": float(CuCrZr_D_0_D), "T": (CuCrZr_D_0_T)},
    E_D={"D": float(CuCrZr_E_D_D), "T": (CuCrZr_E_D_T)},
    K_S_0=4.28e23, 
    E_K_S=0.387, 
    thermal_conductivity=350,
)

my_model.mesh = F.Mesh(mesh)

my_model.facet_meshtags = facet_tags
my_model.volume_meshtags = cell_tags

W_volume = F.VolumeSubdomain(id=227, material=tungsten)
Cu_volume = F.VolumeSubdomain(id=228, material=copper)
CuCrZr_volume = F.VolumeSubdomain(id=229, material=cucrzr)

top = F.SurfaceSubdomain(id=230,)
bottom = F.SurfaceSubdomain(id=232,)
W_sides = F.SurfaceSubdomain(id=231,)
Cu_sides = F.SurfaceSubdomain(id=236,)
CuCrZr_sides = F.SurfaceSubdomain(id=237,)
W_Cu_interlayer = F.SurfaceSubdomain(id=233,)
Cu_CuCrZr_interlayer = F.SurfaceSubdomain(id=234,)
coolant_face = F.SurfaceSubdomain(id=235,)

all_subdomains = [top, bottom, W_sides, Cu_sides, CuCrZr_sides, W_Cu_interlayer, Cu_CuCrZr_interlayer, coolant_face, W_volume, Cu_volume, CuCrZr_volume]

import ufl
phi = ((0.23e24) / 2) /avo
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

my_model.subdomains = all_subdomains

my_model.temperature = 1000

my_model.settings = F.Settings(
    transient=True,
    atol=1e-9,
    rtol=1e-9,
    final_time=3.2e7, 
)
my_model.settings.stepsize = F.Stepsize(
    initial_value=10000,
    growth_factor=1.1,
    cutback_factor=0.9,
    target_nb_iterations=4,
)

my_model.initialise()
my_model.run()



from dolfinx import plot
import pyvista

# DEUTERIUM
Deuterium_concentration = Deuterium.post_processing_solution

topology, cell_types, geometry = plot.vtk_mesh(Deuterium_concentration.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["c of D"] = Deuterium_concentration.x.array.real
u_grid.set_active_scalars("c of D")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration_of_D.png")

from dolfinx import plot
# ---------------------------------

# TRITIUM
Tritium_concentration = Tritium.post_processing_solution

topology, cell_types, geometry = plot.vtk_mesh(Tritium_concentration.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["c of T"] = Tritium_concentration.x.array.real
u_grid.set_active_scalars("c of T")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration_of_T.png")

from dolfinx import plot
# --------------------------------