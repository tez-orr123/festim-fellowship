# Case 1:
# Set-up:
# -	Continuous hydrogen transport problem
# -	Transient, over a FPY
# -	Single species
# -	No traps
# -	Constant temperature

import festim as F
from mpi4py import MPI
from dolfinx.io import XDMFFile, gmshio

avo = 6.022e23

mesh, cell_tags, facet_tags = gmshio.read_from_msh(
    "SALOME_meshes/main_monoblock_mesh.msh", MPI.COMM_WORLD, 0, gdim=3
)

mesh.geometry.x[:] *= 1e-3

assert facet_tags is not None
assert cell_tags is not None

facet_tags.name = "Facet Markers"
cell_tags.name = "Cell Markers"

shared_mesh = F.Mesh(mesh)
shared_mesh_facet_tags = facet_tags
shared_mesh_cell_tags = cell_tags

my_model = F.HydrogenTransportProblem()

H = F.Species("H")
my_model.species = [H]

tungsten = F.Material(
    D_0=4.1e-7,
    E_D=0.39,
    K_S_0=1.87e24/avo,
    E_K_S=1.04,
    thermal_conductivity=100,
)

copper = F.Material(
    D_0=6.6e-7, 
    E_D=0.387,
    K_S_0=3.14e24/avo,
    E_K_S=0.572,
    thermal_conductivity=350,
)

cucrzr = F.Material(
    D_0=3.92e-7, 
    E_D=0.418,
    K_S_0=4.28e23/avo, 
    E_K_S=0.387, 
    thermal_conductivity=350
)

my_model.mesh = F.Mesh(mesh)

my_model.facet_meshtags = facet_tags
my_model.volume_meshtags = cell_tags

W_volume = F.VolumeSubdomain(id=1, material=tungsten)
Cu_volume = F.VolumeSubdomain(id=2, material=copper)
CuCrZr_volume = F.VolumeSubdomain(id=3, material=cucrzr)

top = F.SurfaceSubdomain(id=4,)
bottom = F.SurfaceSubdomain(id=6,)
W_sides = F.SurfaceSubdomain(id=5,)
Cu_sides = F.SurfaceSubdomain(id=7,)
CuCrZr_sides = F.SurfaceSubdomain(id=8,)
W_Cu_interlayer = F.SurfaceSubdomain(id=11,)
Cu_CuCrZr_interlayer = F.SurfaceSubdomain(id=12,)
coolant_face = F.SurfaceSubdomain(id=10,)

all_subdomains = [top, bottom, W_sides, Cu_sides, CuCrZr_sides, W_Cu_interlayer, Cu_CuCrZr_interlayer, coolant_face, W_volume, Cu_volume, CuCrZr_volume]

import ufl
phi = ((0.23e24)) /avo
R_p = 1.1e-9 
my_model.boundary_conditions = [
    F.FixedConcentrationBC(
        subdomain=top,
        value=lambda T: phi * R_p / (tungsten.D_0 * ufl.exp(-tungsten.E_D / F.k_B / T)),
        species=H
    ),
    F.FixedConcentrationBC(
        subdomain=coolant_face, 
        value=0, 
        species=H
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

# Modelling Mesh
tdim = mesh.topology.dim

mesh.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(mesh, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
# plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("mesh.png")
# ---------------------------------------------

# Modelling H concentration
hydrogen_concentration = H.post_processing_solution

topology, cell_types, geometry = plot.vtk_mesh(hydrogen_concentration.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["c"] = hydrogen_concentration.x.array.real
u_grid.set_active_scalars("c")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")

from dolfinx import plot
# -----------------------------------------------------






