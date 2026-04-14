# Case 3
# Set-up:
# -	Continuous hydrogen transport problem
# -	Transient FPY
# -	Single species, H
# -	No traps
# -	Temperature gradient of 1173K at top surface decreasing down to 773K at centre of pipe hole.

import festim as F
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI

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

mesh = F.Mesh(mesh)
my_model.mesh = mesh

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
phi = (0.23e24) /avo
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

my_model.temperature = lambda x: ((x[1] *1e3 * 28.5714) + 344.4286)
# This corrected temperature equation works!!!!!

my_model.settings = F.Settings(
    transient=True,
    atol=1e-11,
    rtol=1e-10,
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

import dolfinx 
from dolfinx import plot
import pyvista
from basix.ufl import element

# NO HTML PART NO NO NO!!!!

el = element("Lagrange", mesh.mesh.topology.cell_name(), 3)
V = dolfinx.fem.functionspace(mesh.mesh, el)
temperature = dolfinx.fem.Function(V)

coords = ufl.SpatialCoordinate(temperature.function_space.mesh)
x = coords[0]
y = coords[1]
z = coords[2]

# THIS NEEDS () at the end ??? FOR SOME REASON NOW!!!!!!
interpolation = temperature.function_space.element.interpolation_points()
expr = dolfinx.fem.Expression((((y * 1e3) * 28.5714) + 344.4286), interpolation)                
temperature.interpolate(expr)

u_plotter = pyvista.Plotter()
topology, cell_types, geometry = plot.vtk_mesh(V)
function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
function_grid.point_data["T"] = temperature.x.array.real
function_grid.set_active_scalars("T")
u_plotter.add_mesh(function_grid, cmap="inferno", show_edges=False, opacity=1)

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("temperature.png")

c = H.post_processing_solution

topology, cell_types, geometry = plot.vtk_mesh(c.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["c"] = c.x.array.real
u_grid.set_active_scalars("c")
u_plotter = pyvista.Plotter()

u_plotter.add_mesh(u_grid, cmap="viridis", show_edges=False)

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")


