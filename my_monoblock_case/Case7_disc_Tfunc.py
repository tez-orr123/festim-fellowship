# Case 7
# Set-up:
# -	Discontinuous
# -	Transient
# -	Single species
# -	No traps
# -	Temperature gradient
# -	Changing all atomistic values to moles

import festim as F
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

AVOGADRO = 6.02e23  # mol-1

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

my_model = F.HydrogenTransportProblemDiscontinuous()

tungsten = F.Material(
    D_0=1.5e-7,
    E_D=0.265,
    K_S_0=2.7e24 / AVOGADRO,
    E_K_S=1.14,
    thermal_conductivity=173,
)

copper = F.Material(
    D_0=6.6e-7,
    E_D=0.387,
    K_S_0=3.14e24 / AVOGADRO,
    E_K_S=0.572,
    thermal_conductivity=350,
)

cucrzr = F.Material(
    D_0=4.8e-7, E_D=0.42, K_S_0=4.27e23 / AVOGADRO, E_K_S=0.39, thermal_conductivity=320
)

mesh = F.Mesh(mesh)
my_model.mesh = mesh

my_model.facet_meshtags = facet_tags
my_model.volume_meshtags = cell_tags

W_volume = F.VolumeSubdomain(id=227, material=tungsten)
Cu_volume = F.VolumeSubdomain(id=228, material=copper)
CuCrZr_volume = F.VolumeSubdomain(id=229, material=cucrzr)

top = F.SurfaceSubdomain(id=230)
bottom = F.SurfaceSubdomain(id=232)
W_sides = F.SurfaceSubdomain(id=231)
Cu_sides = F.SurfaceSubdomain(id=236)
CuCrZr_sides = F.SurfaceSubdomain(id=237)
W_Cu_interlayer = F.SurfaceSubdomain(id=233)
Cu_CuCrZr_interlayer = F.SurfaceSubdomain(id=234)
coolant_face = F.SurfaceSubdomain(id=235)

all_subdomains = [
    top,
    bottom,
    W_sides,
    Cu_sides,
    CuCrZr_sides,
    W_Cu_interlayer,
    Cu_CuCrZr_interlayer,
    coolant_face,
    W_volume,
    Cu_volume,
    CuCrZr_volume,
]


my_model.method_interface = "penalty"

my_model.subdomains = all_subdomains
H = F.Species("H", subdomains=my_model.volume_subdomains)
my_model.species = [H]


my_model.surface_to_volume = {
    top: W_volume,
    coolant_face: CuCrZr_volume,
    W_sides: W_volume,
    Cu_sides: Cu_volume,
    CuCrZr_sides: CuCrZr_volume,
    bottom: W_volume,
}

penalty_term = 1e-3
my_model.interfaces = [
    F.Interface(id=233, subdomains=(W_volume, Cu_volume), penalty_term=penalty_term),
    F.Interface(
        id=234, subdomains=(Cu_volume, CuCrZr_volume), penalty_term=penalty_term
    ),
]

import ufl

phi = 0.23e24 / AVOGADRO
R_p = 1.1e-9
my_model.boundary_conditions = [
    F.FixedConcentrationBC(
        subdomain=top,
        value=lambda T: phi * R_p / (tungsten.D_0 * ufl.exp(-tungsten.E_D / F.k_B / T)),
        species=H,
    ),
    F.FixedConcentrationBC(subdomain=coolant_face, value=0, species=H),
]

my_model.subdomains = all_subdomains

my_model.temperature = lambda x: (x[1] + 0.0401225) / (3.25e-5)

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


from dolfinx.log import LogLevel, set_log_level

set_log_level(LogLevel.INFO)

my_model.initialise()
my_model.run()


from dolfinx import plot
import pyvista
from basix.ufl import element
import dolfinx

pyvista.set_jupyter_backend("html")

el = element("Lagrange", mesh.mesh.topology.cell_name(), 3)
V = dolfinx.fem.functionspace(mesh.mesh, el)
temperature = dolfinx.fem.Function(V)

coords = ufl.SpatialCoordinate(temperature.function_space.mesh)
x = coords[0]
y = coords[1]
z = coords[2]

interpolation = temperature.function_space.element.interpolation_points
expr = dolfinx.fem.Expression(((y) + 0.0401225) / (3.25e-5), interpolation)
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


u_plotter = pyvista.Plotter()

for vol in my_model.volume_subdomains:
    sol = H.subdomain_to_post_processing_solution[vol]

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
