# Case 8 but it refined mesh (from SALOME)
# Set-up:
# -	Discontinuous
# -	Transient
# -	Multispecies
# -	No traps
# -	Temperature gradient

import festim as F
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

avo = 6.022e23

import festim as F

my_model = F.HydrogenTransportProblemDiscontinuous()

mesh = F.MeshFromXDMF("SALOME_meshes/my_monoblock_mesh_domains.xdmf", "SALOME_meshes/my_monoblock_mesh_boundaries.xdmf")
my_model.mesh = mesh

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

# have to change all the id tags to the new SALOME ones
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

my_model.subdomains = all_subdomains

Deuterium = F.Species("D", subdomains=my_model.volume_subdomains)
Tritium = F.Species("T", subdomains=my_model.volume_subdomains)
my_model.species = [Deuterium, Tritium]

my_model.method_interface = "penalty"

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

# Temperature is the problem, this is an equation for a mesh
# with different co-ordinates.
# Need to re-write it for the SALOME mesh points...
# This temperature gradient is fixed now but there is still
# zero mesh error so this wasn't the problem
my_model.temperature = lambda x: ((x[1]*28.5714) + 344.4286)

my_model.settings = F.Settings(
    transient=True,
    atol=1e-10,
    rtol=1e-10,
    final_time=3.2e7,
)
my_model.settings.stepsize = F.Stepsize(
    initial_value=5e6,
    growth_factor=1.1,
    cutback_factor=0.9,
    target_nb_iterations=4,
)


my_model.exports = [
        F.VTXSpeciesExport(filename=f"{spe.name}_{subdomain.id}.bp", field=spe, subdomain=subdomain)
        for spe in my_model.species
        for subdomain in my_model.volume_subdomains
        ]

from dolfinx.log import LogLevel, set_log_level

set_log_level(LogLevel.INFO)

my_model.initialise()


my_model.run()

# VISUALISATION:
# -------------------------------------------
# Just the mesh
# -------------------------------------------
ft = mesh.define_surface_meshtags()
ct = mesh.define_volume_meshtags()

from dolfinx import plot
import pyvista

pyvista.set_jupyter_backend("html")

fdim = mesh.mesh.topology.dim - 1
tdim = mesh.mesh.topology.dim
mesh.mesh.topology.create_connectivity(fdim, tdim)
topology, cell_types, x = plot.vtk_mesh(
    mesh.mesh, tdim, ct.indices
)

p = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Cell Marker"] = ct.values
grid.set_active_scalars("Cell Marker")
p.add_mesh(grid, show_edges=True)
if pyvista.OFF_SCREEN:
    figure = p.screenshot("volume_marker.png")
#p.show()

mesh.mesh.topology.create_connectivity(fdim, tdim)
topology, cell_types, x = plot.vtk_mesh(
    mesh.mesh, fdim, ft.indices
)

p = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Facet Marker"] = ft.values
grid.set_active_scalars("Facet Marker")
p.add_mesh(grid, show_edges=True)
if pyvista.OFF_SCREEN:
    figure = p.screenshot("facet_marker.png")
#p.show()
# --------------------------------------------------


# -----------------------------------------
# Temp and concs
# -----------------------------------------
# u_plotter = pyvista.Plotter()

# for vol in my_model.volume_subdomains:
#     sol = Deuterium.subdomain_to_post_processing_solution[vol]

#     topology, cell_types, geometry = plot.vtk_mesh(sol.function_space)
#     u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
#     u_grid.point_data["c"] = sol.x.array.real
#     u_grid.set_active_scalars("c")
#     u_plotter.add_mesh(u_grid, cmap="viridis", show_edges=False)
#     u_plotter.add_mesh(u_grid, style="wireframe", color="white", opacity=0.2)

# u_plotter.view_xy(negative=True)


# if not pyvista.OFF_SCREEN:
#     u_plotter.show()
# else:
#     figure = u_plotter.screenshot("concentration.png")




# ____________________________________________
from dolfinx import plot
import pyvista
from basix.ufl import element
import dolfinx

pyvista.set_jupyter_backend("html") # This line is important...

el = element("Lagrange", mesh.mesh.topology.cell_name(), 3)
V = dolfinx.fem.functionspace(mesh.mesh, el)
temperature = dolfinx.fem.Function(V)

coords = ufl.SpatialCoordinate(temperature.function_space.mesh)
x = coords[0]
y = coords[1]
z = coords[2]

interpolation = temperature.function_space.element.interpolation_points
expr = dolfinx.fem.Expression((((y)*28.5714) + 344.4286), interpolation)
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




# Error Here: ValueError: Empty meshes cannot be plotted. Input mesh has zero points. To allow plotting empty meshes, set `pv.global_theme.allow_empty_mesh = True`
# So the problem isn't actually being simulated?
# The temperature can plot but the hydrogen transport problem has no values??
# iteration 0 has rnorm = 0 and rel_res = inf.....
u_plotter = pyvista.Plotter()

for vol in my_model.volume_subdomains:
    sol = Deuterium.subdomain_to_post_processing_solution[vol]

    topology, cell_types, geometry = plot.vtk_mesh(sol.function_space)
    u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    u_grid.point_data["c of D"] = sol.x.array.real
    u_grid.set_active_scalars("c of D")
    u_plotter.add_mesh(u_grid, cmap="viridis", show_edges=False)
    u_plotter.add_mesh(u_grid, style="wireframe", color="white", opacity=0.2)

u_plotter.view_xy(negative=True)


if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration of D.png")


u_plotter = pyvista.Plotter()

for vol in my_model.volume_subdomains:
    sol = Tritium.subdomain_to_post_processing_solution[vol]

    topology, cell_types, geometry = plot.vtk_mesh(sol.function_space)
    u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    u_grid.point_data["c of T"] = sol.x.array.real
    u_grid.set_active_scalars("c of T")
    u_plotter.add_mesh(u_grid, cmap="viridis", show_edges=False)
    u_plotter.add_mesh(u_grid, style="wireframe", color="white", opacity=0.2)

u_plotter.view_xy(negative=True)


if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration of T.png")

