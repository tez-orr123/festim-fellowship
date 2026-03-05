# THIS IS RUNNING THROUGH THE MONOBLOCK EXAMPLE IN THE WORKSHOP
# but I am using my own mesh cos I don't have their CAD

# --------------
# Import mesh
# --------------
import meshio
import festim as F
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

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
# ----------------
# Define materials
#-----------------
tungsten = F.Material(
    D_0=4.1e-7,
    E_D=0.39,
    K_S_0=1.87e24,
    E_K_S=1.04,
    thermal_conductivity=100,
)

copper = F.Material(
    D_0=6.6e-7,
    E_D=0.387,
    K_S_0=3.14e24,
    E_K_S=0.572,
    thermal_conductivity=350,
)

cucrzr = F.Material(
    D_0=3.92e-7, 
    E_D=0.418, 
    K_S_0=4.28e23, 
    E_K_S=0.387, 
    thermal_conductivity=350
)

# --------------
# Define subdomains and all
# --------------
my_model.mesh = F.Mesh(mesh)

my_model.facet_meshtags = facet_tags
my_model.volume_meshtags = cell_tags

# model_mesh.mesh.geometry.x[:] *= 1e-3 # converts mm to m 
# # maybe helpful in the future but not really right now

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
# They didn't even define the interfaces in the example yet so we won't

all_subdomains = [top, bottom, W_sides, Cu_sides, CuCrZr_sides, W_Cu_interlayer, Cu_CuCrZr_interlayer, coolant_face, W_volume, Cu_volume, CuCrZr_volume]

# ----------------
# Hydrogen transport problem
# ----------------

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
    bottom: W_volume
}

penalty_term = 1e24
my_model.interfaces = [
    F.Interface(
        id=233, subdomains=(W_volume, Cu_volume), penalty_term=penalty_term
        ),
    F.Interface(id=234, subdomains=(Cu_volume, CuCrZr_volume), penalty_term=penalty_term)
]

import ufl
phi = 1.61e22
R_p = 9.52e-10

my_model.boundary_conditions = [F.FixedConcentrationBC(
    subdomain=top,
    value=lambda T: phi * R_p / (tungsten.D_0 * ufl.exp(-tungsten.E_D / F.k_B / T)),
    species=H,
),
F.FixedConcentrationBC(subdomain=coolant_face, value=0, species=H)]

# implantation_flux_top = F.FixedConcentrationBC(
#     subdomain=top,
#     value=lambda T: phi * R_p / (tungsten.D_0 * ufl.exp(-tungsten.E_D / F.k_B / T)),
#     species=H,
# )

# Changed this so the concentration is set to 0 only at the bottom and at the
# coolant surface as is realistic to model if H is being swept away
recombination_fluxes = [
    F.FixedConcentrationBC(subdomain=surf, value=0, species=H)
    for surf in [
        bottom,
        coolant_face,
    ]
]

# my_model.boundary_conditions = [implantation_flux_top] + recombination_fluxes

my_model.temperature = 1200

my_model.settings = F.Settings(
    atol=1e10,
    rtol=1e-10,
    transient=False,
    max_iterations=10,
)

my_model.initialise()
my_model.run()

# Okay cool it runs with some jigging around of the problem
# and the mesh, maybe the mesh wasn't being read at the right time before is all lol

# --------------------
# POST-PROCESSING
# --------------------
from dolfinx import plot
import pyvista

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

# This makes a lovely model in pyvista that resembles a typical tritium
# transport model I made back in day.