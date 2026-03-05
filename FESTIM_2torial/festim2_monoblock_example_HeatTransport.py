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
mesh = F.Mesh(mesh)

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
# Define problem
# ----------------
heat_transfer_problem = F.HeatTransferProblem()
heat_transfer_problem.subdomains = all_subdomains
heat_transfer_problem.mesh = mesh

# ---------------------
# THEN GO BACK AND ADD THE FACET AND CELL TAGS TO THE MODEL 
# YOU HAVE JUST CREATED
# ---------------------
heat_transfer_problem.facet_meshtags = facet_tags
heat_transfer_problem.volume_meshtags = cell_tags

# ----------------
# Boundary conditions
# ----------------
heat_flux_top = F.HeatFluxBC(subdomain=top, value=10e6)
convective_flux_coolant = F.HeatFluxBC(subdomain=coolant_face, value=lambda T: 7e04 * (323 - T)) # What value is this??

heat_transfer_problem.boundary_conditions = [heat_flux_top, convective_flux_coolant]

# ---------------
# Exports and settings
# ---------------
heat_transfer_problem.exports = [F.VTXTemperatureExport("out.bp")]

heat_transfer_problem.settings = F.Settings(
    atol=1e-10,
    rtol=1e-10,
    transient=False,
)

heat_transfer_problem.initialise()
heat_transfer_problem.run()

# The simulation runs so this is with my CAD and the
# tutorials example, 
# THIS SET-UP WORKS

# NOW TRY AND CREATE H TRANSPORT PROBLEM IN A DIFFERENT SCRIPT

# -------------------
# POST-PROCESSING
# -------------------

from dolfinx import plot
import pyvista

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
