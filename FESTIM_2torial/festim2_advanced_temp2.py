import festim as F
import dolfinx
from mpi4py import MPI
import numpy as np
import ufl

# build mesh
nx = ny = 20
mesh_fenics = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
mesh = F.Mesh(mesh=mesh_fenics)

# define transient worthy material with density and thermal conductivity
mat = F.Material(D_0=1, E_D=0.01, thermal_conductivity=3, density=2, heat_capacity=5)

# define subdomains as usual
volume_subdomain = F.VolumeSubdomain(id=1, material=mat)
top_bot = F.SurfaceSubdomain(id=2, locator=lambda x: np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)))
left = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 0.0))
right = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[0], 1.0))
subdomains = [volume_subdomain, top_bot, left, right]

# define your problems, heat and hydrotransport problem
heat_transfer_model =F.HeatTransferProblem()
hydrogen_transport_model = F.HydrogenTransportProblem()

# define the mesh for each problem
heat_transfer_model.mesh = mesh
hydrogen_transport_model.mesh = mesh

# define the species for the hydrogen transport problem
H = F.Species("H")
hydrogen_transport_model.species = [H]

# define boundary conditions for each problem
fixed_temperature_left = F.FixedTemperatureBC(
    subdomain=left, value=lambda x: 350 + 20 * ufl.cos(x[0]) * ufl.sin(x[1])
)

heat_transfer_model.boundary_conditions = [
    fixed_temperature_left,
]

hydrogen_transport_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=top_bot, value=1, species=H),
    F.FixedConcentrationBC(subdomain=left, value=0, species=H)
]

# define each problems subdomains
heat_transfer_model.subdomains = subdomains
hydrogen_transport_model.subdomains = subdomains

# define each problems settings
hydrogen_transport_model.settings = F.Settings(
    transient=True,
    atol=1e-09,
    rtol=1e-09,
    stepsize=1,
    final_time=50
)

heat_transfer_model.settings = F.Settings(
    transient=True,
    atol=1e-09,
    rtol=1e-09,
    stepsize=1,
    final_time=50
)

# Now we have the two problems set up individually, we couple them by
# using CoupledTransientHeatTransferHydrogenTransport

problem = F.CoupledTransientHeatTransferHydrogenTransport(
    heat_problem=heat_transfer_model,
    hydrogen_problem=hydrogen_transport_model
)

problem.initialise()
problem.run()

# -----------------------------
import pyvista
from dolfinx import plot

T = problem.heat_problem.u
c = problem.hydrogen_problem.species[0].post_processing_solution

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

    
topology, cell_types, geometry = plot.vtk_mesh(c.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["c"] = c.x.array.real
u_grid.set_active_scalars("c")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, cmap="viridis", show_edges=False)
u_plotter.add_mesh(u_grid, style="wireframe", color="white", opacity=0.2)

contours = u_grid.contour(9)
u_plotter.add_mesh(contours, color="white")

u_plotter.view_xy()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")

