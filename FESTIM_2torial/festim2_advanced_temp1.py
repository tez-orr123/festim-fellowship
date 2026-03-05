import festim as F

# Use FESTIMs internal solver .HeatTransferProblem for multiphysics coupling.
# Couple transient hydrogen transport and heat-transfer simulations

heat_transfer_model = F.HeatTransferProblem()

# DEFINING THERMAL CONDUCTIVITY FUNCTION, LAMBDA
def thermal_cond_function(T):
    return 3 + 0.1 * T

mat = F.Material(D_0=1, E_D=0.1, thermal_conductivity=thermal_cond_function)

# ADD HEAT SOURCES AND FIXED AND FLUX HEAT BC'S
import dolfinx
from mpi4py import MPI
import numpy as np

nx = ny = 20
mesh_fenics = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
heat_transfer_model.mesh = F.Mesh(mesh=mesh_fenics)

volume_subdomain = F.VolumeSubdomain(id=1, material=mat)

top_bot = F.SurfaceSubdomain(id=2, locator=lambda x: np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)))
left = F.SurfaceSubdomain(id=3, locator=lambda x: np.isclose(x[0], 0.0))
right = F.SurfaceSubdomain(id=4, locator=lambda x: np.isclose(x[0], 1.0))

heat_transfer_model.subdomains = [volume_subdomain, top_bot, left, right]

heat_transfer_model.sources = [
    F.HeatSource(value=lambda x: 1 + 0.1 * x[0], volume=volume_subdomain)
]

import ufl

fixed_temp_left = F.FixedTemperatureBC(
    subdomain=left, value=lambda x: 350 + 20 * ufl.cos(x[0]) * ufl.sin(x[1])
)

def h_coeff(x):
    return 100 * x[0]

def T_ext(x):
    return 300 + 3 * x[1]

convective_heat_transfer = F.HeatFluxBC(
    subdomain=top_bot, value=lambda  x, T: h_coeff(x) * (T_ext(x) - T)
)

heat_flux = F.HeatFluxBC(
    subdomain=right, value=lambda x: 10 + 3 * ufl.cos(x[0]) + ufl.sin(x[1])
)

heat_transfer_model.boundary_conditions = [
    fixed_temp_left,
    convective_heat_transfer,
    heat_flux,
]

heat_transfer_model.settings = F.Settings(
    transient=False,
    atol=1e-9,
    rtol=1e-9,
)

heat_transfer_model.initialise()
heat_transfer_model.run()

# -------------------

import pyvista

from dolfinx import plot

T = heat_transfer_model.u

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

# HEAT TRANSFER PROBLEM WORKS FINE AND CAN BE VISUALISED
#
# NOW WE RUN A HYDROGEN TRANSPORT SIMULATION AND COUPLE THEM
#
# we define our hydrogen problem using HydrogenTransportProblem and simply
# assign the ouput from the heat transfer simulation to the temperature 
# attribute of our hydrogen simulation

hydrogen_problem = F.HydrogenTransportProblem()

hydrogen_problem.mesh = heat_transfer_model.mesh
H = F.Species("H")
hydrogen_problem.species = [H]
hydrogen_problem.temperature = heat_transfer_model.u

hydrogen_problem.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=left, value=1, species=H),
    F.FixedConcentrationBC(subdomain=right, value=0, species=H),
]

hydrogen_problem.subdomains = heat_transfer_model.subdomains

hydrogen_problem.settings = F.Settings(
    transient = False,
    atol = 1e-9,
    rtol = 1e-9,
)

hydrogen_problem.initialise()
hydrogen_problem.run()

# -----------------------------

from dolfinx import plot

c = hydrogen_problem.u

topology, cell_types, geometry = plot.vtk_mesh(T.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["c"] = c.x.array.real
u_grid.set_active_scalars("c")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, cmap="viridis", show_edges=False)

u_plotter.view_xy()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("temperature.png")

# STEADY-STATE COUPLING OF HEAT TRANSFER AND HYDROGEN TRANSPORT PROBLEMS DONE SUCCESSFULLY
#
# NOW TO MAKE HEAT TRANSFER TRANSIENT
# we must add density and heat_capacity 

mat = F.Material(D_0=1, E_D=0.01, thermal_conductivity=thermal_cond_function, density=20, heat_capacity=50)
volume_subdomain = F.VolumeSubdomain(id=1, material=mat)

heat_transfer_model.subdomains = [volume_subdomain, top_bot, left, right]

heat_transfer_model.settings = F.Settings(
    atol=1e-9,
    rtol=1e-9,
    stepsize=1,
    final_time=5
)

heat_transfer_model.initialise()
heat_transfer_model.run()

# ----------------------------

T = heat_transfer_model.u

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

# NOW WANT TO COUPLE HEAT AND H TRANSPORT IN A TRANSIENT PROBLEM
# this requires a whole new set-up of the problem so will do in a new file