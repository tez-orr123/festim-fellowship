# Space dependent function: use lambda functions:
import festim as F
import ufl

T0 = 300
my_model = F.HydrogenTransportProblem()
my_model.temperature = lambda x: T0 * ufl.exp(-x[0])  


# time dependent would be lamda t:
# my_model.temperature = lambda t: T0 * ufl.sin(t)

# For function in x, y, z space:
my_model.temperature = lambda x: T0 * (ufl.cos(x[0])*ufl.sin(x[1]) - 2*x[2])
# x[0] is x, x[1] is y, x[2] is z

from dolfinx.mesh import create_unit_cube
from mpi4py import MPI
import ufl
import festim as F
import numpy as np

mesh = F.Mesh(create_unit_cube(MPI.COMM_WORLD, 10, 10, 10))
my_model.mesh = mesh

mat = F.Material(D_0=1, E_D=0)

volume = F.VolumeSubdomain(id=1, material=mat)
top_surface = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[2], 1.0))
bottom_surface = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[2], 0.0))
my_model.subdomains = [top_surface, bottom_surface, volume]

H = F.Species("H")
my_model.species = [H]

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=top_surface, value=1.0, species=H),
    F.FixedConcentrationBC(subdomain=bottom_surface, value=0.0, species=H),
]

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

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
expr = dolfinx.fem.Expression(T0 * ufl.cos(x)*ufl.sin(y) - 2*z, interpolation)                    
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

pyvista.set_jupyter_backend("html")

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