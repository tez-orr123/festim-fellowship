from dolfinx.mesh import create_unit_square
from mpi4py import MPI
from dolfinx import plot
import pyvista
from dolfinx.mesh import create_unit_cube
from dolfinx.mesh import create_rectangle
from scipy.spatial.transform import Rotation

# Create square mesh

nx, ny = 10, 10  # Number of divisions in x and y directions
mesh = create_unit_square(MPI.COMM_WORLD, nx, ny)

print(pyvista.global_theme.jupyter_backend)

tdim = mesh.topology.dim

mesh.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(mesh, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("mesh.png")

# Create cube mesh



nx, ny, nz = 10, 10, 10  # Number of divisions in x, y, and z directions
mesh = create_unit_cube(MPI.COMM_WORLD, nx, ny, nz)

tdim = mesh.topology.dim

mesh.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(mesh, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_isometric()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("mesh.png")

# Create rectangle mesh



mesh = create_rectangle(MPI.COMM_WORLD, [[0, 0], [2, 1]], [20, 10])
tdim = mesh.topology.dim

mesh.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(mesh, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_isometric()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("mesh.png")

# # Adding a celltype to the mesh to above

# # MESH TRANSFORMATIONS
# # Scaling done
# # Translating done
# # Rotating done
# # Rotating by an arbitrary angle 



# # mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)

# # # Define rotation angle in degrees
# # degrees = 130

# # # Create a rotation object for rotation about the z-axis
# # rotation = Rotation.from_euler("z", degrees, degrees=True)

# # # Apply rotation to all mesh coordinates
# # mesh.geometry.x[:, :] = rotation.apply(mesh.geometry.x)

# # tdim = mesh.topology.dim

# # mesh.topology.create_connectivity(tdim, tdim)
# # topology, cell_types, geometry = plot.vtk_mesh(mesh, tdim)
# # grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# # plotter = pyvista.Plotter()
# # plotter.add_mesh(grid, show_edges=True)
# # plotter.view_xy()

# # # show orientation
# # plotter.show_axes()

# # if not pyvista.OFF_SCREEN:
# #     plotter.show()
# # else:
# #     figure = plotter.screenshot("mesh.png")

import festim as F

fenics_mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)

festim_mesh = F.Mesh(fenics_mesh)

import numpy as np

# Instantiate the surface subdomains with unique IDs
top_surface = F.SurfaceSubdomain(id=1, locator = lambda x: np.isclose(x[1], 1.0))
bottom_surface = F.SurfaceSubdomain(id=2, locator = lambda x: np.isclose(x[1], 0.0))

# volume subdomains need a material
material = F.Material(name="test_material", D_0=1, E_D=0)

# Instantiate the volume subdomains, both using the same material
top_volume = F.VolumeSubdomain(id=1, material=material, locator=lambda x: x[1] >= 0.5)
bottom_volume = F.VolumeSubdomain(id=2, material=material, locator=lambda x: x[1] <= 0.5)

my_model = F.HydrogenTransportProblem()
my_model.mesh = festim_mesh

# Assign subdomains (surface and volume)
my_model.subdomains = [top_surface, bottom_surface, top_volume, bottom_volume]

# Generate dolfinx.MeshTags for visualisation and boundary/material definition
my_model.define_meshtags_and_measures()

print(my_model.facet_meshtags)
print(my_model.volume_meshtags)

# Why do this fdim part?
fdim = my_model.mesh.mesh.topology.dim - 1
tdim = my_model.mesh.mesh.topology.dim
my_model.mesh.mesh.topology.create_connectivity(fdim, tdim)
topology, cell_types, x = plot.vtk_mesh(
    my_model.mesh.mesh, tdim, my_model.volume_meshtags.indices
)

# Visualising volume subdomains
p = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Cell Marker"] = my_model.volume_meshtags.values
grid.set_active_scalars("Cell Marker")
p.add_mesh(grid, show_edges=True)

if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure = p.screenshot("volume_markers.png")

# Visualising surface subdomains
my_model.mesh.mesh.topology.create_connectivity(fdim, tdim)
topology, cell_types, x = plot.vtk_mesh(
    my_model.mesh.mesh, fdim, my_model.facet_meshtags.indices
)

p = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Facet Marker"] = my_model.facet_meshtags.values
grid.set_active_scalars("Facet Marker")
p.add_mesh(grid, show_edges=True)

if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure = p.screenshot("facet_markers.png")

# Saving domain tags to a file to be used in other simulations
import adios4dolfinx

my_model.facet_meshtags.name = "facet_tags"

# write
adios4dolfinx.write_meshtags("FESTIM_2torial/facet_tags.bp", my_model.mesh.mesh, my_model.facet_meshtags)

# read
ft = adios4dolfinx.read_meshtags("FESTIM_2torial/facet_tags.bp", my_model.mesh.mesh, meshtag_name="facet_tags")

# -----------------------------------------
# A complete example and Implementation

# imports
import festim as F
import numpy as np

# make a fenics mesh
fenics_mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)

# import to FESTIM
festim_mesh = F.Mesh(fenics_mesh)

# define materials
material_top = F.Material(D_0=1, E_D=0)
material_bottom = F.Material(D_0=2, E_D=0)

# define volume subdomains
top_volume = F.VolumeSubdomain(id=1, material=material_top, locator=lambda x: x[1] >= 0.5)
bottom_volume = F.VolumeSubdomain(id=2, material=material_bottom, locator=lambda x: x[1] <= 0.5)

# define surface subdomains
top_surface = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[1], 1.0))
bottom_surface = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[1], 0.0))

# set-up the model
my_model = F.HydrogenTransportProblem()
my_model.mesh = festim_mesh
my_model.subdomains = [top_surface, bottom_surface, top_volume, bottom_volume]

H = F.Species("H")
my_model.species = [H]

my_model.temperature = 400

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=top_surface, value=1.0, species=H),
    F.FixedConcentrationBC(subdomain=bottom_surface, value=0.0, species=H),
]

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

my_model.initialise()
my_model.run()

# Visualising this implementation
h_conc = H.post_processing_solution

topology, cell_types, geometry = plot.vtk_mesh(h_conc.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["c"] = h_conc.x.array.real
u_grid.set_active_scalars("c")

u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges = True)
u_plotter.view_xy()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")