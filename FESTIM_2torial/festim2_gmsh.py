# import festim as F
# import gmsh
# import numpy as np
# import os

# # Start by initialising the gmsh API
# gmsh.initialize()
# gmsh.model.add("DFG 3D")

# # Then, define the geometry parameters that we desire. length L, breadth, B, height H, cylinder radius r etc.
# L = 2.5
# B = 0.41
# H = 0.41
# r = 0.05

# # Create main channel, here a rectangular box
# channel = gmsh.model.occ.addBox(0, 0, 0, L, B, H)

# # I assume the first three numbers are coordinates but that's not clear

# # Create the obstacle cylinder inside the channel
# cylinder = gmsh.model.occ.addCylinder(0.5, 0, 0.2, 0, B, 0, r)

# # Subtract cylinder from channel to get the fluid region
# fluid = gmsh.model.occ.cut([(3, channel)], [(3, cylinder)]) # What does the 3 even mean here? 3D??
# gmsh.model.occ.synchronize()

# # Mark the fluid volume for later (with a seemingly arbitrary or after thought number that doesn't go with the flow of this tutorial)
# volumes = gmsh.model.getEntities(dim=3)
# fluid_marker = 11
# gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
# gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid volume")

# # Identify and tag boundary surfaces on their center of mass
# # This part makes no sense, needs an explanation as to what you're doing and why you're doing it
# surfaces = gmsh.model.occ.getEntities(dim=2)
# inlet, outlet = None, None
# walls, obstacles = [], []

# inlet_marker, outlet_marker = 1, 3
# wall_marker, obstacle_marker = 5, 7

# for dim, tag in surfaces:
#     com = gmsh.model.occ.getCenterOfMass(dim, tag)
#     if np.allclose(com, [0, B / 2, H / 2]):
#         gmsh.model.addPhysicalGroup(dim, [tag], inlet_marker)
#         gmsh.model.setPhysicalName(dim, inlet_marker, "Fluid inlet")
#         inlet = tag
#     elif np.allclose(com, [L, B / 2, H / 2]):
#         gmsh.model.addPhysicalGroup(dim, [tag], outlet_marker)
#         gmsh.model.setPhysicalName(dim, outlet_marker, "Fluid outlet")
#     elif np.isclose(com[2], 0) or np.isclose(com[1], B) or \
#          np.isclose(com[2], H) or np.isclose(com[1], 0):
#         walls.append(tag)
#     else:
#         obstacles.append(tag)

# # Tagging wall and obstacle surfaces
# gmsh.model.addPhysicalGroup(2, walls, wall_marker)
# gmsh.model.setPhysicalName(2, wall_marker, "Walls")
# gmsh.model.addPhysicalGroup(2, obstacles, obstacle_marker)
# gmsh.model.setPhysicalName(2, obstacle_marker, "Obstacle")

# # Define mesh size field to refine near the obstacle
# distance = gmsh.model.mesh.field.add("Distance")
# gmsh.model.mesh.field.setNumbers(distance, "FacesList", obstacles)
# resolution = r / 10
# threshold = gmsh.model.mesh.field.add("Threshold")
# gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
# gmsh.model.mesh.field.setNumber(threshold, "LcMin", resolution)
# gmsh.model.mesh.field.setNumber(threshold, "LcMax", 20 * resolution)
# gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.5 * r)
# gmsh.model.mesh.field.setNumber(threshold, "DistMax", r)

# # Optionally refine mesh near inlet
# inlet_dist = gmsh.model.mesh.field.add("Distance")
# gmsh.model.mesh.field.setNumbers(inlet_dist, "FacesList", [inlet])
# inlet_thre = gmsh.model.mesh.field.add("Threshold")
# gmsh.model.mesh.field.setNumber(inlet_thre, "IField", inlet_dist)
# gmsh.model.mesh.field.setNumber(inlet_thre, "LcMin", 5 * resolution)
# gmsh.model.mesh.field.setNumber(inlet_thre, "LcMax", 10 * resolution)
# gmsh.model.mesh.field.setNumber(inlet_thre, "DistMin", 0.1)
# gmsh.model.mesh.field.setNumber(inlet_thre, "DistMax", 0.5)

# # Apply the minimal field combining both refinement regions
# minimum = gmsh.model.mesh.field.add("Min")
# gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold, inlet_thre])
# gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

# # Synchronize and generate 3D mesh
# gmsh.model.occ.synchronize()
# gmsh.model.mesh.generate(3)


# # Ensure the output folder exists
# os.makedirs("gmsh", exist_ok=True)

# # Save the mesh in GMSH format for downstream use
# gmsh.write("gmsh/mesh3D.msh")

# # ----------------
# # Reading GMSH models with dolfinx to be read in FESTIM
# # ----------------

# from dolfinx.io import gmsh as gmshio
# from mpi4py import MPI

# model_rank = 0
# mesh_data = gmshio.model_to_mesh(
#     gmsh.model, MPI.COMM_WORLD, model_rank
# )

import gmsh
import numpy as np
import os
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

# Can load in a .msh file, I will try my monoblock mesh
mesh_data = gmshio.read_from_msh(
    "gmsh/DIVMON_3D_attempt3.msh", MPI.COMM_WORLD, 0, gdim=3
)
mesh = mesh_data.mesh
assert mesh_data.facet_tags is not None
facet_tags = mesh_data.facet_tags
facet_tags.name = "Facet markers"

assert mesh_data.cell_tags is not None
cell_tags = mesh_data.cell_tags
cell_tags.name = "Cell markers"

print(f"Cell tags: {np.unique(cell_tags.values)}")
print(f"Facet tags: {np.unique(facet_tags.values)}")

from dolfinx import plot
import pyvista

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

fdim = mesh.topology.dim - 1
tdim = mesh.topology.dim
mesh.topology.create_connectivity(fdim, tdim)
topology, cell_types, x = plot.vtk_mesh(mesh, fdim, facet_tags.indices)

p = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Facet Marker"] = facet_tags.values
grid.set_active_scalars("Facet Marker")
p.add_mesh(grid, show_edges=True)
if pyvista.OFF_SCREEN:
    figure = p.screenshot("facet_marker.png")
p.show()

topology, cell_types, x = plot.vtk_mesh(mesh, tdim, cell_tags.indices)
p = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Cell Marker"] = cell_tags.values
grid.set_active_scalars("Cell Marker")
p.add_mesh(grid, show_edges=True)
if pyvista.OFF_SCREEN:
    figure = p.screenshot("cell_marker.png")
p.show()

# Setting up a FESTIM model with this mesh:
# Steady state diffusion equation to solve where D = 1, BCs are dirichlet, c=1 on top, c=2 on bottom, c=3 on obstacle

import festim as F

material_1 = F.Material(D_0=1, E_D=0)
material_2 = F.Material(D_0=0.5, E_D=0)
material_3 = F.Material(D_0=2,E_D=0)

volume_1 = F.VolumeSubdomain(id=227, material=material_1)
volume_2 = F.VolumeSubdomain(id=228, material=material_2)
volume_3 = F.VolumeSubdomain(id=229, material=material_3)

tube_surface = F.SurfaceSubdomain(id=235)
walls = F.SurfaceSubdomain(id=231)
top_surface = F.SurfaceSubdomain(id=230)
bottom_surface = F.SurfaceSubdomain(id=232)
cu_curve_surface = F.SurfaceSubdomain(id=233)
cucrzr_curve_surface = F.SurfaceSubdomain(id=234)
cu_flat_surface = F.SurfaceSubdomain(id=236)
cucrzr_flat_surface = F.SurfaceSubdomain(id=237)

my_model = F.HydrogenTransportProblem()

my_model.mesh = F.Mesh(mesh)

# we need to pass the meshtags to the model directly
my_model.facet_meshtags = facet_tags
my_model.volume_meshtags = cell_tags

my_model.subdomains = [top_surface, bottom_surface, tube_surface, walls, cu_curve_surface, cu_flat_surface, cucrzr_curve_surface, cucrzr_flat_surface, volume_1, volume_2, volume_3]

H = F.Species("H")
my_model.species = [H]

my_model.temperature = 400

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=tube_surface, value=0, species=H),
    F.FixedConcentrationBC(subdomain=top_surface, value=1, species=H),
    F.FixedConcentrationBC(subdomain=bottom_surface, value=2, species=H),
]

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

my_model.initialise()
my_model.run()

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

# importing a CAD in GMSH, can use the method I already know for this but be good to brush up on it
# and check this tutorial method also works for my CADs.
# my CADs started as .dxf files and the tutorial is showing how to convert .step files
# so I'll just follow the tutorial 

import gmsh
import os

gmsh.initialize()

# download cad from https://gitlab.onelab.info/gmsh/gmsh/-/raw/gmsh_4_8_4/tutorial/t20_data.step?inline=false
import requests

if not os.path.exists(os.path.join(os.pardir, "gmsh/t20_data.step")):
    url = "https://gitlab.onelab.info/gmsh/gmsh/-/raw/gmsh_4_8_4/tutorial/t20_data.step?inline=false"
    response = requests.get(url)
    with open("gmsh/t20_data.step", "wb") as f:
        f.write(response.content)

gmsh.model.add("t20")
v = gmsh.model.occ.importShapes("gmsh/t20_data.step")

gmsh.model.occ.synchronize()
volumes = gmsh.model.getEntities(dim=3)
vol_marker = 1
gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], vol_marker)
gmsh.model.setPhysicalName(volumes[0][0], vol_marker, "Volume")

surfaces = gmsh.model.occ.getEntities(dim=2)
gmsh.model.addPhysicalGroup(2, [surfaces[0][1]], 1)
gmsh.model.setPhysicalName(2, 1, "Surf1")

gmsh.model.addPhysicalGroup(2, [surfaces[3][1]], 2)
gmsh.model.setPhysicalName(2, 2, "Surf2")

# Finally, let's specify a global mesh size and mesh the partitioned model:
gmsh.option.setNumber("Mesh.MeshSizeMin", 3)
gmsh.option.setNumber("Mesh.MeshSizeMax", 3)
gmsh.model.mesh.generate(3)
gmsh.write("gmsh/t20.msh")
gmsh.finalize()

model_rank = 0
mesh_data = gmshio.read_from_msh(
    "gmsh/t20.msh", MPI.COMM_WORLD, model_rank, gdim=3
)

mesh = mesh_data.mesh
assert mesh_data.facet_tags is not None
facet_tags = mesh_data.facet_tags
facet_tags.name = "Facet markers"
assert mesh_data.cell_tags is not None
cell_tags = mesh_data.cell_tags
cell_tags.name = "Cell markers"

print(f"Cell tags: {np.unique(cell_tags.values)}")
print(f"Facet tags: {np.unique(facet_tags.values)}")

my_model = F.HydrogenTransportProblem()

my_model.mesh = F.Mesh(mesh)

material = F.Material(D_0=1, E_D=0)

vol = F.VolumeSubdomain(id=1, material=material)

surf1 = F.SurfaceSubdomain(id=1)
surf2 = F.SurfaceSubdomain(id=2)

# we need to pass the meshtags to the model directly
my_model.facet_meshtags = facet_tags
my_model.volume_meshtags = cell_tags

my_model.subdomains = [surf1, surf2, vol]

H = F.Species("H")
my_model.species = [H]

my_model.temperature = 400

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=surf1, value=1, species=H),
    F.FixedConcentrationBC(subdomain=surf2, value=0, species=H),
]

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

my_model.initialise()
my_model.run()

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