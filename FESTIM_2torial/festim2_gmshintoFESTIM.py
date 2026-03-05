import festim as F
import gmsh
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
import festim as F
import gmsh
import numpy as np
import os

# Start by initialising the gmsh API
gmsh.initialize()
gmsh.model.add("DFG 3D")

# Then, define the geometry parameters that we desire. length L, breadth, B, height H, cylinder radius r etc.
L = 2.5
B = 0.41
H = 0.41
r = 0.05

# Create main channel, here a rectangular box
channel = gmsh.model.occ.addBox(0, 0, 0, L, B, H)

# I assume the first three numbers are coordinates but that's not clear

# Create the obstacle cylinder inside the channel
cylinder = gmsh.model.occ.addCylinder(0.5, 0, 0.2, 0, B, 0, r)

# Subtract cylinder from channel to get the fluid region
fluid = gmsh.model.occ.cut([(3, channel)], [(3, cylinder)]) # What does the 3 even mean here? 3D??
gmsh.model.occ.synchronize()

# Mark the fluid volume for later (with a seemingly arbitrary or after thought number that doesn't go with the flow of this tutorial)
volumes = gmsh.model.getEntities(dim=3)
fluid_marker = 11
gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid volume")

# Identify and tag boundary surfaces on their center of mass
# This part makes no sense, needs an explanation as to what you're doing and why you're doing it
surfaces = gmsh.model.occ.getEntities(dim=2)
inlet, outlet = None, None
walls, obstacles = [], []

inlet_marker, outlet_marker = 1, 3
wall_marker, obstacle_marker = 5, 7

for dim, tag in surfaces:
    com = gmsh.model.occ.getCenterOfMass(dim, tag)
    if np.allclose(com, [0, B / 2, H / 2]):
        gmsh.model.addPhysicalGroup(dim, [tag], inlet_marker)
        gmsh.model.setPhysicalName(dim, inlet_marker, "Fluid inlet")
        inlet = tag
    elif np.allclose(com, [L, B / 2, H / 2]):
        gmsh.model.addPhysicalGroup(dim, [tag], outlet_marker)
        gmsh.model.setPhysicalName(dim, outlet_marker, "Fluid outlet")
    elif np.isclose(com[2], 0) or np.isclose(com[1], B) or \
         np.isclose(com[2], H) or np.isclose(com[1], 0):
        walls.append(tag)
    else:
        obstacles.append(tag)

# Tagging wall and obstacle surfaces
gmsh.model.addPhysicalGroup(2, walls, wall_marker)
gmsh.model.setPhysicalName(2, wall_marker, "Walls")
gmsh.model.addPhysicalGroup(2, obstacles, obstacle_marker)
gmsh.model.setPhysicalName(2, obstacle_marker, "Obstacle")

# Define mesh size field to refine near the obstacle
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", obstacles)
resolution = r / 10
threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold, "LcMin", resolution)
gmsh.model.mesh.field.setNumber(threshold, "LcMax", 20 * resolution)
gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.5 * r)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", r)

# Optionally refine mesh near inlet
inlet_dist = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(inlet_dist, "FacesList", [inlet])
inlet_thre = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(inlet_thre, "IField", inlet_dist)
gmsh.model.mesh.field.setNumber(inlet_thre, "LcMin", 5 * resolution)
gmsh.model.mesh.field.setNumber(inlet_thre, "LcMax", 10 * resolution)
gmsh.model.mesh.field.setNumber(inlet_thre, "DistMin", 0.1)
gmsh.model.mesh.field.setNumber(inlet_thre, "DistMax", 0.5)

# Apply the minimal field combining both refinement regions
minimum = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [threshold, inlet_thre])
gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

# Synchronize and generate 3D mesh
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)


# Ensure the output folder exists
os.makedirs("gmsh", exist_ok=True)

# Save the mesh in GMSH format for downstream use
gmsh.write("gmsh/mesh3D.msh")


# DOLFINx provides tools to convert GMSH models directly into DOLFINx meshes and associated mesh tags
# which can then be used within FESTIM

model_rank = 0
mesh_data = gmshio.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, model_rank
)