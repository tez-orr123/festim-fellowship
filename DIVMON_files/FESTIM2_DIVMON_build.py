import festim as F
import gmsh
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
import festim as F
import gmsh
import numpy as np
import os

# Intiitalising gmsh.api
gmsh.initialize()
gmsh.model.add("DFG 3D")

# Define geometry lengths: (all in mm so change accordingly)
block_length = 25 # X
block_height = 29 # Y
block_thickness = 10 # Z
cu_outer_radius = 8
cu_cucrzr_interface_radius = 7
cucrzr_inner_radius = 5

# Create main channel, here a rectangular box
channel = gmsh.model.occ.addBox(0, 0, 0, block_length, block_height, block_thickness)

# Create the obstacle cylinder inside the channel
outer_cylinder = gmsh.model.occ.addCylinder(block_length/2, block_height/2, 0, 0, 0, block_thickness, cu_outer_radius)
interface_cylinder = gmsh.model.occ.addCylinder(block_length/2, block_height/2, 0, 0, 0, block_thickness, cu_cucrzr_interface_radius)
inner_cylinder = gmsh.model.occ.addCylinder(block_length/2, block_height/2, 0, 0, 0, block_thickness, cucrzr_inner_radius)

# SO i think i want to add the outer radius cylinder, then add the interface cylinder, then cut out the inner radius cylinder
# # Subtract cylinder from channel to get the fluid region
# fluid = gmsh.model.occ.cut([(3, channel)], [(3, inner_cylinder)])
# # fluid = gmsh.model.occ.cut([(3, outer_cylinder)], [(3, inner_cylinder)])

fluid_out, _ = gmsh.model.occ.cut(
    objectDimTags=[(3, channel), (3, outer_cylinder), (3, interface_cylinder)],
    toolDimTags=[(3, inner_cylinder)],
    removeObject=True,   # remove the original channel
    removeTool=True     # set to True to remove the inner cylinder once done with it
)
gmsh.model.occ.synchronize()

# Mark the fluid volume for later: Setting with three different volumes **
volumes = gmsh.model.getEntities(dim=3)
W_marker = 1
Cu_marker = 2
CuCrZr_marker = 3
gmsh.model.addPhysicalGroup(3, [volumes[0][1]], W_marker)
gmsh.model.setPhysicalName(3, W_marker, "W_Volume")

gmsh.model.addPhysicalGroup(3, [volumes[1][1]], Cu_marker)
gmsh.model.setPhysicalName(3, Cu_marker, "Cu_Volume")

gmsh.model.addPhysicalGroup(3, [volumes[2][1]], CuCrZr_marker)
gmsh.model.setPhysicalName(3, CuCrZr_marker, "CUCrZr_Volume")

surfaces = gmsh.model.occ.getEntities(dim=2)
inlet_top, outlet_bottom = None, None

top_surface = 1
bottom_surface = 2
w_walls = 3
cu_sides = 4
cucrzr_sides = 5
cucrzr_pipe_surface = 6
w_cu_interface = 7 
cu_cucrzr_interface = 8


for dim, tag in surfaces:
    com = gmsh.model.occ.getCenterOfMass(dim, tag)
    if np.allclose(com, [block_length / 2, 0, block_thickness / 2]):
        gmsh.model.addPhysicalGroup(dim, [tag], bottom_surface)
        gmsh.model.setPhysicalName(dim, bottom_surface, "H outlet (back wall)")
        inlet_bottom = tag
    elif np.allclose(com, [block_length / 2, block_height, block_thickness / 2]):
        gmsh.model.addPhysicalGroup(dim, [tag], top_surface)
        gmsh.model.setPhysicalName(dim, top_surface, "H inlet (plasma facing wall)")
    # elif np.isclose(com[2], 0) or np.isclose(com[1], B) or \
    #      np.isclose(com[2], H) or np.isclose(com[1], 0):
    #     walls.append(tag)
    # else:
    #     obstacles.append(tag)
# -----------------------------------------------------------------
# THIS BIT TO TEST THE MESH IS LOOKING OKAY

# Synchronize and generate 3D mesh
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)


# Ensure the output folder exists
os.makedirs("gmsh", exist_ok=True)

# Save the mesh in GMSH format for downstream use
gmsh.write("gmsh/DIVMON_test.msh")
# -----------------------------------------------------------------

