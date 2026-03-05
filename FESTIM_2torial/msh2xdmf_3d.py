import meshio
import numpy as np

directory = "gmsh/"
filename = "testing_DIVMON"
# The .msh file needs to come with respective .geo file and if you read the .geo file, it makes it clear what each surface and point is labelled as
# with both name and number. This can also be edited to change any surface to have a different number.

msh = meshio.read(directory + filename + ".msh")

for cell in msh.cells:
    print(cell.type)


# Initialize lists to store cells and their corresponding data
tetra_cells_list = []
triangle_cells_list = []

tetra_data_list = []
triangle_data_list = []


# Extract cell data for all types
for cell in msh.cells:
    if cell.type == "tetra":
        tetra_cells_list.append(cell.data)
    elif cell.type == "triangle":
        triangle_cells_list.append(cell.data)
  

# Extract physical tags
for key, data in msh.cell_data_dict["gmsh:physical"].items():
    if key == "tetra":
        tetra_data_list.append(data)
    elif key == "triangle":
        triangle_data_list.append(data)




# Concatenate all triangular cells and their data
triangle_cells = np.concatenate(triangle_cells_list)
triangle_data = np.concatenate(triangle_data_list)

# Concatenate all tetra cells and their data
tetra_cells = np.concatenate(tetra_cells_list)
tetra_data = np.concatenate(tetra_data_list)


# Create the triangular mesh for the surface
triangle_mesh = meshio.Mesh(
    points=msh.points,
    cells=[("triangle", triangle_cells)],
    cell_data={"f": [triangle_data]},
)

# Create the tetrahedral mesh for the surface
tetra_mesh = meshio.Mesh(
    points=msh.points,
    cells=[("tetra", tetra_cells)],
    cell_data={"f": [tetra_data]},
)

# Write the mesh files
meshio.write(directory + filename + "_volume_mesh.xdmf", tetra_mesh)
meshio.write(directory + filename + "_surface_mesh.xdmf", triangle_mesh)

import festim as F 

model_3D = F.HydrogenTransportProblemDiscontinuous()

model_3D.mesh = F.MeshFromXDMF(volume_file = directory + filename + "_volume_mesh.xdmf" , facet_file = directory + filename + "_surface_mesh.xdmf")

# Then we will run a real problem over it:
tungsten = F.Material(
    name = "W",
    D_0=1.5e-07,  # m2/s
    E_D=0.265,  # eV
    K_S_0=2.7e24,
    E_K_S=1.14,
    #thermal_cond=173,  # W/mK
)

copper = F.Material(
    name = "Cu",
    D_0=6.6e-7,
    E_D=0.387,
    K_S_0=3.14e24,
    E_K_S=0.572,
    #thermal_cond=350,
)

cucrzr = F.Material(
	name = "CuCrZr", 
	D_0=4.8e-7, 
	E_D=0.42, 
	K_S_0=4.27e23, 
	E_K_S=0.39,
	#thermal_cond=320
)

model_3D.materials = [tungsten, copper, cucrzr]

# ----------------------------------------------------------------------
# WILL LEAVE UNTRAPPED FOR NOW

# w_density = 6.3e28  # atom/m3  
# copper_density = 9.2e28
# cucrzr_density = 9.2e28  

# damage_percent = 0.01 # 0.1-2 dpa/fpy W neutronic damage

# trap_W_1 = F.Trap( 
# 	k_0 = [1.5e-7/((1.1e-10)**2 * 6 * w_density),6.6e-7 / ((3.6e-10)**2 * 1 * copper_density),4.8e-7 / ((3.6e-10)**2 * 1 * cucrzr_density)],
# 	E_k= [0.265,0.387,0.418],
# 	p_0=[1.2397e13,5.0926e12,7.3472e12],
# 	E_p=[0.83,0.5,0.53], 
# 	density = [w_density*0.00118,copper_density*0.00005,3.7e24],
# 	materials = [tungsten,copper,cucrzr]
# )
            
# trap_W_2 = F.Trap(
# 	k_0 = 1.5e-7/((1.1e-10)**2 * 6 * w_density),
# 	E_k= 0.265, 
# 	p_0=1.2397e13,
# 	E_p=0.97,
# 	density = w_density*0.000722,
# 	materials = [tungsten]
# )

# trap_W_3 = F.Trap(
# 	k_0=1.5e-7/(1.1e-10**2 * 6 * w_density),
# 	E_k=0.265,
# 	p_0=1e13,
# 	E_p=1.51, # change to 2.05 when necessary
# 	density = damage_percent*w_density,
# 	materials = tungsten
# ) 
# model_3D.traps = [trap_W_1, trap_W_2, trap_W_3]
# ----------------------------------------------------------------------------

# FESTIM 2 has this bit where you need to define subdomains, volumes and interfaces with name tags:
W_volume = F.VolumeSubdomain(id=227, material=tungsten)
Cu_volume = F.VolumeSubdomain(id=228, material=copper)
CuCrZr_volume = F.VolumeSubdomain(id=229, material=cucrzr)

top_surface = F.SurfaceSubdomain(id=32)
bottom_surface = F.SurfaceSubdomain(id=33)


model_3D.subdomains = [top_surface, bottom_surface]

H = F.Species("H")
model_3D.species = [H]
for species in model_3D.species:
    species_subdomains = [227, 228, 229] # Get volume numbers from the .geo file

model_3D.temperature = 700

model_3D.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=230, value=1, species="H"), # mean depth may be too low, maybe order of magnitude higher
    F.DirichletBC(subdomain=235, value=0, species="H"),
    F.DirichletBC(subdomain=230, value=1073, species="T"),
    F.DirichletBC(subdomain=235, value=773, species="T"), # Set coolant temp to 573K as 473K messes it up for some reason
]

# Leave source terms out for now
#model_2D.sources = [ 
		#F.Source(value=28.8e6, volume=31, field='T'), # Neutronic heating W: 1.3-28.8 MW/m3
		#F.Source(value=8.1e6, volume=[30, 29], field='T'), # Neutronic heating Cu & CuCrZr: 0.3-8.1 MW/m3
#]
		
# No stepsize necessary


# Settings
model_3D.settings = F.Settings(
    atol=1e10, rtol=1e-10, transient=False
)

model_3D.initialise()
model_3D.run()