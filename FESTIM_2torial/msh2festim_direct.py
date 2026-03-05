directory = "gmsh/"
filename = "testing_DIVMON.msh"

import festim as F
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

# Creating a simple simulation on my monoblock mesh, 
# Not transient, no temperature gradient, temperature is set to 700K
# boundary conditions will be simple, some values simplified to just 1 or 0
# Just want it to RUN with this mesh in FESTIM2 then will add all the 
# simulation meat to it after

# Can now put .msh files directly into FESTIM with this method:
# -------------------------------------------------------------
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
# -------------------------------------------------------------
# To check with your .geo file or .msh file tags:
import numpy as np

print(f"Cell tags: {np.unique(cell_tags.values)}")
print(f"Facet tags: {np.unique(facet_tags.values)}")

# Running a HYDROGEN TRANSPORT PROBLEM
model_3D = F.HydrogenTransportProblem()

model_3D.mesh = F.Mesh(mesh)

# Then we will run a real problem over it:

# MATERIALS: 
# S_0 is now K_S_0 and E_S is no E_K_S
# No borders defined here since this is from a mesh 
# thermal conductivity not defined here
W = F.Material(
    name = "W",
    D_0=1.5e-7,  
    E_D=0.265,  
)

Cu = F.Material(
    name = "Cu",
    D_0=6.6e-7,
    E_D=0.387,
)

CuCrZr = F.Material(
	name = "CuCrZr", 
	D_0=4.8e-7, 
	E_D=0.42,
)

model_3D.materials = [W, Cu, CuCrZr]

# Subdomain labelling
# This is a new requirement that they are hoping to make automatic in the future
# i.e. the user shouldn't have to define which volumes a surface is apart of,
# the mesh should be giving that information already.
# For now, have to assign facet_meshtags and volume_meshtags to the model as follows:

W_volume = F.VolumeSubdomain(id=227, material=W)
Cu_volume = F.VolumeSubdomain(id=228, material=Cu)
CuCrZr_volume = F.VolumeSubdomain(id=229, material=CuCrZr)


top = F.SurfaceSubdomain(id=230,)
bottom = F.SurfaceSubdomain(id=232,)
W_sides = F.SurfaceSubdomain(id=231,)
Cu_sides = F.SurfaceSubdomain(id=236,)
CuCrZr_sides = F.SurfaceSubdomain(id=237,)
W_Cu_interlayer = F.SurfaceSubdomain(id=233,)
Cu_CuCrZr_interlayer = F.SurfaceSubdomain(id=234,)
coolant_face = F.SurfaceSubdomain(id=235,)


# Passing meshtags to the model directly
model_3D.facet_meshtags = facet_tags
model_3D.volume_meshtags = cell_tags

# SUBDOMAINS MUST BE DEFINED AS ONE LIST
# NOT LIST OF LISTS
model_3D.subdomains = [top, bottom, W_sides, Cu_sides, CuCrZr_sides, W_Cu_interlayer, Cu_CuCrZr_interlayer, coolant_face, W_volume, Cu_volume, CuCrZr_volume]

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


# SPECIES
H = F.Species(name="H", mobile=True)
model_3D.species = [H]

# Arbitrary temperature for now, will add the proper gradient and 
# thermal conductivities once simple simulation done
model_3D.temperature = 500

# Boundary conditions
model_3D.boundary_conditions = [
    F.DirichletBC(subdomain=top, value=1.1e-9, species="H"), 
    F.DirichletBC(subdomain=coolant_face, value=0, species="H"),
]

# Leave source terms out for now
#model_2D.sources = [ 
		#F.Source(value=28.8e6, volume=31, field='T'), # Neutronic heating W: 1.3-28.8 MW/m3
		#F.Source(value=8.1e6, volume=[30, 29], field='T'), # Neutronic heating Cu & CuCrZr: 0.3-8.1 MW/m3
#]
		
# No stepsize necessary


# Settings
model_3D.settings = F.Settings(
    atol=1e-10, rtol=1e-10, transient=False
)

print("type(formulation) =", type(model_3D.formulation), "value =", getattr(model_3D, "formulation", None))

model_3D.initialise()
model_3D.run()

# POST-PROCESSING
from dolfinx import plot
import pyvista
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

# Just a simple simulation, should run

# ERROR LOG:
#
# AttributeError: 'NonlinearProblem' object has no attribute '_snes'
# This error was caused by the species.subdomains section not accepting
# a list of lists, it needs to just be ONE list straight into it!!

# CHANGE LOG:
#
# Making a much finer mesh:
# To do so, need to edit the .geo file and then re-save the .msh 
# from 14278 to 478733 elements
# Runs, just doesn't look like a very fine mesh at the surfaces
#
# Adding my actual parameters
# Diffusivities, solubilitites
# Tomorrow add boundary conditions and thermal conductivity and the heat gradient