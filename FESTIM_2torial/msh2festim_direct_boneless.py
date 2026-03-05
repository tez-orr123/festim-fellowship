directory = "gmsh/"
filename = "testing_DIVMON.msh"

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

import numpy as np

print(f"Cell tags: {np.unique(cell_tags.values)}")
print(f"Facet tags: {np.unique(facet_tags.values)}")


model_3D = F.HydrogenTransportProblem()

model_3D.mesh = F.Mesh(mesh)


W = F.Material(
    name = "W",
    D_0=1,  
    E_D=0.2,  
)

Cu = F.Material(
    name = "Cu",
    D_0=1,
    E_D=0.3,
)

CuCrZr = F.Material(
	name = "CuCrZr", 
	D_0=1, 
	E_D=0.4,
)

model_3D.materials = [W, Cu, CuCrZr]



W_volume = F.VolumeSubdomain(id=227, material=W)
Cu_volume = F.VolumeSubdomain(id=228, material=Cu)
CuCrZr_volume = F.VolumeSubdomain(id=229, material=CuCrZr)
#volume_subdomains = [W_volume, Cu_volume, CuCrZr_volume]

top = F.SurfaceSubdomain(id=230,)
bottom = F.SurfaceSubdomain(id=232,)
W_sides = F.SurfaceSubdomain(id=231,)
Cu_sides = F.SurfaceSubdomain(id=236,)
CuCrZr_sides = F.SurfaceSubdomain(id=237,)
W_Cu_interlayer = F.SurfaceSubdomain(id=233,)
Cu_CuCrZr_interlayer = F.SurfaceSubdomain(id=234,)
coolant_face = F.SurfaceSubdomain(id=235,)
#surface_subdomains = [top, bottom, W_sides, Cu_sides, CuCrZr_sides, W_Cu_interlayer, Cu_CuCrZr_interlayer, coolant_face]


model_3D.facet_meshtags = facet_tags
model_3D.volume_meshtags = cell_tags

# IS this issue that I have a list of lists...
model_3D.subdomains = [top, bottom, W_sides, Cu_sides, CuCrZr_sides, W_Cu_interlayer, Cu_CuCrZr_interlayer, coolant_face, W_volume, Cu_volume, CuCrZr_volume]


H = F.Species(name="H", mobile=True)
model_3D.species = [H]

model_3D.temperature = 500

model_3D.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=top, value=1, species="H"), 
    F.DirichletBC(subdomain=coolant_face, value=0, species="H"),
]


model_3D.settings = F.Settings(
    atol=1e-10, rtol=1e-10, transient=False
)

print("type(formulation) =", type(model_3D.formulation), "value =", getattr(model_3D, "formulation", None))

model_3D.initialise()
model_3D.run()
