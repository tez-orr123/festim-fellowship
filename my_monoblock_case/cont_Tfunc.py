# THIS IS RUNNING THROUGH THE MONOBLOCK EXAMPLE IN THE WORKSHOP
# but I am using my own mesh cos I don't have their CAD

# --------------
# Import mesh
# --------------
import meshio
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

my_model = F.HydrogenTransportProblem()

# Set-out each species transport parameters for each material
W_D_0_D = 4.1e-7
W_D_0_T = 4.1e-7

W_E_D_D = 0.38
W_E_D_T = 0.39

Cu_D_0_D = 6.6e-7
Cu_D_0_T = 6.6e-7

Cu_E_D_D = 0.377
Cu_E_D_T = 0.387

CuCrZr_D_0_D = 3.92e-7
CuCrZr_D_0_T = 3.92e-7

CuCrZr_E_D_D = 0.408
CuCrZr_E_D_T = 0.418

# ----------------
# Define materials
#-----------------
tungsten = F.Material(
    #D_0=4.1e-7,
    D_0={"D": float(W_D_0_D), "T": (W_D_0_T)},
    #E_D=0.39,
    E_D={"D": float(W_E_D_D), "T": (W_E_D_T)}, # By defining the explicit list here, makes it harder
    # to understand later in the BCs
    K_S_0=1.87e24,
    E_K_S=1.04,
    thermal_conductivity=100,
)

copper = F.Material(
    #D_0=6.6e-7, 
    D_0={"D": float(Cu_D_0_D), "T": (Cu_D_0_T)},
    #E_D=0.387,
    E_D={"D": float(Cu_E_D_D), "T": (Cu_E_D_T)},
    K_S_0=3.14e24,
    E_K_S=0.572,
    thermal_conductivity=350,
)

cucrzr = F.Material(
    #D_0=3.92e-7, 
    D_0={"D": float(CuCrZr_D_0_D), "T": (CuCrZr_D_0_T)},
    #E_D=0.418,
    E_D={"D": float(CuCrZr_E_D_D), "T": (CuCrZr_E_D_T)},
    K_S_0=4.28e23, 
    E_K_S=0.387, 
    thermal_conductivity=350
)

# --------------
# Define subdomains and all
# --------------
my_model.mesh = F.Mesh(mesh)

my_model.facet_meshtags = facet_tags
my_model.volume_meshtags = cell_tags

# model_mesh.mesh.geometry.x[:] *= 1e-3 # converts mm to m 
# # maybe helpful in the future but not really right now

W_volume = F.VolumeSubdomain(id=227, material=tungsten)
Cu_volume = F.VolumeSubdomain(id=228, material=copper)
CuCrZr_volume = F.VolumeSubdomain(id=229, material=cucrzr)

top = F.SurfaceSubdomain(id=230,)
bottom = F.SurfaceSubdomain(id=232,)
W_sides = F.SurfaceSubdomain(id=231,)
Cu_sides = F.SurfaceSubdomain(id=236,)
CuCrZr_sides = F.SurfaceSubdomain(id=237,)
W_Cu_interlayer = F.SurfaceSubdomain(id=233,)
Cu_CuCrZr_interlayer = F.SurfaceSubdomain(id=234,)
coolant_face = F.SurfaceSubdomain(id=235,)

all_subdomains = [top, bottom, W_sides, Cu_sides, CuCrZr_sides, W_Cu_interlayer, Cu_CuCrZr_interlayer, coolant_face, W_volume, Cu_volume, CuCrZr_volume]

# ----------------
# Hydrogen transport problem
# ----------------
D = F.Species("D")
T = F.Species("T")

# BOUNDARY CONDITIONS SHOULD CHANGE TO DIRICHLET SOON
import ufl
phi = (0.23e24) / 2
R_p = 1.1e-9 
my_model.boundary_conditions = [
    F.FixedConcentrationBC(
        subdomain=top,
        value=lambda T: phi * R_p / (W_D_0_D * ufl.exp(-W_E_D_D / F.k_B / T)),
        species=D
    ),
    F.FixedConcentrationBC(
        subdomain=coolant_face, 
        value=0, 
        species=D
    ),
    F.FixedConcentrationBC(
        subdomain=top,
        value=lambda T: phi * R_p / (W_D_0_T * ufl.exp(-W_E_D_T / F.k_B / T)),
        species=T
    ),
    F.FixedConcentrationBC(
        subdomain=coolant_face, 
        value=0, 
        species=T
    ),
]


# my_model.boundary_conditions = [implantation_flux_top] + recombination_fluxes

my_model.temperature = lambda x: ((400 * x[0]) + 2.135) / 0.005
my_model.temperature = 1000

my_model.settings = F.Settings(
    transient=True,
    atol=1e-9,
    rtol=1e-9,
    final_time=3.2e7, 
)
my_model.settings.stepsize = F.Stepsize(
    initial_value=10000,
    growth_factor=1.1,
    cutback_factor=0.9,
    target_nb_iterations=4,
)

my_model.initialise()
my_model.run()

# Okay cool it runs with some jigging around of the problem
# and the mesh, maybe the mesh wasn't being read at the right time before is all lol

# --------------------
# POST-PROCESSING
# --------------------
from dolfinx import plot
import pyvista
