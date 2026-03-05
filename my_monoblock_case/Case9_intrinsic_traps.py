# Case 9
# Set-up:
# -	Discontinuous 
# -	Transient
# -	Single species
# -	Traps, 1 in W, 1 in Cu, 1 in CuCrZr, intrinsic traps only
# - Temperature gradient

import festim as F
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

avo = 6.02e23  # mol-1

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

my_model = F.HydrogenTransportProblemDiscontinuous()

tungsten = F.Material(
    D_0=1.5e-7,
    E_D=0.265,
    K_S_0=2.7e24 / avo,
    E_K_S=1.14,
    thermal_conductivity=173,
)

copper = F.Material(
    D_0=6.6e-7,
    E_D=0.387,
    K_S_0=3.14e24 / avo,
    E_K_S=0.572,
    thermal_conductivity=350,
)

cucrzr = F.Material(
    D_0=4.8e-7, 
    E_D=0.42, 
    K_S_0=4.27e23 / avo, 
    E_K_S=0.39, 
    thermal_conductivity=320
)

mesh = F.Mesh(mesh)
my_model.mesh = mesh

my_model.facet_meshtags = facet_tags
my_model.volume_meshtags = cell_tags

W_volume = F.VolumeSubdomain(id=227, material=tungsten)
Cu_volume = F.VolumeSubdomain(id=228, material=copper)
CuCrZr_volume = F.VolumeSubdomain(id=229, material=cucrzr)

top = F.SurfaceSubdomain(id=230)
bottom = F.SurfaceSubdomain(id=232)
W_sides = F.SurfaceSubdomain(id=231)
Cu_sides = F.SurfaceSubdomain(id=236)
CuCrZr_sides = F.SurfaceSubdomain(id=237)
W_Cu_interlayer = F.SurfaceSubdomain(id=233)
Cu_CuCrZr_interlayer = F.SurfaceSubdomain(id=234)
coolant_face = F.SurfaceSubdomain(id=235)

all_subdomains = [
    top,
    bottom,
    W_sides,
    Cu_sides,
    CuCrZr_sides,
    W_Cu_interlayer,
    Cu_CuCrZr_interlayer,
    coolant_face,
    W_volume,
    Cu_volume,
    CuCrZr_volume,
]
my_model.subdomains = all_subdomains

my_model.surface_to_volume = {
    top: W_volume,
    coolant_face: CuCrZr_volume,
    W_sides: W_volume,
    Cu_sides: Cu_volume,
    CuCrZr_sides: CuCrZr_volume,
    bottom: W_volume,
}


H = F.Species("H", subdomains=my_model.volume_subdomains)
trapped_H = F.Species("H_trapped", mobile=False, subdomains=my_model.volume_subdomains)

w_density = 6.3e28 / avo
trap_density = 1e25 / avo
empty_trap = F.ImplicitSpecies(n = trap_density, others=[trapped_H])
my_model.species = [H, trapped_H]


my_model.method_interface = "penalty"
penalty_term = 1e-3
my_model.interfaces = [
    F.Interface(id=233, subdomains=(W_volume, Cu_volume), penalty_term=penalty_term),
    F.Interface(
        id=234, subdomains=(Cu_volume, CuCrZr_volume), penalty_term=penalty_term
    ),
]

lattice_length = 1.1e-10  # m
n_solute_per_site = 6
my_model.reactions = [
    F.Reaction(
        reactant=[H, empty_trap],
        product=[trapped_H],
        k_0=(((tungsten.D_0)/((lattice_length)**2 * n_solute_per_site))/avo), # trapping pre-exponential factor k_0 = (1/6) * 1e13 / rho <- from sanjeet task
        E_k=0.265, # trapping activation energy
        p_0=1.2397e11, # detrapping pre-exponential factor
        E_p = 1.3, # detrapping activation energy, p = p_0 exp( - E_p/kT )
        volume=W_volume,
    ),
]

my_model.temperature = lambda x: ((x[1] + 0.0401225) / (3.25e-5))

import ufl

phi = 0.23e24 / avo
R_p = 1.1e-9
my_model.boundary_conditions = [
    F.FixedConcentrationBC(
        subdomain=top,
        value=lambda T: phi * R_p / (tungsten.D_0 * ufl.exp(-tungsten.E_D / F.k_B / T)),
        species=H,
    ),
    F.FixedConcentrationBC(subdomain=coolant_face, value=0, species=H),
]


my_model.temperature = lambda x: (x[1] + 0.0401225) / (3.25e-5)

my_model.settings = F.Settings(
    transient=True,
    atol=1e-15,
    rtol=1e-10,
    final_time=3.2e7,
)
my_model.settings.stepsize = F.Stepsize(
    initial_value=100,
    growth_factor=1.1,
    cutback_factor=0.9,
    target_nb_iterations=4,
)


# from dolfinx.log import LogLevel, set_log_level

# set_log_level(LogLevel.INFO)

my_model.exports = [F.VTXSpeciesExport(field=trapped_H, filename="trapped_H_disc.bp", subdomain=W_volume)]

my_model.initialise()
my_model.run()





