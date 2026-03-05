import festim as F
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

# CHEMICAL SPECIES

# explicit species = species which concentrations are explicitly 
# governed by a PDE in the governing equations, they also mobile species

model = F.HydrogenTransportProblem()

species_1 = F.Species(name = "Species 1", mobile=True)
species_2 = F.Species(name = "Species 2", mobile=True)

model.species = [species_1, species_2]

# Can set species to be immobile and also have implicit species, will go into if needed


# A complete problem:
import festim as F
import numpy as np

my_model = F.HydrogenTransportProblem()

protium = F.Species("H")
deuterium = F.Species("D")
tritium = F.Species("T")
my_model.species = [protium, deuterium, tritium]

my_model.mesh = F.Mesh1D(np.linspace(0, 1, 100))

left_surf = F.SurfaceSubdomain1D(id=1, x=0)
right_surf = F.SurfaceSubdomain1D(id=2, x=1)

# assumes the same diffusivity for all species
material = F.Material(D_0=1, E_D=0)

vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=material)

my_model.subdomains = [vol, left_surf, right_surf]

my_model.boundary_conditions = [
    # Protium BCs
    F.FixedConcentrationBC(left_surf, value=10, species=protium),
    F.FixedConcentrationBC(right_surf, value=0, species=protium),
    # Deuterium BCs
    F.FixedConcentrationBC(left_surf, value=5, species=deuterium),
    F.FixedConcentrationBC(right_surf, value=0, species=deuterium),
    # Tritium BCs
    F.FixedConcentrationBC(left_surf, value=0, species=tritium),
    F.FixedConcentrationBC(right_surf, value=2, species=tritium),
]

my_model.temperature = 300

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=100)

my_model.settings.stepsize = F.Stepsize(1)
my_model.initialise()
my_model.run()




