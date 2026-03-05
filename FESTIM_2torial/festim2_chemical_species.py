# Learn how to define chemical species in FESTIM and understand the concept of implicit species

# Explicit species are species which concentrations are explicitly governed by a PDE 
# in the governing equations.

# multi-species tutorial

import festim as F

my_model = F.HydrogenTransportProblem()

species_1 = F.Species(name="Species 1", mobile=True)
species_2 = F.Species(name="Species 2", mobile=True)

my_model.species = [species_1, species_2]

my_model = F.HydrogenTransportProblem()

species_1 = F.Species(name="Species 1")
species_2 = F.Species(name="Species 2", mobile=False)
species_3 = F.Species(name="Species 3", mobile=False)

my_model.species = [species_1, species_2] # Why not species_3 defined here? 
# Is this a mistake?? 

# Implicit species
my_model = F.HydrogenTransportProblem()

species_1 = F.Species(name="Species 1")
species_3 = F.Species(name="Species 3", mobile=False)

species_2 = F.ImplicitSpecies(name="Species 2", n=20, others=[species_3])

# only pass explicit species to the model
my_model.species = [species_1, species_3]

def n_fun(x, t):
    return 2 * x[0] + x[1] + 20 * t

species_2 = F.ImplicitSpecies(name="Species 2", n=n_fun, others=[species_3])

# More compact form... okay?
species_2 = F.ImplicitSpecies(
    name="Species 2",
    n=lambda x, t: 2 * x[0] + x[1] + 20 * t,
    others=[species_3],
)

# Complete example:
import festim as F
import numpy as np

my_model = F.HydrogenTransportProblem()

# define our three species:
protium = F.Species("H")
deuterium = F.Species("D")
tritium = F.Species("T")

# set models species
my_model.species = [protium, deuterium, tritium]

# make and define mesh
my_model.mesh = F.Mesh1D(np.linspace(0, 1, 100))

# define subdomains
left_surf = F.SurfaceSubdomain1D(id=1, x=0)
right_surf = F.SurfaceSubdomain1D(id=2, x=1)

# assumes the same diffusivity for all species
material = F.Material(D_0=1, E_D=0)

# define volume and assign these to subdomain
vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=material)

my_model.subdomains = [vol, left_surf, right_surf]

# boundary conditions for each species just all in one go
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

# set temperature
my_model.temperature = 300

# settings and stepsize
my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=100)

my_model.settings.stepsize = F.Stepsize(1)

my_model.initialise()
my_model.run()

# visualising using matplotlib this time not pyvista
import matplotlib.pyplot as plt

def plot_profile(species, **kwargs):
    c = species.post_processing_solution.x.array[:]
    x = species.post_processing_solution.function_space.mesh.geometry.x[:,0]
    return plt.plot(x, c, **kwargs)

for species in my_model.species:
    plot_profile(species, label=species.name)

plt.xlabel('Position')
plt.ylabel('Concentration')
plt.legend()
plt.show()

# multi-species seems pretty simple to implement so let's uhhh just give it a go