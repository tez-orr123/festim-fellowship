# hydrogen trapping can be represented as a reaction too. 

# "reactions" section

# Chemical reactions between species are defined with Reaction objects

# SIMPLE REACTION:
# Say A + B -> C

import festim as F
import numpy as np

my_model = F.HydrogenTransportProblem()

my_model.mesh = F.Mesh1D(np.linspace(0, 1, 100))

left_surf = F.SurfaceSubdomain1D(id=1, x=0)
right_surf = F.SurfaceSubdomain1D(id=2, x=1)

# Assumes same diffusivity for all species
# I am confused if the whole point of having multispecies is so that they diffuse 
# differently ? No ?
material = F.Material(D_0=1, E_D=0)

vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=material)

my_model.subdomains = [vol, left_surf, right_surf]

# NOW, your reaction object needs a list of REACTANTS, a list of PRODUCTS
# AND forward backwards reaction rates expressed as Arrhenius laws

# Define our three species, A, B, and C
# C is immobile

A = F.Species("A")
B = F.Species("B")
C = F.Species("C", mobile=False) # LABEL C AS IMMOBILE
my_model.species = [A, B, C]

my_model.reactions = [
    F.Reaction(
        reactant=[A,B],
        product=[C],
        k_0=0.01, # forward reaction rate is defined as 0.01 here
        E_k=0,
        p_0=0, # Reverse reaction rate is defined as zero here
        E_p=0,
        volume=vol,
    )
]
# TIP: if reaction rate isnt an Arrhenius law but say a constant value, 
# simply set the activation energy E_k or E_p to 0

my_model.boundary_conditions = [
    #A BCs:
    F.FixedConcentrationBC(left_surf, value=10, species=A),
    F.FixedConcentrationBC(right_surf, value=0, species=A),
    #B BCs
    F.FixedConcentrationBC(left_surf, value=0, species=B),
    F.FixedConcentrationBC(right_surf, value=5, species=B)
]

my_model.temperature = 300

my_model.settings = F.Settings(
    atol=1e-10,
    rtol=1e-10,
    final_time=50
)

my_model.settings.stepsize = F.Stepsize(1)

my_model.initialise()
my_model.run()

# VISUALISING
import matplotlib.pyplot as plt


def plot_profile(species, **kwargs):
    index = my_model.species.index(species)
    V0, dofs = my_model.function_space.sub(index).collapse()
    coords = V0.tabulate_dof_coordinates()[:, 0]
    sort_coords = np.argsort(coords)
    c = my_model.u.x.array[dofs][sort_coords]
    x = coords[sort_coords]
    return plt.plot(x, c, **kwargs)


for species in my_model.species:
    plot_profile(species, label=species.name)

plt.xlabel("Position")
plt.ylabel("Concentration")
plt.legend()
#plt.show()


# -----------------------
# TWO-WAY REACTION
# A + B <=> C

# Same species

# Same mesh

# Same surfaces

# Same material

# Same volume

# Different reactions:
# well msotly the same except p_0 is non zero now
my_model.reactions = [
    F.Reaction(
        reactant=[A, B],
        product=[C],
        k_0=0.01,
        E_k=0,
        p_0=0.1, # Backwards reaction rate is non zero
        E_p=0,
        volume=vol
    )
]

# Same BCs

# Same temperature

# Same settings and stepsize

my_model.initialise()
my_model.run()

# VISUALISE TWO WAY REACTION
for species in my_model.species:
    plot_profile(species, label=species.name)

plt.xlabel("Position")
plt.ylabel("Concentration")
plt.legend()
#plt.show()
# ---------------------------

# ---------------------------
# CHAIN REACTION
# A + B <=> C and C -> D

# Add species D
D = F.Species("D")
my_model.species = [A, B, C, D]

# Same mesh, surfaces, volume, subdomains

# Add second reaction:
my_model.reactions = [
    F.Reaction(
        reactant=[A, B],
        product=[C],
        k_0=0.01,
        E_k=0,
        p_0=0.1,
        E_p=0,
        volume=vol
    ),
    F.Reaction(
        reactant=[C],
        product=[D],
        k_0=0.2,
        E_k=0,
        p_0=0,
        E_p=0,
        volume=vol,
    )
]

# Same BCs, temp, settings

my_model.initialise()
my_model.run()
