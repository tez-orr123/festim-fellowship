import festim as F
import numpy as np

# ------------------------------------------------
# PART 1: EXPLICIT AND IMPLICIT TRAPPING SITES
# ------------------------------------------------

# Hydrogen trapping can be represented as a reaction too, here
# mobile hydrogen is reacting with an empty trap to form a trapped hydrogen.
# H + [] <=> [H] 

# Following on from the species tutorial, will show that empty traps
# can be represented either as an explicit or implicit species

# TDS tutorial provides a more complete and realistic example including
# temperature-dependent trapping/detrapping rates 
# So will go through that after this

# ---------------------
# EXPLICIT Trapping Sites
# ---------------------

my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh1D(np.linspace(0, 1, 100))

material = F.Material(D_0=1, E_D=0)

vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=material)

mobile_H = F.Species("H")
trapped_H = F.Species("H_trapped", mobile=False)
empty_traps = F.Species("empty_traps", mobile=False)

my_model.species = [mobile_H, trapped_H, empty_traps]

# Setting intitial conditions:
my_model.initial_conditions = [F.InitialConcentration(value=2, species=empty_traps, volume=vol)]

left_surf = F.SurfaceSubdomain1D(id=1, x=0)
right_surf = F.SurfaceSubdomain1D(id=2, x=1)

my_model.subdomains = [vol, left_surf, right_surf]

my_model.reactions = [
    F.Reaction(
        reactant=[mobile_H, empty_traps],
        product=[trapped_H],
        k_0=0.01,
        E_k=0,
        p_0=0.1, # The detrapping rate is order of magnitude higher than the trapping rate here
        E_p=0,
        volume=vol,
    ),
]

my_model.boundary_conditions = [
    F.FixedConcentrationBC(left_surf, value=10, species=mobile_H),
    F.FixedConcentrationBC(right_surf, value=0, species=mobile_H),
]

my_model.temperature = 300

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=50)

my_model.settings.stepsize = F.Stepsize(1)

my_model.initialise()
my_model.run()

import matplotlib.pyplot as plt


def plot_profile(species, **kwargs):
    index = my_model.species.index(species)
    V0, dofs = my_model.function_space.sub(index).collapse()
    coords = V0.tabulate_dof_coordinates()[:, 0]
    sort_coords = np.argsort(coords)
    c = my_model.u.x.array[dofs][sort_coords]
    x = coords[sort_coords]
    return plt.plot(x, c, **kwargs)

# for species in my_model.species:
#     plot_profile(species, label=species.name)

# plt.xlabel("Position")
# plt.ylabel("Concentration")
# plt.legend()
#plt.show()


# --------------------------
# IMPLICIT Trapping Sites
# --------------------------

# Empty traps can be defined as implicit species with their concentration defined as
# n_empty = n_total - c_t


my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh1D(np.linspace(0, 1, 100))

mobile_H = F.Species("H")
trapped_H = F.Species("H_trapped", mobile=False)
empty_traps = F.ImplicitSpecies(n=2, others=[trapped_H])

my_model.species = [mobile_H, trapped_H]

left_surf = F.SurfaceSubdomain1D(id=1, x=0)
right_surf = F.SurfaceSubdomain1D(id=2, x=1)

material = F.Material(D_0=1, E_D=0)

vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=material)

my_model.subdomains = [vol, left_surf, right_surf]

my_model.reactions = [
    F.Reaction(
        reactant=[mobile_H, empty_traps],
        product=[trapped_H],
        k_0=0.01,
        E_k=0,
        p_0=0.1,
        E_p=0,
        volume=vol
    )
]

my_model.boundary_conditions = [
    F.FixedConcentrationBC(left_surf, value=10, species=mobile_H),
    F.FixedConcentrationBC(right_surf, value=0, species=mobile_H),
]

my_model.temperature = 300

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=50)

my_model.settings.stepsize = F.Stepsize(1)

my_model.initialise()
my_model.run()

for species in my_model.species:
    plot_profile(species, label=species.name)

plt.xlabel("Position")
plt.ylabel("Concentration")
plt.legend()
plt.show()

# Can see that implicit series is used to set-up this 'reaction' between
# hydrogen and traps without actually considering empty traps as a species
# since it's not really anything


