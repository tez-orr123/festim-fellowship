# Annihilation:
# - radioactive decay
# - vacancy-interstitial annihilation

# Reaction objects can also have no products at all, simulates annihilation

# -------------------
# RADIOACTIVE DECAY
# -------------------
# Can deal with radioactive species, use annihilation to simulate decay
# Rate of reaction will be the decay constant
# 'A' will be our decaying radioisotope

import festim as F
import numpy as np

my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh1D(np.linspace(0, 1, 100))

material = F.Material(D_0=1, E_D=0)
vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=material)
left_surf = F.SurfaceSubdomain1D(id=1, x=0)
right_surf = F.SurfaceSubdomain1D(id=2, x=1)

my_model.subdomains = [vol, left_surf, right_surf]

A = F.Species("A")

my_model.species = [A]
left_surf = F.SurfaceSubdomain1D(id=1, x=0)
right_surf = F.SurfaceSubdomain1D(id=2, x=1)

material = F.Material(D_0=1, E_D=0)

vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=material)

my_model.subdomains = [vol, left_surf, right_surf]

my_model.initial_conditions = [F.InitialConcentration(value=1, species=A, volume=vol)]

my_model.reactions = [
    F.Reaction(reactant=[A], k_0=1, E_k=0, volume=vol),
] # no product so no need for a backwards reaction rate

my_model.temperature = 300

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=1)

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

x = my_model.mesh.mesh.geometry.x[:, 0]
c = A.post_processing_solution.x.array[:]

plt.plot(x, c, label=A.name)
plt.axhline(2, linestyle="--", color="green", label="Initial concentration")

plt.xlabel("Position")
plt.ylabel("Concentration")
plt.ylim(bottom=0)
plt.legend()
plt.show()


# ----------------------------------
# Vacancy-interstitial annihilation
# ----------------------------------

# metal interstitial atoms recombine with vacancies (forming a Frenkel pair) and annihilating
# I + V -> 0

my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh1D(np.linspace(0, 1, 100))

I = F.Species("I")
V = F.Species("V")

my_model.species = [I, V]

left_surf = F.SurfaceSubdomain1D(id=1, x=0)
right_surf = F.SurfaceSubdomain1D(id=2, x=1)

material = F.Material(D_0=1, E_D=0)

vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=material)

my_model.subdomains = [vol, left_surf, right_surf]

my_model.reactions = [
    F.Reaction(reactant=[I, V], k_0=1000, E_k=0, volume=vol), # Rapidly annihilating
    # never any 'separation' back into interstitial and vacancy
]
my_model.boundary_conditions = [
    F.FixedConcentrationBC(left_surf, value=10, species=I),
    F.FixedConcentrationBC(right_surf, value=0, species=I),
    F.FixedConcentrationBC(left_surf, value=0, species=V),
    F.FixedConcentrationBC(right_surf, value=5, species=V),
]
my_model.temperature = 300

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

my_model.initialise()
my_model.run()

for species in my_model.species:
    l, = plot_profile(species, label=species.name)
    plt.fill_between(
        l.get_data()[0],
        0,
        l.get_data()[1],
        alpha=0.2,
        color=l.get_color(),
    )


plt.xlabel("Position")
plt.ylabel("Concentration")
plt.legend()
plt.show()






