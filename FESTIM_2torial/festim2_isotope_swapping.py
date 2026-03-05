# The isotope swapping part of the reactions tutorial

# Reaction objects can have multiple products, useful to simulate things
# like isotope swapping where a mobile species reacts with a trapped species 
# and their positions are swapped

# H + [T] <=> T + [H]
import festim as F
import numpy as np

my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh1D(np.linspace(0, 1, 100))

material = F.Material(D_0=1, E_D=0)

vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=material)

mobile_H = F.Species("H")
mobile_T = F.Species("T")
trapped_H = F.Species("H_trapped", mobile=False)
trapped_T = F.Species("T_trapped", mobile=False)

my_model.species = [mobile_H, mobile_T, trapped_H, trapped_T]

my_model.initial_conditions = [
    F.InitialConcentration(value=2, species=trapped_T, volume=vol),
]

left_surf = F.SurfaceSubdomain1D(id=1, x=0)
right_surf = F.SurfaceSubdomain1D(id=2, x=1)



my_model.subdomains = [vol, left_surf, right_surf]

my_model.initial_conditions = [
    F.InitialConcentration(value=2, species=trapped_T, volume=vol),
]

my_model.reactions = [
    F.Reaction(
        reactant=[mobile_H, trapped_T],
        product=[mobile_T, trapped_H],
        k_0=0.005,
        E_k=0,
        p_0=0.005,
        E_p=0,
        volume=vol,
    ),
]

my_model.boundary_conditions = [
    F.FixedConcentrationBC(left_surf, value=10, species=mobile_H),
    F.FixedConcentrationBC(right_surf, value=0, species=mobile_H),
]

my_model.temperature = 300

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=10)

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

for species in my_model.species:
    if "T" in species.name:
        color = "tab:green"
    else:
        color = "tab:blue"
    if "trapped" in species.name:
        linestyle = "--"
    else:
        linestyle = "-"
    plot_profile(species, label=species.name, color=color, linestyle=linestyle)

plt.xlabel("Position")
plt.ylabel("Concentration")
plt.legend()
plt.show()