import festim as F
import numpy as np

# -------------------------------------------
# Part 2: Convenience class for trapping
# -------------------------------------------

# Convenience class 'Trap' can be used for ONE mobile species and ONE hydrogen per trap.
# only need to define trapping rate, mobile species adn total number of traps

# my_model = F.HydrogenTransportProblem()
# my_model.mesh = F.Mesh1D(np.linspace(0, 1, 100))

# material = F.Material(D_0=1, E_D=0)

# left_surf = F.SurfaceSubdomain1D(id=1, x=0)
# right_surf = F.SurfaceSubdomain1D(id=2, x=1)

# vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=material)

# my_model.subdomains = [vol, left_surf, right_surf]

# mobile_H = F.Species("H")
# my_model.species = [mobile_H]
# # So not specifying traps as a species here

# my_model.traps = [
#     F.Trap(
#         mobile_species=mobile_H,
#         k_0=0.01,
#         E_k=0,
#         p_0=0.1,
#         E_p=0,
#         volume=vol,
#         n=2,  # number of traps, which I guess makes sense as it's a small 1D mesh
#         name="Trap1",
#     )
# ]

# my_model.boundary_conditions = [
#     F.FixedConcentrationBC(left_surf, value=10, species=mobile_H),
#     F.FixedConcentrationBC(right_surf, value=0, species=mobile_H),
# ]

# my_model.temperature = 300
# my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=50)
# my_model.settings.stepsize = F.Stepsize(1)

# my_model.initialise()
# my_model.run()

# import matplotlib.pyplot as plt


# def plot_profile(species, **kwargs):
#     index = my_model.species.index(species)
#     V0, dofs = my_model.function_space.sub(index).collapse()
#     coords = V0.tabulate_dof_coordinates()[:, 0]
#     sort_coords = np.argsort(coords)
#     c = my_model.u.x.array[dofs][sort_coords]
#     x = coords[sort_coords]
#     return plt.plot(x, c, **kwargs)


# for species in my_model.species:
#     plot_profile(species, label=species.name)

# plt.xlabel("Position")
# plt.ylabel("Concentration")
# plt.legend()
# plt.show()

# # ------------------------
# # Multi-Occupancy Trapping
# # ------------------------

# # Trapping scheme:
# # H + [] <=> [1H]
# # H + [1H] <=> [2H]
# # H + [2H] <=> [3H]

# my_model = F.HydrogenTransportProblem()
# my_model.mesh = F.Mesh1D(np.linspace(0, 1, 100))

# mobile_H = F.Species("H")
# trapped_1H = F.Species("1H_trapped", mobile=False)
# trapped_2H = F.Species("2H_trapped", mobile=False)
# trapped_3H = F.Species("3H_trapped", mobile=False)
# empty_traps = F.ImplicitSpecies(n=2, others=[trapped_1H, trapped_2H, trapped_3H])

# my_model.species = [mobile_H, trapped_1H, trapped_2H, trapped_3H]

# left_surf = F.SurfaceSubdomain1D(id=1, x=0)
# right_surf = F.SurfaceSubdomain1D(id=2, x=1)

# material = F.Material(D_0=1, E_D=0)

# vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=material)

# my_model.subdomains = [vol, left_surf, right_surf]

# my_model.reactions = [
#     F.Reaction(
#         reactant = [mobile_H, empty_traps],
#         product = [trapped_1H],
#         k_0 = 0.01,
#         E_k = 0,
#         p_0 = 0.1,
#         E_p = 0,
#         volume = vol
#     ),
#     F.Reaction(
#         reactant = [mobile_H, trapped_1H],
#         product = [trapped_2H],
#         k_0 = 0.02,
#         E_k = 0,
#         p_0 = 0.1,
#         E_p = 0,
#         volume = vol
#     ),
#         F.Reaction(
#         reactant = [mobile_H, trapped_2H],
#         product = [trapped_3H],
#         k_0 = 0.03,
#         E_k = 0,
#         p_0 = 0.1,
#         E_p = 0,
#         volume = vol
#     )
# ]

# my_model.boundary_conditions = [
#     F.FixedConcentrationBC(left_surf, value=10, species=mobile_H),
#     F.FixedConcentrationBC(right_surf, value=0, species=mobile_H),
# ]

# my_model.temperature = 300

# my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=50)

# my_model.settings.stepsize = F.Stepsize(1)

# my_model.initialise()
# my_model.run()

# for species in my_model.species:
#     plot_profile(species, label=species.name)

# plt.xlabel("Position")
# plt.ylabel("Concentration")
# plt.legend()
# plt.show()

# ---------------------------------------
# Two species, one trap, odd name innit
# ---------------------------------------

# One trap that can accept two different species, H and D
# H +[] <=> [H]
# T + [] <=> [T]

my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh1D(np.linspace(0, 1, 100))
# same problem, mesh, 

mobile_H = F.Species("H")
mobile_T = F.Species("T")
trapped_H = F.Species("H_trapped", mobile=False)
trapped_T = F.Species("T_Trapped", mobile=False)
empty_traps = F.ImplicitSpecies(n=2, others=[trapped_H, trapped_T])

my_model.species = [mobile_H, mobile_T, trapped_H, trapped_T]

left_surf = F.SurfaceSubdomain1D(id=1, x=0)
right_surf = F.SurfaceSubdomain1D(id=2, x=1)

material = F.Material(D_0=1, E_D=0)

vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=material)

my_model.subdomains = [vol, left_surf, right_surf]


my_model.reactions = [
    F.Reaction(
        reactant = [mobile_H, empty_traps],
        product = [trapped_H],
        k_0=0.1,
        E_k=0,
        p_0=0.001,
        E_p=0,
        volume=vol,
    ),
    F.Reaction(
        reactant=[mobile_T, empty_traps],
        product=[trapped_T],
        k_0=0.1,
        E_k=0,
        p_0=0.001,
        E_p=0,
        volume=vol,
    ),
]

my_model.boundary_conditions = [
    F.FixedConcentrationBC(left_surf, value=10, species=mobile_H),
    F.FixedConcentrationBC(right_surf, value=0, species=mobile_H),
    F.FixedConcentrationBC(left_surf, value=0, species=mobile_T),
    F.FixedConcentrationBC(right_surf, value=5, species=mobile_T),
]

my_model.temperature = 300

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=20)

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

# Good


