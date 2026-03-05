# Simulating a thermo-desorption experiment for a sample of tungsten

import festim as F
import numpy as np

my_model = F.HydrogenTransportProblem()

vertices = np.concatenate(
    [
        np.linspace(0, 30e-9, 200),
        np.linspace(30e-9, 3e-6, 300),
        np.linspace(3e-6, 20e-6, 200)
    ]
)

my_model.mesh = F.Mesh1D(vertices)

tungsten = F.Material(
    D_0=4.1e-07,  # m2/s
    E_D=0.39,  # eV
)

volume_subdomain = F.VolumeSubdomain1D(id=1, borders=[0, 20e-6], material=tungsten)
left_boundary = F.SurfaceSubdomain1D(id=1, x=0)
right_boundary = F.SurfaceSubdomain1D(id=2, x=20e-6)

my_model.subdomains = [
    volume_subdomain,
    left_boundary,
    right_boundary,
]

avogadro = 6.02214076e23  # 1/mol
w_atom_density = 6.3e28  / avogadro # atom/m3 

H = F.Species("H")
trapped_H1 = F.Species("trapped_H1", mobile=False)
trapped_H2 = F.Species("trapped_H2", mobile=False)
# The concentration of the traps is W density multiplied by 1.3e-3
empty_trap1 = F.ImplicitSpecies(n=1.3e-3 * w_atom_density, others=[trapped_H1]) 
empty_trap2 = F.ImplicitSpecies(n=4e-4 * w_atom_density, others=[trapped_H2])
my_model.species = [H, trapped_H1, trapped_H2]

trapping_reaction_1 = F.Reaction(
    reactant=[H, empty_trap1],
    product=[trapped_H1],
    k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density), # surely density isn't in atom/m3 or this number would be too big and it wouldn't compute
    E_k=0.39,
    p_0=1e13,
    E_p=0.87,
    volume=volume_subdomain
)

trapping_reaction_2 = F.Reaction(
    reactant=[H, empty_trap2],
    product=[trapped_H2],
    k_0=4.1e-7 / (1.1e-10**2 * 6 * w_atom_density),
    E_k=0.39,
    p_0=1e13,
    E_p=1.0,
    volume=volume_subdomain
)

my_model.reactions = [trapping_reaction_1, trapping_reaction_2]

import ufl

implantation_time = 400  # s
incident_flux = 2.5e19/ avogadro  # H/m2/s

def ion_flux(t):
    return ufl.conditional(t <= implantation_time, incident_flux, 0)


def gaussian_distribution(x, center, width):
    return (
        1
        / (width * (2 * ufl.pi) ** 0.5)
        * ufl.exp(-0.5 * ((x[0] - center) / width) ** 2)
    )


source_term = F.ParticleSource(
    value=lambda x, t: ion_flux(t) * gaussian_distribution(x, 4.5e-9, 2.5e-9),
    volume=volume_subdomain,
    species=H,
)

my_model.sources = [source_term]

my_model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=left_boundary, value=0, species=H),
    F.FixedConcentrationBC(subdomain=right_boundary, value=0, species=H),
]

# In this example, the temperature is constant from t=0 to t=450 s, then
# increases from t=450 to t=500s in order to perform the thermo-desorption (TDS phase)

implantation_temp = 300  # K
temperature_ramp = 8  # K/s

start_tds = implantation_time + 50  # s


def temp_fun(t):
    if t <= start_tds:
        return implantation_temp
    else:
        return implantation_temp + temperature_ramp * (t - start_tds)


my_model.temperature = temp_fun

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, final_time=500)

my_model.settings.stepsize = F.Stepsize(
    initial_value=0.5,
    growth_factor=1.1,
    cutback_factor=0.9,
    target_nb_iterations=4,
    max_stepsize=lambda t: 0.5 if t > start_tds else None,
    milestones=[implantation_time, start_tds, start_tds + 50],
)

# We want to plot the evolution of the surface fluxes as a function of time
# use derived quantities object, use TotalVolume and HydrogenFlux

# Now it said HydrogenFlux but then SurfaceFlux is used, there is no HydrogenFlux object
left_flux = F.SurfaceFlux(surface=left_boundary, field=H)
right_flux = F.SurfaceFlux(surface=right_boundary, field=H)
total_mobile_H = F.TotalVolume(field=H, volume=volume_subdomain)
total_trapped_H1 = F.TotalVolume(field=trapped_H1, volume=volume_subdomain)
total_trapped_H2 = F.TotalVolume(field=trapped_H2, volume=volume_subdomain)

profile_exports = [
    F.Profile1DExport(
        field=spe,
        times=[implantation_time, start_tds, start_tds + 50],
    )
    for spe in [H, trapped_H1, trapped_H2]
]

my_model.exports = [
    total_mobile_H,
    total_trapped_H1,
    total_trapped_H2,
    left_flux,
    right_flux,
] + profile_exports

my_model.initialise()
my_model.run()

import matplotlib.pyplot as plt

time_labels = ["After implantation", "Start of TDS", "50 s after TDS start"]

fig, axs = plt.subplots(3, 3, sharex=True, sharey="row", figsize=(12, 8))
for i, profile in enumerate(profile_exports):
    axs[i, 0].set_ylabel(profile.field.name)
    for j, time in enumerate(profile.times):
        axs[0, j].set_title(time_labels[j])
        plt.sca(axs[i, j])
        plt.plot(profile.x, profile.data[j])
        plt.fill_between(
            profile.x,
            profile.data[j],
            alpha=0.2,
        )

for ax in axs.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)
plt.xlim(0, 3.5e-6)
axs[2, 0].set_xlabel("Position (m)")
plt.show()

t = left_flux.t
flux_total = np.array(left_flux.data) + np.array(right_flux.data)

plt.plot(t, flux_total, linewidth=3)

plt.ylabel(r"Desorption flux (mol m$^{-2}$ s$^{-1}$)")
plt.xlabel(r"Time (s)")
plt.show()

contribution_trap_1 = -np.diff(total_trapped_H1.data) / np.diff(t)
contribution_trap_2 = -np.diff(total_trapped_H2.data) / np.diff(t)

plt.plot(t, flux_total, linewidth=3)
plt.plot(t[1:], contribution_trap_1, linestyle="--", color="grey")
plt.fill_between(t[1:], 0, contribution_trap_1, facecolor="grey", alpha=0.1)
plt.plot(t[1:], contribution_trap_2, linestyle="--", color="grey")
plt.fill_between(t[1:], 0, contribution_trap_2, facecolor="grey", alpha=0.1)

plt.xlim(450, 500)
plt.ylim(-0.2e-5, 1e-5)
plt.ylabel(r"Desorption flux (mol m$^{-2}$ s$^{-1}$)")
plt.xlabel(r"Time (s)")

plt.ylabel(r"Desorption flux (mol m$^{-2}$ s$^{-1}$)")
plt.xlabel(r"Time (s)")
plt.show()



