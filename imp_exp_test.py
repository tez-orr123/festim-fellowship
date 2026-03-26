import festim as F
import numpy as np
import matplotlib.pyplot as plt

transient = True
empty_mode = "implicit"

model = F.HydrogenTransportProblemDiscontinuous()

model.mesh = F.Mesh1D(vertices=np.linspace(0, 1, 11))

mat = F.Material(D_0=1, E_D=0, K_S_0=1, E_K_S=0)

vol = F.VolumeSubdomain1D(id=1, borders=(0, 1), material=mat)
left = F.SurfaceSubdomain1D(id=2, x=0)
right = F.SurfaceSubdomain1D(id=3, x=1)
model.subdomains = [vol, left, right]

H = F.Species("H")
H1 = F.Species("H1", mobile=False)
H2 = F.Species("H2", mobile=False)
n = 10
model.species = [H, H1, H2]

match empty_mode:
    case "implicit":
        empty = F.ImplicitSpecies(n=n, name="empty", others=[H1, H2])

    case "explicit":
        empty = F.Species("empty", mobile=False)
        model.initial_conditions = [
            F.InitialConcentration(species=empty, value=n, volume=vol)
        ]
        model.species.append(empty)

for s in model.species:
    s.subdomains = model.volume_subdomains

model.surface_to_volume = {left: vol, right: vol}
k, p = 0.001, 0.01
model.reactions = [
    F.Reaction(
        reactant=[empty, H], product=[H1], k_0=k, E_k=0, p_0=p, E_p=0, volume=vol
    ),
    F.Reaction(reactant=[H, H1], product=[H2], k_0=k, E_k=0, p_0=p, E_p=0, volume=vol),
]

model.boundary_conditions = [
    F.FixedConcentrationBC(subdomain=left, value=2, species=H),
    F.FixedConcentrationBC(subdomain=right, value=2, species=H),
]

model.temperature = 300

model.settings = F.Settings(
    atol=1e-12,
    rtol=1e-12,
    transient=transient,
    final_time=1000 if transient else None,
    stepsize=10 if transient else None,
)

model.exports = [F.Profile1DExport(field=spe, subdomain=vol) for spe in model.species]

model.initialise()
model.run()


plt.figure()
plt.title(f"{F.__version__} - \n Concentration profiles at steady state")
for profile in model.exports:
    plt.plot(profile.x, profile.data[-1], label=profile.field.name)
plt.legend()
plt.xlabel("x")
plt.ylabel("concentration")

plt.figure()
plt.title(
    f"{F.__version__} - \n Comparison of numerical and theoretical concentrations at steady state"
)
spe_to_val = {profile.field.name: profile.data[-1][0] for profile in model.exports}

theoretical_ct2 = k / p * spe_to_val["H"] * spe_to_val["H1"]
theoretical_ct1 = (
    k * spe_to_val["H"] * (n - spe_to_val["H2"]) + p * spe_to_val["H2"]
) / (2 * k * spe_to_val["H"] + p)

x = np.arange(len(["H1", "H2"]))
width = 0.35
plt.bar(x - width / 2, [theoretical_ct1, theoretical_ct2], width, label="theoretical")
plt.bar(x + width / 2, [spe_to_val["H1"], spe_to_val["H2"]], width, label="numerical")
plt.xticks(x, ["H1", "H2"])
plt.legend()
plt.show()