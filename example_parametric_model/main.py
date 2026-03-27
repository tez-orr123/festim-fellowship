from model import run_model
import matplotlib.pyplot as plt
import festim as F
import numpy as np
from typing import Tuple

def plot_profiles(model: F.HydrogenTransportProblemDiscontinuous, label_prefix="") -> list:
    """PLot the concentrations profiles of a given model (last timestep)

    Args:
        model: the festim model object
    """
    lines = []
    for profile in model.exports:
        l, = plt.plot(profile.x, profile.data[-1], label=label_prefix + profile.field.name)
        lines.append(l)

    return lines

def plot_h2_profile(model, label_prefix=""):
    lines = []
    for profile in model.exports:
        if profile.field.name != "H2":
            continue
        l, = plt.plot(profile.x, profile.data[-1], label=label_prefix)
        lines.append(l)

    return lines

def get_mean_concentrations(model: F.HydrogenTransportProblemDiscontinuous) -> Tuple[list[str], list[float]]:
    concentrations = []
    labels = []
    for profile in model.exports:
        concentration_profile = profile.data
        mean_conc = np.mean(concentration_profile)

        concentrations.append(mean_conc)
        labels.append(profile.field.name)

    return labels, concentrations

n= 10.0
k, p = 0.001, 0.1

plt.figure()
plt.title(f"{F.__version__} - \n Concentration profiles at steady state")

for n in np.linspace(10, 100, num=4):
    model = run_model(k=k, p=p, n=n)
    # plot_profiles(model, label_prefix=f"n={n}_")
    plot_h2_profile(model, label_prefix=f"n={n}_")

plt.legend()
plt.xlabel("x")
plt.ylabel("concentration")



plt.figure()
ns = np.linspace(10, 100, num=10)
means = []
for n_val in ns:
    model = run_model(k=k, p=p, n=n_val)
    labels, concentrations = get_mean_concentrations(model)
    means.append(concentrations)

plt.plot(ns, means, label=labels, marker="o")

plt.xlabel("$n$")
plt.ylabel("Mean concentration")
plt.legend()
plt.yscale("log")


# run parametric study
ps = np.linspace(0.1, 0.5, num=10)


models = {}
for p in ps:
    print(f"Running festim model for {n=}, {k=}, {p=}")
    model = run_model(k=k, p=p, n=n)
    models[p] = model

print(models)
breakpoint()

plt.figure()
means = []
for p, model in models.items():
    labels, concentrations = get_mean_concentrations(model)
    means.append(concentrations)

plt.plot(ps, means, label=labels, marker="o")

plt.xlabel("$p$")
plt.ylabel("Mean concentration")
plt.legend()
plt.yscale("log")
plt.show()
