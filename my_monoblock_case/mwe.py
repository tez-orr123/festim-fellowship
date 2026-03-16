import festim as F
import numpy as np

tungsten = F.Material(
    D_0=1,
    E_D=0, 
    K_S_0=1,
    E_K_S=0,
)

vol1 = F.VolumeSubdomain1D(id=1, borders=[0, 0.5], material=tungsten)
vol2 = F.VolumeSubdomain1D(id=2, borders=[0.5, 1], material=tungsten)

left = F.SurfaceSubdomain1D(id=1, x=0)
right = F.SurfaceSubdomain1D(id=2, x=1)

my_model = F.HydrogenTransportProblemDiscontinuous()

my_model.subdomains = [vol1, vol2]

my_model.surface_to_volume = {
    left: vol1,
    right: vol2,
}

species1 = F.Species("T", subdomains=my_model.volume_subdomains)
species2 = F.Species("T_trapped", mobile=False, subdomains=my_model.volume_subdomains)
implicit_sep = F.ImplicitSpecies(n=1, others=[species2])

my_model.species = [species1, species2]

# Mesh
my_model.mesh = F.Mesh1D(np.linspace(0, 1, 100))

my_model.surface_to_volume = {}


my_model.reactions = [
    F.Reaction(
        reactant=[species1, implicit_sep],
        product=[species2],
        k_0=1,
        E_k=0,
        p_0=1,
        E_p=0,
        volume=vol1,
    ),

]

my_model.temperature = 300

# Settings
my_model.settings = F.Settings(
    transient=True,
    atol=1e-8, 
    rtol=1e-10,
    final_time=10,
    stepsize=1,
)

my_model.initialise()

print(species1.concentration)

my_model.run()