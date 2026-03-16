import festim as F
import numpy as np

tungsten = F.Material(
    D_0=1,
    E_D=0, 
    K_S_0=1,
    E_K_S=0,
)

# Subdomains
W_volume = F.VolumeSubdomain1D(id=6, borders=[0, 1], material=tungsten)

# Problem
my_model = F.HydrogenTransportProblemDiscontinuous()
my_model.mesh = F.Mesh1D(np.linspace(0, 1, 100))

my_model.subdomains =  [
    W_volume,
]

Deuterium = F.Species("D", subdomains=my_model.volume_subdomains)
trapped_T = F.Species("T_trapped", mobile=False, subdomains=my_model.volume_subdomains)
trapped_D = F.Species("D_trapped", mobile=False, subdomains=my_model.volume_subdomains)

empty_traps = F.ImplicitSpecies(n=1, others=[trapped_T, trapped_D], name='implicit_species')

my_model.species = [Deuterium, trapped_D, trapped_T]

my_model.reactions = [
    F.Reaction(
        reactant=[Deuterium, empty_traps],
        product=[trapped_T],
        k_0=1,
        E_k=0,
        p_0=1,
        E_p=0,
        volume=W_volume,
    ),
]

my_model.temperature = 300

my_model.settings = F.Settings(
    transient=False,
    atol=1e-8,
    rtol=1e-10,
)

my_model.initialise()
my_model.run()