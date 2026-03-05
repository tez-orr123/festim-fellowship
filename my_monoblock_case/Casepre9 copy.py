# Case pre 9
# Set-up:
# -	Continuous
# -	Transient, maybe short
# -	Single species
# -	Just 1 trap in tungsten
# -	Set temp

import festim as F
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

# mesh
avo = 6.022e23

mesh_data = gmshio.read_from_msh(
    "gmsh_files/testing_DIVMON.msh", MPI.COMM_WORLD, 0, gdim=3
)
mesh = mesh_data.mesh
assert mesh_data.facet_tags is not None
facet_tags = mesh_data.facet_tags
facet_tags.name = "Facet markers"

assert mesh_data.cell_tags is not None
cell_tags = mesh_data.cell_tags
cell_tags.name = "Cell markers"

my_model = F.HydrogenTransportProblem()
my_model.mesh = F.Mesh(mesh)

# species and traps
H = F.Species("H")
trapped_H = F.Species("H_trapped", mobile=False)
# instead of the initial conditions method, nabbing this from
# the TDS simulation:
w_density = 6.3e28 / avo
trap_density = 1e25 /avo
empty_trap = F.ImplicitSpecies(n = trap_density, others=[trapped_H])
my_model.species = [H, trapped_H]

# material
tungsten = F.Material(
    D_0=4.1e-7,
    E_D=0.39,
    K_S_0=1.87e24/avo,
    E_K_S=1.04,
    thermal_conductivity=100,
)

copper = F.Material(
    D_0=6.6e-7, 
    E_D=0.387,
    K_S_0=3.14e24/avo,
    E_K_S=0.572,
    thermal_conductivity=350,
)

cucrzr = F.Material(
    D_0=3.92e-7, 
    E_D=0.418,
    K_S_0=4.28e23/avo, 
    E_K_S=0.387, 
    thermal_conductivity=350
)
# volume
my_model.facet_meshtags = facet_tags
my_model.volume_meshtags = cell_tags

W_volume = F.VolumeSubdomain(id=227, material=tungsten)
Cu_volume = F.VolumeSubdomain(id=228, material=copper)
CuCrZr_volume = F.VolumeSubdomain(id=229, material=cucrzr)



# surfaces
top = F.SurfaceSubdomain(id=230,)
bottom = F.SurfaceSubdomain(id=232,)
W_sides = F.SurfaceSubdomain(id=231,)
Cu_sides = F.SurfaceSubdomain(id=236,)
CuCrZr_sides = F.SurfaceSubdomain(id=237,)
W_Cu_interlayer = F.SurfaceSubdomain(id=233,)
Cu_CuCrZr_interlayer = F.SurfaceSubdomain(id=234,)
coolant_face = F.SurfaceSubdomain(id=235,)

# subdomains set
all_subdomains = [top, bottom, W_sides, Cu_sides, CuCrZr_sides, W_Cu_interlayer, Cu_CuCrZr_interlayer, coolant_face, W_volume, Cu_volume, CuCrZr_volume]
my_model.subdomains = all_subdomains

# reactions
lattice_length = 1.1e-10  # m
n_solute_per_site = 6
my_model.reactions = [
    F.Reaction(
        reactant=[H, empty_trap],
        product=[trapped_H],
        k_0=(((tungsten.D_0)/((lattice_length)**2 * n_solute_per_site))/avo),
        E_k=0.265,
        p_0=1.2397e11, # The detrapping rate is order of magnitude higher than the trapping rate here
        E_p=1.3,
        volume=W_volume,
    ),
] # Everything is in the same order of magnitude as the TDS simulation example so should work


# my_model.initial_conditions = [
#     F.InitialConcentration(
#         value=0.0,
#         species=trapped_H,
#         volume=W_volume
#     ),
#     F.InitialConcentration(
#         value=1e10,
#         species=H,
#         volume=W_volume
#     )
# ]

my_model.temperature = lambda x: ((x[1] + 0.0401225) / (3.25e-5))

# boundary conditions
import ufl
phi = ((0.23e24)) /avo
R_p = 1.1e-9 
my_model.boundary_conditions = [
    F.FixedConcentrationBC(
        subdomain=top,
        value=lambda T: phi * R_p / (tungsten.D_0 * ufl.exp(-tungsten.E_D / F.k_B / T)),
        species=H
    ),
    F.FixedConcentrationBC(
        subdomain=coolant_face, 
        value=0, 
        species=H
    ),
]

# temperature


# settings and stepsize
my_model.settings = F.Settings(
    transient=True,
    atol=1e-15, # My tolerances may have been way too high from the discontinuous case before this
    rtol=1e-10,
    final_time=3.2e7, # Works when full year but initial time is less than 10
)
my_model.settings.stepsize = F.Stepsize(
    initial_value=100, # doesn't like when my initial value isn't 1
    growth_factor=1.1,
    cutback_factor=0.9,
    target_nb_iterations=4, # decreasing this makes it work again
)




from dolfinx.log import LogLevel, set_log_level

set_log_level(LogLevel.INFO)


my_model.exports = [F.VTXSpeciesExport(field=trapped_H, filename="trapped_H.bp")]

# run
my_model.initialise()
my_model.run()

from dolfinx import plot
import pyvista

# Modelling H concentration MOBILE
hydrogen_concentration = H.post_processing_solution

topology, cell_types, geometry = plot.vtk_mesh(hydrogen_concentration.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["c of mobile H"] = hydrogen_concentration.x.array.real
u_grid.set_active_scalars("c of mobile H")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration_mobile_H.png")

h_trapped_concentration = trapped_H.post_processing_solution

topology, cell_types, geometry = plot.vtk_mesh(h_trapped_concentration.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["c of trapped H"] = h_trapped_concentration.x.array.real
u_grid.set_active_scalars("c of trapped H")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration_trapped_H.png")

# All of the models have 0 H concentration over them
# something wrong with my trap values perhaps?

# might try increase mesh fineness for this one


# Possible fixes to the H stuck at the top issue:
# - no temperature function means the BC that uses T
#   has indefinite H piling up near the top
# - k_0 being divided by density may be wrong
# - p_0 could be order of magnitudes lower and
# - E_p could be higher 1-1.4 eV
# - my trapping and detrapping rates differed by 
#   17-20 orders of magnitude, solvers reduce timesteps
#   to aboid blowups which results in a freeze.