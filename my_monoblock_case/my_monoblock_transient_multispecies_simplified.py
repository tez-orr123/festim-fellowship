# Simplifying by taking away the Heat Transfer and Hydrogen Transport coupling
# and replacing it with a function of temperature

# BUT FIRST, will add the more accurate diffusivities to each species to make them three
# actually unique species. And maybe have no protium- will there be much of that implanted
# on the first wall?
import festim as F
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

# build mesh
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

# define problems, heat transfer and hydrogen transport
H_model = F.HydrogenTransportProblem()

# Define species
H = F.Species("H")
D = F.Species("D")
T = F.Species("T")

H_model.species = [H, D, T]

# Set-out each species transport parameters for each material
W_D_0_H = 4.1e-7
W_D_0_D = 4.1e-7
W_D_0_T = 4.1e-7

W_E_D_H = 0.37
W_E_D_D = 0.38
W_E_D_T = 0.39

Cu_D_0_H = 6.6e-7
Cu_D_0_D = 6.6e-7
Cu_D_0_T = 6.6e-7

Cu_E_D_H = 0.367
Cu_E_D_D = 0.377
Cu_E_D_T = 0.387

CuCrZr_D_0_H = 3.92e-7
CuCrZr_D_0_D = 3.92e-7
CuCrZr_D_0_T = 3.92e-7

CuCrZr_E_D_H = 0.398
CuCrZr_E_D_D = 0.408
CuCrZr_E_D_T = 0.418

# define transient worthy materials:
tungsten = F.Material(
    #D_0=4.1e-7,
    D_0={"H": float(W_D_0_H), "D": float(W_D_0_D), "T": (W_D_0_T)},
    #E_D=0.39,
    E_D={"H": float(W_E_D_H), "D": float(W_E_D_D), "T": (W_E_D_T)}, # By defining the explicit list here, makes it harder
    # to understand later in the BCs
    K_S_0=1.87e24,
    E_K_S=1.04,
    thermal_conductivity=100,
    density = 19300, # kg/m3
    heat_capacity=134 # J/kg/K
)

# make the other two materials have the list set-ups then find accurate values and implant them
copper = F.Material(
    #D_0=6.6e-7, 
    D_0={"H": float(Cu_D_0_H), "D": float(Cu_D_0_D), "T": (Cu_D_0_T)},
    #E_D=0.387,
    E_D={"H": float(Cu_E_D_H), "D": float(Cu_E_D_D), "T": (Cu_E_D_T)},
    K_S_0=3.14e24,
    E_K_S=0.572,
    thermal_conductivity=350,
    density=8900,
    heat_capacity=390 # at around 900 celsius
)

cucrzr = F.Material(
    #D_0=3.92e-7, 
    D_0={"H": float(CuCrZr_D_0_H), "D": float(CuCrZr_D_0_D), "T": (CuCrZr_D_0_T)},
    #E_D=0.418,
    E_D={"H": float(CuCrZr_E_D_H), "D": float(CuCrZr_E_D_D), "T": (CuCrZr_E_D_T)},
    K_S_0=4.28e23, 
    E_K_S=0.387, 
    thermal_conductivity=350,
    density = 8960,
    heat_capacity=383 
)

# define mesh for each problem
H_model.mesh = F.Mesh(mesh)

H_model.facet_meshtags = facet_tags
H_model.volume_meshtags = cell_tags

# define subdomains:
W_volume = F.VolumeSubdomain(id=227, material=tungsten)
Cu_volume = F.VolumeSubdomain(id=228, material=copper)
CuCrZr_volume = F.VolumeSubdomain(id=229, material=cucrzr)

top = F.SurfaceSubdomain(id=230,)
bottom = F.SurfaceSubdomain(id=232,)
W_sides = F.SurfaceSubdomain(id=231,)
Cu_sides = F.SurfaceSubdomain(id=236,)
CuCrZr_sides = F.SurfaceSubdomain(id=237,)
W_Cu_interlayer = F.SurfaceSubdomain(id=233,)
Cu_CuCrZr_interlayer = F.SurfaceSubdomain(id=234,)
coolant_face = F.SurfaceSubdomain(id=235,)

all_subdomains = [top, bottom, W_sides, Cu_sides, CuCrZr_sides, W_Cu_interlayer, Cu_CuCrZr_interlayer, coolant_face, W_volume, Cu_volume, CuCrZr_volume]




# boundary conditions for each problem



import ufl
phi = (0.23e24) / 3
R_p = 1.1e-9 
H_model.boundary_conditions = [
    F.FixedConcentrationBC(
        subdomain=top,
        value=lambda T: phi * R_p / (W_D_0_H * ufl.exp(-W_E_D_H / F.k_B / T)), # replace tungsten.D_0 and tungsten.E_D with W_D_0_H for example
        species=H
    ),
    F.FixedConcentrationBC(
        subdomain=coolant_face, 
        value=0, 
        species=H
    ),
    F.FixedConcentrationBC(
        subdomain=top,
        value=lambda T: phi * R_p / (W_D_0_D * ufl.exp(-W_E_D_D / F.k_B / T)),
        species=D
    ),
    F.FixedConcentrationBC(
        subdomain=coolant_face, 
        value=0, 
        species=D
    ),
    F.FixedConcentrationBC(
        subdomain=top,
        value=lambda T: phi * R_p / (W_D_0_T * ufl.exp(-W_E_D_T / F.k_B / T)),
        species=T
    ),
    F.FixedConcentrationBC(
        subdomain=coolant_face, 
        value=0, 
        species=T
    ),
]

# define each problems subdomains
H_model.subdomains = all_subdomains

# define each problems settings
# doing arbitrary numbers for now to get it running
# change to FPY time later
H_model.settings = F.Settings(
    transient=True,
    atol=1e-9,
    rtol=1e-9,
    final_time=3.2e7, 
)
H_model.settings.stepsize = F.Stepsize(
    initial_value=10000,
    growth_factor=1.1,
    cutback_factor=0.9,
    target_nb_iterations=4,
)

# Set temperature as a function, 1173 at top surface and 773 at the coolant
# But what function do I use when the whole coolant face is a temperature, that's not like a gradient line there
H_model.temperature = lambda x: ((400 * x[0]) + 2.135) / 0.005
H_model.temperature = 1000

# EXPORTS TIME.... but perhaps I could decouple and simplify to just a temperature function and 
# hydrogen transport problem

H_model.initialise()
H_model.run()


# Visualising I HOPE:
# how does the multi-species change the visualisation, can I visualise one at a time?
# -----------------------------
from dolfinx import plot
import pyvista

u_plotter = pyvista.Plotter()

for vol in H_model.volume_subdomains:
    sol = H.subdomain_to_post_processing_solution[vol]

    topology, cell_types, geometry = plot.vtk_mesh(sol.function_space)
    u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    u_grid.point_data["c"] = sol.x.array.real
    u_grid.set_active_scalars("c")
    u_plotter.add_mesh(u_grid, cmap="viridis", show_edges=False)
    u_plotter.add_mesh(u_grid, style="wireframe", color="white", opacity=0.2)

u_plotter.view_xy(negative=True)


if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")