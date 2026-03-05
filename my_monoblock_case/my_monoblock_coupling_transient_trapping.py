
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
heat_model = F.HeatTransferProblem()
H_model = F.HydrogenTransportProblem()

# define transient worthy materials:
tungsten = F.Material(
    D_0=4.1e-7,
    E_D=0.39,
    K_S_0=1.87e24,
    E_K_S=1.04,
    thermal_conductivity=100,
    density = 19300, # kg/m3
    heat_capacity=134 # J/kg/K
)

copper = F.Material(
    D_0=6.6e-7,
    E_D=0.387,
    K_S_0=3.14e24,
    E_K_S=0.572,
    thermal_conductivity=350,
    density=8900,
    heat_capacity=390 # at around 900 celsius
)

cucrzr = F.Material(
    D_0=3.92e-7, 
    E_D=0.418, 
    K_S_0=4.28e23, 
    E_K_S=0.387, 
    thermal_conductivity=350,
    density = 8960,
    heat_capacity=383 
)

# define mesh for each problem
heat_model.mesh = F.Mesh(mesh)
H_model.mesh = F.Mesh(mesh)

# Possibly don't need ...
H_model.facet_meshtags = facet_tags
H_model.volume_meshtags = cell_tags
heat_model.facet_meshtags = facet_tags
heat_model.volume_meshtags = cell_tags

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


# define species for H model
H = F.Species("H")
H_model.species = [H]

# boundary conditions for each problem
heat_model.boundary_conditions = [
    F.FixedTemperatureBC(
        subdomain=top, 
        value=1173
    ),
    F.FixedTemperatureBC(
        subdomain=coolant_face,
        value=773
    )
]



import ufl
phi = 0.23e24
R_p = 1.1e-9
H_model.boundary_conditions = [
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

# define each problems subdomains
heat_model.subdomains = all_subdomains
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
    initial_value=100,
    growth_factor=1.1,
    cutback_factor=0.9,
    target_nb_iterations=4,
)

heat_model.settings = F.Settings(
    transient=True,
    atol=1e-9,
    rtol=1e-09,
    final_time=3.2e7,
)
heat_model.settings.stepsize = F.Stepsize(
    initial_value=100,
    growth_factor=1.1,
    cutback_factor=0.9,
    target_nb_iterations=4, # FIDDLE WITH THIS
)

problem = F.CoupledTransientHeatTransferHydrogenTransport(
    heat_problem=heat_model,
    hydrogen_problem=H_model
)

problem.initialise()
problem.run()




# Visualising I HOPE:
# -----------------------------
# import pyvista
# from dolfinx import plot

# T = problem.heat_problem.u
# c = problem.hydrogen_problem.species[0].post_processing_solution

# topology, cell_types, geometry = plot.vtk_mesh(T.function_space)
# u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# u_grid.point_data["T"] = T.x.array.real
# u_grid.set_active_scalars("T")
# u_plotter = pyvista.Plotter()
# u_plotter.add_mesh(u_grid, cmap="inferno", show_edges=False)
# u_plotter.add_mesh(u_grid, style="wireframe", color="white", opacity=0.2)

# contours = u_grid.contour(9)
# u_plotter.add_mesh(contours, color="white")

# u_plotter.view_xy()

# if not pyvista.OFF_SCREEN:
#     u_plotter.show()
# else:
#     figure = u_plotter.screenshot("temperature.png")

    
# topology, cell_types, geometry = plot.vtk_mesh(c.function_space)
# u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# u_grid.point_data["c"] = c.x.array.real
# u_grid.set_active_scalars("c")
# u_plotter = pyvista.Plotter()
# u_plotter.add_mesh(u_grid, cmap="viridis", show_edges=False)
# u_plotter.add_mesh(u_grid, style="wireframe", color="white", opacity=0.2)

# contours = u_grid.contour(9)
# u_plotter.add_mesh(contours, color="white")

# u_plotter.view_xy()

# if not pyvista.OFF_SCREEN:
#     u_plotter.show()
# else:
#     figure = u_plotter.screenshot("concentration.png")

