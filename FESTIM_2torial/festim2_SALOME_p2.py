import festim as F

model = F.HydrogenTransportProblem()

model.mesh = F.MeshFromXDMF(
    volume_file="SALOME_meshes/mesh_domains.xdmf", 
    facet_file="SALOME_meshes/mesh_boundaries.xdmf"
)

material_1 = F.Material(
    D_0=1,
    E_D=0.1
)
material_2 = F.Material(
    D_0=5,
    E_D=0.3
)

volume_left = F.VolumeSubdomain(id=6, material=material_1)
volume_right = F.VolumeSubdomain(id=7, material=material_2)

left_surface = F.SurfaceSubdomain(id=8)
right_surface = F.SurfaceSubdomain(id=9)

model.subdomains = [volume_left, volume_right, left_surface, right_surface]

H = F.Species("H")
model.species = [H]

model.boundary_conditions = [
    F.FixedConcentrationBC(
        species=H,
        value=1,
        subdomain=left_surface,
    ),
    F.FixedConcentrationBC(
        species=H,
        value=0,
        subdomain=right_surface,
    )
]

model.temperature = 823

model.settings = F.Settings(
    atol=1e-10,
    rtol=1e-10,
    transient=False,
)

model.initialise()
model.run()

from dolfinx import plot
import pyvista


u = H.post_processing_solution

topology, cell_types, geometry = plot.vtk_mesh(u.function_space)
u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["c"] = u.x.array.real
u_grid.set_active_scalars("c")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, cmap="viridis", show_edges=False)
u_plotter.add_mesh(u_grid, style="wireframe", color="white", opacity=0.2)

u_plotter.view_xy()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("concentration.png")