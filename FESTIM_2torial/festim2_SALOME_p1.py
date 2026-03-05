# Mesh with SALOME part of tutorial

# SALOME v9.12

import festim as F

mesh = F.MeshFromXDMF("SALOME_meshes/mesh_domains.xdmf", "SALOME_meshes/mesh_boundaries.xdmf")

# mesh scaling
mesh.mesh.geometry.x[:] *= 1e-3

ft = mesh.define_surface_meshtags() # ft = facet tags
ct = mesh.define_volume_meshtags() # ct = cell tags

from dolfinx import plot
import pyvista

fdim = mesh.mesh.topology.dim - 1
tdim = mesh.mesh.topology.dim
mesh.mesh.topology.create_connectivity(fdim, tdim)
topology, cell_types, x = plot.vtk_mesh(
    mesh.mesh, tdim, ct.indices
)

p = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Cell Marker"] = ct.values
grid.set_active_scalars("Cell Marker")
p.add_mesh(grid, show_edges=True)
if pyvista.OFF_SCREEN:
    figure = p.screenshot("volume_marker.png")
p.show()

mesh.mesh.topology.create_connectivity(fdim, tdim)
topology, cell_types, x = plot.vtk_mesh(
    mesh.mesh, fdim, ft.indices
)

p = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Facet Marker"] = ft.values
grid.set_active_scalars("Facet Marker")
p.add_mesh(grid, show_edges=True)
if pyvista.OFF_SCREEN:
    figure = p.screenshot("facet_marker.png")
p.show()