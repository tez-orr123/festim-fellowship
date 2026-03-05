import dolfinx
import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh
from mpi4py import MPI

from dolfinx.fem.petsc import NonlinearProblem


# Coefficient
def g(u):
    return 1 + u**2


# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)  # Cell type added here?

# Defining source term
x = ufl.SpatialCoordinate(domain)
u_ufl = 1 + x[0] + 2 * x[1]  # 1 + x + 2y
f = -ufl.div(g(u_ufl) * ufl.grad(u_ufl))

V = fem.functionspace(domain, ("Lagrange", 1))


def u_exact(x):
    return eval(str(u_ufl))


# Defining bcs
u_D = fem.Function(V)
u_D.interpolate(u_exact)
tdim = domain.topology.dim
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))

# We must replace the trial function, u, with a Function, uh, for nonlinear problems, serves as our unknown
uh = fem.Function(V)
v = ufl.TestFunction(V)
# This below would also work if you wrote out a and L like in the linear problem and subtracted them
# F = a - L, just add in the coefficient g(uh)
# F is the residual we are finding in the Newton method
F = g(uh) * ufl.dot(ufl.grad(uh), ufl.grad(v)) * ufl.dx - f * v * ufl.dx

# -----------
# Newtons Method
# Defining the nonlinear variational problem
# Newton’s method requires methods for evaluating the residual F (including application of boundary conditions),
# as well as a method for computing the Jacobian matrix.
# DOLFINx provides the function NonlinearProblem that implements these methods.
# The DOLFINx NonlinearProblem is an interface to the PETSc SNES solver
# -----------

petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_atol": 1e-6,
    "snes_rtol": 1e-6,
    "snes_monitor": None,
    "ksp_error_if_not_converged": True,
    "ksp_type": "gmres",
    "ksp_rtol": 1e-8,
    "ksp_monitor": None,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_max_iter": 1,
    "pc_hypre_boomeramg_cycle_type": "v",
}

problem = NonlinearProblem(
    F,
    uh,
    bcs=[bc],
    petsc_options=petsc_options,
    petsc_options_prefix="nonlinpoisson",
)

problem.solve()
converged = problem.solver.getConvergedReason()
num_iter = problem.solver.getIterationNumber()
assert converged > 0, f"Solver did not converge, got {converged}."
print(
    f"Solver converged after {num_iter} iterations with converged reason {converged}."
)

from dolfinx import plot
import pyvista

domain.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("fundamentals_mesh.png")
# -----------
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)

u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()

warped = u_grid.warp_by_scalar()
plotter2 = pyvista.Plotter()
plotter2.add_mesh(warped, show_edges=True, show_scalar_bar=True)
if not pyvista.OFF_SCREEN:
    plotter2.show()
