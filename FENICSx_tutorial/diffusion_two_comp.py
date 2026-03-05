import dolfinx

"https://jsdokken.com/dolfinx-tutorial/"

print(dolfinx.__version__)

from dolfinx.fem import functionspace  # Click `functionspace`
from ufl import div, grad  # Click `div` and `grad`
import numpy as np  # Click `numpy`
import ufl

# DOLFINx is C++ backend of FEniCSx
# Builds meshes, function spaces, functions
# Computes FE assembly and mesh refinement algorithms.
# Uses PETSc for linear algebra and solvers.
# UFL for variational formulations, high level math syntax
# FFCx is form compiler of FEniCSx,
# given variational formulations written by UFL generates efficient C code

# -----------
# generating simple mesh, imported a built-in mesh generator
# built a unit square mesh spanning [0,1] x [0,1] consisting of quadrilaterals
# -----------
from mpi4py import MPI
from dolfinx import mesh

domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)

# -----------
# defining function space on the mesh, V
# 'domain' is the mesh the space is defined on, 'lagrange; is the element family, '1' is the degree of the element
# -----------
from dolfinx import fem

import basix.ufl
from basix.ufl import element
cg_element = element("Lagrange", domain.basix_cell(), degree=1)

mixed_element = basix.ufl.mixed_element([cg_element, cg_element])

V = fem.functionspace(domain, mixed_element)

u = fem.Function(V)
print(u.sub(0))
print(ufl.split(u))
print(u.split())


cm, ct = ufl.split(u)

cm_post, ct_post = u.split()

# -----------
# defining boundary condition
# uD is the function defining the Dirichlet boundary condition
# interpolate method is used to project the expression onto the function space V
# -----------

V_bc, _ = V.sub(0).collapse()
uD = fem.Function(V_bc)
uD.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)

# -----------
# locating boundary dofs and creating DirichletBC object
# dofs in 1st order lagrange space are located at mesh vertices, therefore facet contains two dofs
# -----------
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)


def boundary_left(x):
    return np.isclose(x[0], 0)


def boundary_right(x):
    return np.isclose(x[0], 1)


def boundary_top(x):
    return np.isclose(x[1], 1)


dofs_right = fem.locate_dofs_geometrical(V_bc, boundary_right)
dofs_left = fem.locate_dofs_geometrical(V_bc, boundary_left)
dofs_top = fem.locate_dofs_geometrical(V_bc, boundary_top)

bc_left = fem.dirichletbc(uD, dofs_left)
bc_right = fem.dirichletbc(fem.Constant(domain, 1.0), dofs_right, V_bc)

# -----------
# defining the trial and test functions
# sufficient to use a common space for the two functions
# UFL is employed here to specify variational forms
# -----------

v_cm, v_ct = ufl.TestFunctions(V)

# -----------
# defining the variational problem
# creating the weak form of the Poisson equation
# a is the bilinear form, L is the linear form
# -----------

k = 0.001
n = 1
p = 1
trapping = k * cm * (n - ct)
detrapping = p * ct

F_mobile = ufl.dot(ufl.grad(cm), ufl.grad(v_cm)) * ufl.dx - trapping * v_cm *ufl.dx + detrapping * v_cm * ufl.dx
F_trapped = +trapping * v_ct *ufl.dx - detrapping * v_ct * ufl.dx

F = F_mobile + F_trapped
# Key feature of FEniCSx, formulas in variational forms translate directly to very similar python code, makes it east to sepcify and solve complicted PDE problems

# -----------
# Forming and solving the linear system
# create a linear problem to solve the variational problem
# Find u_h in V such that a(u_h, v) = L(v) for all v in V_hat
# PETSc is linear algebra backend, using direct solver (LU-factorisation)
# To ensure options passed to the LinearPRoblem are only used for the kiven KSP solver, we pass a UNIQUE option prefix as well
# -----------
from dolfinx.fem.petsc import NonlinearProblem

problem = NonlinearProblem(
    F,
    u,
    bcs=[bc_left, bc_right],
    petsc_options={  # How am I supposed to know which options to use?
        "snes_type": "newtonls",
        "snes_linesearch_type": "none" ,
        "snes_stol": 1e-10,
        "snes_divergence_tolerance": "PETSC_UNLIMITED",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "superlu_dist",
        "snes_monitor": None,
        "ksp_monitor": None,
    },
    # It don't like this line:
    petsc_options_prefix="Poisson",
)
problem.solve()
converged = problem.solver.getConvergedReason()
num_iter = problem.solver.getIterationNumber()
assert converged > 0, f"Solver did not converge, got {converged}."
print(
    f"Solver converged after {num_iter} iterations with converged reason {converged}."
)

cm_post, ct_post = u.split()
cm_post = cm_post.collapse()
ct_post = ct_post.collapse()

import pyvista

print(pyvista.global_theme.jupyter_backend)

from dolfinx import plot

domain.topology.create_connectivity(tdim, tdim)
# V_sub, _ = V.sub(0).collapse()
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(cm_post.function_space)

u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = cm_post.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()
    