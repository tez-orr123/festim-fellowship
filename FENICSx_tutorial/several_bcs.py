import dolfinx

"https://jsdokken.com/dolfinx-tutorial/"

print(dolfinx.__version__)

from dolfinx.fem import functionspace  # Click `functionspace`
from ufl import div, grad  # Click `div` and `grad`
import numpy as np  # Click `numpy`

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

V = fem.functionspace(domain, ("Lagrange", 1))

# -----------
# defining boundary condition
# uD is the function defining the Dirichlet boundary condition
# interpolate method is used to project the expression onto the function space V
# -----------
uD = fem.Function(V)
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


def boundary_bottom(x):
    return np.isclose(x[1], 0)


dofs_right = fem.locate_dofs_geometrical(V, boundary_right)
dofs_left = fem.locate_dofs_geometrical(V, boundary_left)
dofs_top = fem.locate_dofs_geometrical(V, boundary_top)
dofs_bottom = fem.locate_dofs_geometrical(V, boundary_bottom)

bc_left = fem.dirichletbc(uD, dofs_left)
bc_right = fem.dirichletbc(fem.Constant(domain, 0.0), dofs_right, V)
bc_top = fem.dirichletbc(uD, dofs_top)
bc_bottom = fem.dirichletbc(fem.Constant(domain, 0.0), dofs_bottom, V)

# -----------
# defining the trial and test functions
# sufficient to use a common space for the two functions
# UFL is employed here to specify variational forms
# -----------
import ufl

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# -----------
# defining the source term, which is a constant -6
# fem.Constant is used to define the source term over the entire domain
# -----------
from dolfinx import default_scalar_type

f = fem.Constant(domain, default_scalar_type(0.0))

# -----------
# defining the variational problem
# creating the weak form of the Poisson equation
# a is the bilinear form, L is the linear form
# -----------
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# Key feature of FEniCSx, formulas in variational forms translate directly to very similar python code, makes it east to sepcify and solve complicted PDE problems

# -----------
# Forming and solving the linear system
# create a linear problem to solve the variational problem
# Find u_h in V such that a(u_h, v) = L(v) for all v in V_hat
# PETSc is linear algebra backend, using direct solver (LU-factorisation)
# To ensure options passed to the LinearPRoblem are only used for the kiven KSP solver, we pass a UNIQUE option prefix as well
# -----------
from dolfinx.fem.petsc import LinearProblem

problem = LinearProblem(
    a,
    L,
    bcs=[bc_left, bc_right, bc_top, bc_bottom],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    # It don't like this line:
    petsc_options_prefix="Poisson",
)
uh = problem.solve()

# -----------
# Computing the error
# This is the MMS method (Methos of Manufactured Solutions)
# exact solution u_exact is known, we can compute the error between uh and u_exact
# -----------

# First we interpolate the exact solution in a function space that contains it (higher)
V2 = fem.functionspace(domain, ("Lagrange", 2))
uex = fem.Function(V2, name="u_exact")
uex.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)

# Then we compute the L^2 error norm between uh and u_exact
L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

# Then we compute the maximum error at any DOF
# Do this by accessing all the DOFs computed by problem.solve()
# Access DOFs by accessing underlying vector u_h
# Already have interpolated exact solution u_exact into the first order space when creating the BC, can compare teh max values at any DOF of the approximation space, sure
error_max = np.max(np.abs(uD.x.array - uh.x.array))
if domain.comm.rank == 0:  # Only print the error on one process
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")

# Get expected error out despite not having the prefix definition line in the LinearProblem class

# -----------
# Plotting the mesh using pyvista
# -----------
import pyvista

print(pyvista.global_theme.jupyter_backend)

from dolfinx import plot

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
# Plotting a function using pyvista
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

# Can warp the mesh to make use of the 3D plotting capabilities of pyvista
warped = u_grid.warp_by_scalar()
plotter2 = pyvista.Plotter()
plotter2.add_mesh(warped, show_edges=True, show_scalar_bar=True)
if not pyvista.OFF_SCREEN:
    plotter2.show()

# -----------
# External post-processing
# -----------
from dolfinx import io
from pathlib import Path

results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)
filename = results_folder / "fundamentals"
with io.VTXWriter(domain.comm, filename.with_suffix(".bp"), [uh]) as vtx:
    vtx.write(0.0)
with io.XDMFFile(domain.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

# I don't know where this was written on this PC but I'm sure it worked
