import dolfinx
import numpy as np
from dolfinx import fem, mesh, io, plot
import ufl
import matplotlib as mpl
import pyvista
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

# Define time discretization parameters
t = 0.0  # Start time
T = 1.0  # Final time
num_steps = 50
dt = T / num_steps  # time step size

# Domain will be [-2, 2] X [-2, 2] so we don't have a unit square here, we need to make a rectangle using dolfinx.mesh.create_rectangle()
nx, ny = 50, 50 # Resolution
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([-2, -2]), np.array([2, 2])],
    [nx, ny],
    mesh.CellType.triangle,
)
V = fem.functionspace(domain, ("Lagrange", 1))

def initial_condition(x, a=5):
    return np.exp(-a * (x[0] ** 2 + x[1] ** 2))

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

# Creating BC
fdim = domain.topology.dim -1
boundary_facets = mesh.locate_entities_boundary(
    domain,
    fdim,
    lambda x: np.full(x.shape[1], True, dtype=bool)

)
bc = fem.dirichletbc(
    PETSc.ScalarType(0),
    fem.locate_dofs_topological(V, fdim, boundary_facets),
    V
)

# We will want to visualise the transient results in Paraview so will export as xdmf file.

xdmf = io.XDMFFile(domain.comm, "diffusion.xdmf", "w")
xdmf.write_mesh(domain)

uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
xdmf.write_function(uh, t)

# Variational problem and solver
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(0))
a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n + dt * f) * v * ufl.dx

# Preparing linear algebra structures for time dependent problems
# even if u_n is time dependent, we will reuse the same function for f and u_n at every time step.
# Therefore, call fdolfinx.fem.form() to generate assembly kernels for the matrix and vector.
# This functions creates a dolfinx.fem.Form - major difference I'm sure

bilinear_form = fem.form(a)
linear_form = fem.form(L)

# More matrix stuff coming into play, do I need to understand the Jacobian matrices
# stuff going on behind the scenes, I am not sure but I will go with what it says for now.

# Left hand side of system, amtrix A does not change from one time step to another
# therefore only need to assemble it once - cool
# 
A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()
b = create_vector(fem.extract_function_spaces(linear_form))

# Using petsc4py to create a linear solver
# Creating a Krylov subspace solver using PETSc, assigning matrix A to the solver and choosing the solution strategy
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# Visualisation of time dependent problem using pyvista
# Plotting the solution at every 15th time step
# Also adding a colourbar to show the min and max value of u at each time step.
# Don't have to learn how to plot things, just understand the bulk of the simulation

# Doesn't work? MESA and glx errors, perhaps not the right drivers?

# -------------------------------------------------
grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

plotter = pyvista.Plotter()
plotter.open_gif("u_time.gif", fps=10)

grid.point_data["uh"] = uh.x.array
warped = grid.warp_by_scalar("uh", factor=1)

# viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
# sargs = dict(
#     title_font_size=25,
#     label_font_size=20,
#     fmt="%.2e",
#     color="black",
#     position_x=0.1,
#     position_y=0.8,
#     width=0.8,
#     height=0.1,
# )

# renderer = plotter.add_mesh(
#     warped,
#     show_edges=True,
#     lighting=False,
#     cmap=viridis,
#     scalar_bar_args=sargs,
#     clim=[0, max(uh.x.array)],
# )
# ----------------------------------------

# Updating the solution and right hand side per time step
for i in range(num_steps):
    t += dt

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [bilinear_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # Solve linear problem
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array

    # Write solution to file
    xdmf.write_function(uh, t)
    # Update plot
    new_warped = grid.warp_by_scalar("uh", factor=1)
    warped.points[:, :] = new_warped.points
    warped.point_data["uh"][:] = uh.x.array
    plotter.write_frame()

