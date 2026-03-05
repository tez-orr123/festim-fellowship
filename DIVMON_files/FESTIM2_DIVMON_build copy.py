import festim as F
import gmsh
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
import festim as F
import gmsh
import numpy as np
import os

# === Geometry: cuboid with a central pipe and two concentric interlayers ===
#   Block:   L=25 mm (x), B=29 mm (y), H=10 mm (z)
#   Pipe:    inner radius = 5 mm (void)
#   Layers:  2 mm CuCrZr (adjacent to pipe), then 1 mm Cu (outside)
#   Result:  3 solid volumes (Bulk/W, Cu, CuCrZr) + a void pipe.
#
#   Physical groups created:
#     Volumes (dim=3): 31 Bulk_W, 32 Cu, 33 CuCrZr
#     Surfaces (dim=2):
#       11 PipeInner, 12 Interface_Cu_CuCrZr, 13 Interface_Bulk_Cu
#       21 Inlet(x=0), 22 Outlet(x=L), 23 OuterWalls (y/z planes)
#
# Notes:
# - '3' in [(3, tag)] is the OCC entity dimension: 3 = volume, 2 = surface, 1 = curve, 0 = point.
# - addCylinder(x,y,z, dx,dy,dz, r) builds a cylinder with base at (x,y,z) and axis vector (dx,dy,dz).
#   Here we use (0,0,H) so the cylinder passes through the full thickness.
# - We build shells with boolean 'cut', subtract them from the block for non-overlapping volumes,
#   and then label both volumes and interfaces robustly using adjacency.

import gmsh
import numpy as np
import os

# Optional (if you convert to DOLFINx)
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio

# -----------------------------
# 1) Initialize
# -----------------------------
gmsh.initialize()
gmsh.model.add("Pipe_with_Interlayers")

# -----------------------------
# 2) Parameters (mm)
# -----------------------------
L = 25.0  # length  (x)
B = 29.0  # height  (y)
H = 10.0  # width   (z)

r_inner = 5.0       # pipe inner radius (void)
t_cocr  = 2.0       # CuCrZr thickness (adjacent to pipe)
t_cu    = 1.0       # Cu thickness (outside CuCrZr)

r_ifc   = r_inner + t_cocr       # interface radius between CuCrZr and Cu (7 mm)
r_outer = r_ifc   + t_cu         # outer radius of Cu (8 mm)

# Cylinder axis: through thickness (+z), centered in x-y
cx, cy, cz = L/2, B/2, 0.0
ax, ay, az = 0.0, 0.0, H        # axis vector

# -----------------------------
# 3) Primitives
# -----------------------------
# Main block
block_tag = gmsh.model.occ.addBox(0.0, 0.0, 0.0, L, B, H)

# Concentric cylinders
inner_tag = gmsh.model.occ.addCylinder(cx, cy, cz, ax, ay, az, r_inner)
ifc_tag   = gmsh.model.occ.addCylinder(cx, cy, cz, ax, ay, az, r_ifc)
outer_tag = gmsh.model.occ.addCylinder(cx, cy, cz, ax, ay, az, r_outer)

# -----------------------------
# 4) Build non-overlapping solids: Bulk/W, Cu, CuCrZr; keep pipe as void
# -----------------------------
# Shells:
cu_shell, _    = gmsh.model.occ.cut([(3, outer_tag)], [(3, ifc_tag)],   removeObject=True, removeTool=False)
cocr_shell, _  = gmsh.model.occ.cut([(3, ifc_tag)],   [(3, inner_tag)], removeObject=True, removeTool=False)

# Bulk/W: block minus the full outer cylinder (so shells fit into the cavity)
bulk_vol, _ = gmsh.model.occ.cut([(3, block_tag)], [(3, outer_tag)], removeObject=True, removeTool=False)

# Synchronize after boolean ops
gmsh.model.occ.synchronize()

# -----------------------------
# 5) Helpers for tagging
# -----------------------------
def to_tags(dimtags):
    return [t for (d, t) in dimtags if d == 3]

def boundary_surfaces_of(vol_dimtags):
    """Return set of surface tags that bound the given volumes."""
    faces = set()
    for d, vt in vol_dimtags:
        _, surfTags = gmsh.model.getAdjacencies(3, vt)
        faces.update(surfTags)
    return faces

def shared_interface(volsA, volsB):
    """Surfaces on the interface between two volume sets."""
    return list(boundary_surfaces_of(volsA).intersection(boundary_surfaces_of(volsB)))

def faces_on_plane(plane, tol=1e-9):
    """Select faces lying on a specific box plane."""
    faces = gmsh.model.occ.getEntities(dim=2)
    tags = []
    for _, f in faces:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(2, f)
        if plane == "x0" and abs(xmin - 0.0) < tol and abs(xmax - 0.0) < tol:
            tags.append(f)
        elif plane == "xL" and abs(xmin - L) < tol and abs(xmax - L) < tol:
            tags.append(f)
        elif plane == "y0" and abs(ymin - 0.0) < tol and abs(ymax - 0.0) < tol:
            tags.append(f)
        elif plane == "yB" and abs(ymin - B) < tol and abs(ymax - B) < tol:
            tags.append(f)
        elif plane == "z0" and abs(zmin - 0.0) < tol and abs(zmax - 0.0) < tol:
            tags.append(f)
        elif plane == "zH" and abs(zmin - H) < tol and abs(zmax - H) < tol:
            tags.append(f)
    return tags

def add_pg(dim, tags, pid=None, name=None):
    if not tags:
        return None
    gid = gmsh.model.addPhysicalGroup(dim, tags) if pid is None else gmsh.model.addPhysicalGroup(dim, tags, pid)
    if name:
        gmsh.model.setPhysicalName(dim, gid, name)
    return gid

# -----------------------------
# 6) Physical groups: volumes
# -----------------------------
PID_BULK = 31
PID_CU   = 32
PID_COCR = 33

bulk_ids = to_tags(bulk_vol)
cu_ids   = to_tags(cu_shell)
cocr_ids = to_tags(cocr_shell)

add_pg(3, bulk_ids, PID_BULK, "Bulk_W")
add_pg(3, cu_ids,   PID_CU,   "Cu")
add_pg(3, cocr_ids, PID_COCR, "CuCrZr")

# -----------------------------
# 7) Physical groups: interfaces & boundaries (surfaces)
# -----------------------------
# Interfaces between solids
faces_bulk_cu   = shared_interface(bulk_vol, cu_shell)     # Bulk–Cu at r = r_outer
faces_cu_cocr   = shared_interface(cu_shell, cocr_shell)   # Cu–CuCrZr at r = r_ifc

# Pipe inner wall: the inner cylindrical face of the CuCrZr shell (exclude top/bottom and shared faces)
cocr_faces_all  = boundary_surfaces_of(cocr_shell)
# Remove faces on Cu–CuCrZr interface and any accidental overlap with bulk
cocr_faces_rem  = set(cocr_faces_all) - set(faces_cu_cocr) - set(shared_interface(cocr_shell, bulk_vol))
# Keep only lateral cylinder (drop faces on z=0 and z=H)
faces_z0 = set(faces_on_plane("z0"))
faces_zH = set(faces_on_plane("zH"))
pipe_inner_faces = list(cocr_faces_rem - faces_z0 - faces_zH)

PID_PIPEIN  = 11
PID_CUCOCR  = 12
PID_BULKCU  = 13

add_pg(2, pipe_inner_faces, PID_PIPEIN, "PipeInner")
add_pg(2, faces_cu_cocr,    PID_CUCOCR, "Interface_Cu_CuCrZr")
add_pg(2, faces_bulk_cu,    PID_BULKCU, "Interface_Bulk_Cu")

# External boundaries: inlet/outlet/walls of the outer box
PID_INLET  = 21
PID_OUTLET = 22
PID_WALLS  = 23

inlet_faces   = faces_on_plane("x0")
outlet_faces  = faces_on_plane("xL")
outer_walls   = list(set(faces_on_plane("y0") + faces_on_plane("yB") +
                         faces_on_plane("z0") + faces_on_plane("zH")))

add_pg(2, inlet_faces,  PID_INLET,  "Inlet")
add_pg(2, outlet_faces, PID_OUTLET, "Outlet")
add_pg(2, outer_walls,  PID_WALLS,  "OuterWalls")

# -----------------------------
# 8) Optional: mesh refinement near the pipe wall & interfaces
# -----------------------------
# Characteristic element sizes (tune to taste)
lc_min = r_inner / 8.0   # fine near pipe
lc_max = r_inner * 2.0   # coarser away

# Refine near pipe inner + both material interfaces
ref_targets = pipe_inner_faces + faces_cu_cocr + faces_bulk_cu

if ref_targets:
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "FacesList", ref_targets)

    f_thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thr, "IField", f_dist)
    gmsh.model.mesh.field.setNumber(f_thr, "LcMin", lc_min)
    gmsh.model.mesh.field.setNumber(f_thr, "LcMax", lc_max)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMin", 0.5 * r_inner)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMax", 2.0 * r_inner)

    gmsh.model.mesh.field.setAsBackgroundMesh(f_thr)

# -----------------------------
# 9) Mesh and write files
# -----------------------------
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)

os.makedirs("gmsh", exist_ok=True)
gmsh.write("gmsh/DIVMON_testthistest.msh")

# Convert to DOLFINx mesh and tags (for FESTIM / FEniCSx)
model_rank = 0
mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, model_rank)

# Optionally visualize in Gmsh GUI
# gmsh.fltk.run()

gmsh.finalize()

