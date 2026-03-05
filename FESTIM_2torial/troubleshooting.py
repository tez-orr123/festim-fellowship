# print("Min T =", problem.heat_problem.T.x.array.min())
# # OBJECT (problem.heat_problem) has no attribute T.... WHICH IS BAD, jacobian is killed instantly
# print("Max T =", problem.heat_problem.T.x.array.max())

# print(problem.mesh.facet_meshtags.values)


#print("Min T =", heat_model.T.x.array.min()) # NO ATTRIBUTE T, that's not true !!
# print("Min T =", H_model.T.x.array.min())

# -------------------------------------------------
# POSSIBLE TROUBLESHOOTS:
# festim_mesh = F.Mesh()
# festim_mesh.mesh = mesh
# festim_mesh.facet_meshtags = facet_tags
# festim_mesh.volume_meshtags = cell_tags

# heat_model.mesh = festim_mesh
# H_model.mesh = festim_mesh

# problem.heat_problem.T.x.array[:] = 900.0  # K, example
# problem.heat_problem.T.x.scatter_forward()

# put density in kg/m3 not atomistic
# --------------------------------------------------