import meshio
from typing import Optional

def convert_med_to_xdmf(
        #"\\wsl.localhost\Ubuntu\home\tezorr\festim-fellowship\SALOME_meshes\main_monoblock_mesh.med"
    med_file : "SALOME_meshes/main_monoblock_mesh.med",
    cell_file: Optional[str] = "mesh_domains.xdmf",
    facet_file: Optional[str] = "mesh_boundaries.xdmf",
    cell_type: Optional[str] = "tetra",
    facet_type: Optional[str] = "triangle",
):
    """Converts a .med mesh to .xdmf
    
    Args:
        med_file: the name of the MED file
        cell_file: the name of the file containing the volume markers. Defaults to "mesh_domains.xdmf".
        facet_file: the name of the file containing the surface markers.. Defaults to "mesh_boundaries.xdmf".
        cell_type: The topology of the cells. Defaults to "tetra".
        facet_type: The topology of the facets. Defaults to "triangle".
    
    Returns:
        dict, dict: the correspondence dict, the cell types
    """
    
    msh = meshio.read(med_file)

    correspondance_dict = {-k: v for k, v in msh.cell_tags.items()}

    cell_data_types = msh.cell_data_dict["cell_tags"].keys()

    for mesh_block in msh.cells:
        if mesh_block.type == cell_type:
            meshio.write_points_cells(
                cell_file,
                msh.points,
                [mesh_block],
                cell_data={"f": [-1 * msh.cell_data_dict["cell_tags"][cell_type]]},
            )
        elif mesh_block.type == facet_type:
            meshio.write_points_cells(
                facet_file,
                msh.points,
                [mesh_block],
                cell_data={"f": [-1 * msh.cell_data_dict["cell_tags"][facet_type]]},
            )

    return correspondance_dict, cell_data_types

correspondance_dict, cell_data_types = convert_med_to_xdmf(
    "SALOME_meshes/monoblock_refined_relevant_edges.med",
    cell_file="SALOME_meshes/monoblock_mesh_domains.xdmf",
    facet_file="SALOME_meshes/monoblock_mesh_boundaries.xdmf",
    cell_type="tetra",
    facet_type="triangle",
)

print(correspondance_dict)