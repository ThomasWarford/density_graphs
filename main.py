from structure_graph_utils import *


if __name__ == "__main__":
    structure_graph, chgcar = create_structure_graph("mp-25")
    element = structure_graph.structure.elements[0].name
    # structure_graph.graph = sublattice_minimum_spanning_tree(structure_graph.graph, element)
    show_structure_graph(structure_graph, frac_coords=True, label=True)
    # with MPRester("JnHWrxVLfsx0yXzJa96wMIbC0s2Gom0C") as mpr:
    #     docs = mpr.summary.search(material_ids=["mp-7"], fields=["material_id", "nsites", "nelements"])
    #     print(docs)