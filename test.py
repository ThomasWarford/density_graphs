from structure_graph_utils import *
import numpy as np

def test_linear_slice():
    structure_graph, chgcar = create_structure_graph("mp-25")
    element = structure_graph.structure.elements[0].name
    sites = structure_graph.structure.sites


    three = linear_slice(sites[1].frac_coords, sites[5].frac_coords, (0, 1, 0), chgcar)
    # print(3, three)
    four = linear_slice(sites[3].frac_coords, sites[7].frac_coords, (1, 0, 0), chgcar)
    # print(4, four)
    assert(np.isclose(three, four).all())
    del three, four

def test_linear_slices():
    structure_graph, chgcar = create_structure_graph("mp-25")
    element = structure_graph.structure.elements[0].name
    sites = structure_graph.structure.sites


    out = linear_slices(
        [sites[1].frac_coords, sites[3].frac_coords],
        [sites[5].frac_coords, sites[7].frac_coords], 
        [[0, 1, 0],[1, 0, 0]],
        chgcar)
    assert(np.isclose(out[0], out[1]).all())


if __name__ == "__main__":
    test_linear_slice()
    test_linear_slices()
