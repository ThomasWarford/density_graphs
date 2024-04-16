from structure_graph_utils import *
import numpy as np

def test_linear_slice():
    structure_graph, chgcar = create_structure_graph("mp-25")
    element = structure_graph.structure.elements[0].name
    sites = structure_graph.structure.sites


    one = linear_slice(sites[0].frac_coords, sites[4].frac_coords, (-1, -1, -1), chgcar)
    print(1, one)
    two = linear_slice(sites[1].frac_coords, sites[5].frac_coords, (0, 1, 0), chgcar)
    print(2, two)
    three = linear_slice(sites[2].frac_coords, sites[6].frac_coords, (0, 0, 1), chgcar)
    print(3, three)
    four = linear_slice(sites[3].frac_coords, sites[7].frac_coords, (1, 0, 0), chgcar)
    print(4, four)
    assert(np.iscloe(one, three))
    assert(np.iscloe(two, four))


if __name__ == "__main__":
    test_linear_slice()




    # assert(isclose(one, two, rel_tol=0.05))