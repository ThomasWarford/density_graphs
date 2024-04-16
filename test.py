from structure_graph_utils import *
import numpy as np

def test_linear_slice(chgcar):
    structure_graph = create_structure_graph(chgcar)
    element = structure_graph.structure.elements[0].name
    sites = structure_graph.structure.sites

    three = linear_slice(sites[1].frac_coords, sites[5].frac_coords, (0, 1, 0), chgcar, n=10)
    print(3, three)
    four = linear_slice(sites[3].frac_coords, sites[7].frac_coords, (1, 0, 0), chgcar, n=10)
    print(4, four)
    assert(np.isclose(three, four).all())

def test_linear_slices(chgcar):
    structure_graph = create_structure_graph(chgcar)
    element = structure_graph.structure.elements[0].name
    sites = structure_graph.structure.sites

    out = linear_slices(
        [sites[1].frac_coords, sites[3].frac_coords],
        [sites[5].frac_coords, sites[7].frac_coords], 
        [[0, 1, 0],[1, 0, 0]],
        chgcar,
        n=10)
    assert(np.isclose(out[0], out[1]).all())

def mst_slice_regression_test():
    expected = np.genfromtxt("test_data/mp_172_mst_density.csv", delimiter=",")
    actual = get_mst_slices_from_materials_id("mp-172", "Nd", n=100)
    assert(np.isclose(expected, actual).all())


if __name__ == "__main__":
    chgcar = Chgcar.from_file(CHGCAR_DIRECTORY/f"mp-25.chgcar")
    # test_linear_slice(chgcar)
    # test_linear_slices(chgcar)
    mst_slice_regression_test()
