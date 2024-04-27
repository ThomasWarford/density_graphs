from structure_graph_utils import *
import numpy as np
import pandas as pd

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

def test_view_linear_slices():
    actual = get_mst_slices_from_material_id("mp-775989", "Fe", n=100)
    fig, ax = plt.subplots()
    ax.plot(actual.flatten())
    plt.show()

def test_mst_slice_regression():
    expected = np.genfromtxt("test_data/mp_172_mst_density.csv", delimiter=",")
    actual = get_mst_slices_from_material_id("mp-172", "Nd", n=100)
    assert(np.isclose(expected, actual).all())

def test_mst_slice_parallel():
    expected = np.genfromtxt("test_data/mp_172_mst_density.csv", delimiter=",")
    df_input = pd.DataFrame({"material_id":["mp-7", "mp-172"], "sublattice_element": ["S", "Nd"]})
    df_input.set_index("material_id", inplace=True)
    output = get_mst_slices_from_material_ids(df_input)

    assert(np.isclose(expected, output.loc["mp-172", "slices"]).all())

def test_get_fractional_drop_mst():
    material_id = "mp-172"
    chgcar = Chgcar.from_file(CHGCAR_DIRECTORY/f"{material_id}.chgcar")
    neighbour_strategy = CrystalNN()

    structure_graph = StructureGraph.with_local_env_strategy(chgcar.structure, neighbour_strategy, weights=True)
    mst = get_fractional_drop_mst(chgcar, "Nd")
    
    show_graph(mst, frac_coords=True, label=True, label_weights=True)

def test_linear_slices():
    # actual = get_mst_slices_from_material_id("mp-775989", "Fe", n=100)
    material_id = "mp-7"
    chgcar = Chgcar.from_file(CHGCAR_DIRECTORY/f"{material_id}.chgcar")
    sites = chgcar.structure.sites

    expected = chgcar.linear_slice(sites[0].frac_coords, sites[1].frac_coords)
    actual = linear_slice(sites[0].frac_coords, sites[1].frac_coords, (0, 0, 0), chgcar)

    print(actual)
    print(expected)
    assert((actual==expected).all())


if __name__ == "__main__":
    chgcar = Chgcar.from_file(CHGCAR_DIRECTORY/f"mp-25.chgcar")
    # test_linear_slice(chgcar)
    # test_linear_slices(chgcar)
    # test_mst_slice_regression()
    # test_mst_slice_parallel()
    # test_view_linear_slices()
    # test_get_fractional_drop_mst()
    test_linear_slices()
