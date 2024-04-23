from mp_api.client import MPRester
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from pymatgen.io.vasp import Chgcar
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN, MinimumDistanceNN
from itertools import combinations
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
from networkx.exception import NetworkXNoPath
import json

CHGCAR_DIRECTORY = Path("D:/materials_project/charge_density/")

def create_structure_graph(chgcar):
    neighbour_strategy = CrystalNN()
    structure = chgcar.structure

    structure_graph = StructureGraph.with_local_env_strategy(structure, neighbour_strategy, weights=True)
    structure_graph.graph = nx.Graph(structure_graph.graph) # from dimultigraph to graph!

    for u, v, data in structure_graph.graph.edges(data=True):
        data['weight'] = np.exp(data['weight']/2) # prefer edges with significant wavefunction overlap

    for node, data in structure_graph.graph.nodes(data=True):
        site = structure_graph.structure[node]
        data["element"] = site.specie.name
        data["frac_coords"] = site.frac_coords

    return structure_graph

def show_structure_graph(structure_graph, frac_coords=False, label=False, label_weights=False):
    # Get unique color for each element
    elements = structure_graph.structure.elements
    element_colors = {element.name:color for element,color in
                      zip(elements, cm.rainbow(np.linspace(0, 1, len(elements))))}
    
    ax = plt.figure().add_subplot(projection='3d')

    pos = {}
    for i, site in enumerate(structure_graph.structure.sites):
        if frac_coords:
            pos[i] = site.frac_coords
        else:
            pos[i] = site.coords

        ax.scatter(*pos[i], color=element_colors[site.specie.name])
        if label:
            ax.text(*pos[i], str(i))


    for i, j, data in structure_graph.graph.edges(data=True):
        x_y_z = np.stack([pos[i], pos[j]]).T
        ax.plot(*x_y_z, color="black")

        if label_weights:
            ax.text(*np.mean((pos[i], pos[j]), axis=0), data["weight"])

    plt.show()


def get_shortest_path_and_weight(graph, source_target, weight="weight"):
    path = nx.shortest_path(graph, source=source_target[0], target=source_target[1], weight=weight)
    weight = nx.path_weight(graph, path, weight="weight")
    return source_target, path, weight

def sublattice_minimum_spanning_tree_parallel(graph, element:str):
        """NO LONGER NEEDED: Get a subgraph of minimum total weight that connects all nodes of an element."""
        nodes_subset = [node for node, data in graph.nodes(data=True) if data["element"] == element]
        # Create a complete graph with the shortest paths between nodes in the subset
        h = nx.complete_graph(nodes_subset)
        inputs = list(combinations(nodes_subset, 2))
        shortest_path = partial(get_shortest_path_and_weight, graph)
        with Pool(6) as p:
            for h_edge, path, weight in tqdm(p.imap_unordered(shortest_path, inputs)):
                h[h_edge[0]][h_edge[1]]["path"] = path
                h[h_edge[0]][h_edge[1]]["weight"] = weight

        h_mst = nx.minimum_spanning_tree(h)
        graph_copy = nx.create_empty_copy(graph)
        
        for u,v,a in h_mst.edges(data=True):
            path = a["path"]
            for i in range(len(path)-1):
                graph_copy.add_edge(path[i], path[i+1], **graph[path[i]][path[i+1]])
        
        return graph_copy

def sublattice_minimum_spanning_tree(graph, element:str):
    """Get a subgraph of minimum total weight that connects all nodes of an element."""
    nodes_subset = [node for node, data in graph.nodes(data=True) if data["element"] == element]
    # Create a complete graph with the shortest paths between nodes in the subset
    h = nx.complete_graph(nodes_subset)
    inputs = list(combinations(nodes_subset, 2))
    shortest_path = partial(get_shortest_path_and_weight, graph)

    for h_edge, path, weight in map(shortest_path, inputs):
        h[h_edge[0]][h_edge[1]]["path"] = path
        h[h_edge[0]][h_edge[1]]["weight"] = weight

    h_mst = nx.minimum_spanning_tree(h)
    graph_copy = nx.create_empty_copy(graph)

    for u,v,a in h_mst.edges(data=True):
        path = a["path"]
        for i in range(len(path)-1):
            graph_copy.add_edge(path[i], path[i+1], **graph[path[i]][path[i+1]])

    return graph_copy




def linear_slice(f_i, f_f, jimage, chgcar_input, n=100):
    """
    INPUTS:
    f_i: starting fractional coordiates, in central unit cell
    f_f: final fractional coordinate, can be in neighboring cell
    jimage: (1, 0, 0) for unit cell one a vector away
    """
    chgcar = make_supercell(chgcar_input)

    if max((abs(i) for i in jimage)) > 1:
        return  np.full(n, 1e10)

    f_i = (f_i + (1, 1, 1))/3
    f_f = (f_f + (1, 1, 1) + jimage)/3


    return chgcar.linear_slice(f_i, f_f, n=n)

def make_supercell(chgcar_input):
    scale = (3, 3, 3)
    chgcar = chgcar_input.copy()
    chgcar.structure = chgcar.structure.make_supercell(scale)
    chgcar.data["total"] = np.tile(chgcar.data["total"], scale)
    # interpolator created during initialization, needs updating
    chgcar.dim *= np.array(scale)
    chgcar.xpoints = np.linspace(0.0, 1.0, num=chgcar.dim[0])
    chgcar.ypoints = np.linspace(0.0, 1.0, num=chgcar.dim[1])
    chgcar.zpoints = np.linspace(0.0, 1.0, num=chgcar.dim[2])
    chgcar.interpolator = RegularGridInterpolator(
            (chgcar.xpoints, chgcar.ypoints, chgcar.zpoints),
            chgcar.data["total"],
            bounds_error=True,
        )
    return chgcar

def linear_slices(f_1, f_2, jimage, chgcar_input, n=100):
    """
    INPUTS:
    f_1: array of starting fractional coordiates, in central unit cell
    f_2: array of final fractional coordinates, can be in neighboring cell
    jimage: array of jimages (1, 0, 0) for unit cell one a vector away
    """
    chgcar = make_supercell(chgcar_input)

    out = []

    for i, f, image in zip(f_1, f_2, jimage):
        i = (i + (1, 1, 1))/3
        f = (f + (1, 1, 1) + image)/3
        out.append(chgcar.linear_slice(i, f, n=n))

    return out

def get_mst_slices(chgcar_input, structure_graph, sublattice_element, n=100):
    h_mst = sublattice_minimum_spanning_tree(structure_graph.graph, sublattice_element)
    frac_coords = chgcar_input.structure.frac_coords
    chgcar = make_supercell(chgcar_input)
    out = np.zeros((h_mst.number_of_edges() ,n))
    for i, (u, v, data) in enumerate(h_mst.edges(data=True)):
        f_1 = (frac_coords[u] + (1, 1, 1))/3
        f_2 = (frac_coords[v] + (1, 1, 1) + data["to_jimage"])/3
        out[i] = chgcar.linear_slice(f_1, f_2, n=n)

    return out

def get_mst_slices_from_material_id(material_id, sublattice_element, n=100):
    try:
        chgcar = Chgcar.from_file(CHGCAR_DIRECTORY/f"{material_id}.chgcar")
        structure_graph = create_structure_graph(chgcar)
        return get_mst_slices(chgcar, structure_graph, sublattice_element, n=n)
    except Exception as e:
        print(material_id, e)
        return np.nan

def wrapped_function(id_element):
        material_id, sublattice_element = id_element
        return material_id, get_mst_slices_from_material_id(material_id, sublattice_element)

def get_mst_slices_from_material_ids(df_material, n=30):
    """INPUTS: 
    df_material: df indexed by material_id with sublattice_element column.
    """
    output = df_material.copy()

    inputs = zip(df_material.index, df_material["sublattice_element"])
    
    p = Pool(6)
    output["slices"] = ""
    for material_id, slices in tqdm(p.imap_unordered(wrapped_function, inputs), total=len(df_material)):
        try:
            output.at[material_id, "slices"] = json.dumps(slices.to_list())
        except:
            continue

    return output



def get_linear_slice_fractional_drop(f_i, f_f, jimage, chgcar_input, n=100):
    slice = linear_slice(f_i, f_f, jimage, chgcar_input, n)
    maximum = max(slice)
    return (maximum - min(slice)) / maximum

def get_fractional_drop_mst(chgcar_input, sublattice_element=None):
    chgcar = chgcar_input
    neighbour_strategy = MinimumDistanceNN(max(chgcar.structure.lattice.abc))
    structure = chgcar.structure

    if sublattice_element:
        unwanted_elements = [element.name for element in structure.elements]
        unwanted_elements.remove(sublattice_element)
        structure.remove_species(unwanted_elements)

    structure_graph = StructureGraph.with_local_env_strategy(structure, neighbour_strategy, weights=True)
    graph = nx.Graph(structure_graph.graph) # from dimultigraph to graph!

    for node, data in graph.nodes(data=True):
        site = structure_graph.structure[node]
        data["element"] = site.specie.name
        data["frac_coords"] = site.frac_coords

    chgcar = make_supercell(chgcar_input)

    for u, v, data in graph.edges(data=True):
        image = data["to_jimage"]
        i = (graph.nodes[u]["frac_coords"] + (1, 1, 1))/3
        f = (graph.nodes[v]["frac_coords"] + (1, 1, 1) + image)/3

        if max(abs(i) for i in image) <= 1:
            slice = chgcar.linear_slice(i, f, n=100)
            maximum = max(slice)
            data['weight'] = (maximum - min(slice)) / maximum
        else:
            data['weight'] = 1e9
    return nx.minimum_spanning_tree(graph, weight="weight")

def get_fractional_drop_mst_weights(chgcar, sublattice_element):
    mst = get_fractional_drop_mst(chgcar, sublattice_element)

    return [data["weight"] for _, __, data in mst.edges(data=True)]

def get_fractional_drop_mst_weight_from_material_id(id_element):
    material_id, sublattice_element = id_element

    try:
        chgcar = Chgcar.from_file(CHGCAR_DIRECTORY/f"{material_id}.chgcar")
        return material_id, get_fractional_drop_mst_weights(chgcar, sublattice_element)
    except Exception as e:
        print(material_id, e)
        return material_id, np.nan



def get_fractional_drop_msts_from_material_ids(df_material):
    """INPUTS: 
    df_material: df indexed by material_id with sublattice_element column.
    """
    output = df_material.copy()

    inputs = zip(df_material.index, df_material["sublattice_element"])
    
    p = Pool(12)
    output["slices"] = ""
    for material_id, drops in tqdm(p.imap_unordered(get_fractional_drop_mst_weight_from_material_id, inputs), total=len(df_material)):
        output.at[material_id, "slices"] = json.dumps(drops)
    return output