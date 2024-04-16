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

CHGCAR_DIRECTORY = Path("D:/materials_project/charge_density/")

def create_structure_graph(material_id: str):
    chgcar = Chgcar.from_file(CHGCAR_DIRECTORY/f"{material_id}.chgcar")
    # neighbour_strategy = MinimumDistanceNN(tol = 1, cutoff=max(chgcar.structure.lattice.abc), get_all_sites=True)
    neighbour_strategy = CrystalNN(search_cutoff=max(chgcar.structure.lattice.abc))
    # structure = chgcar.structure.make_supercell((2, 2, 2))
    structure = chgcar.structure
    structure_graph = StructureGraph.with_local_env_strategy(structure, neighbour_strategy, weights=True)
    structure_graph.graph = nx.Graph(structure_graph.graph) # from dimultigraph to graph!
    for u, v, data in structure_graph.graph.edges(data=True):
        if 'weight' in data:
            data['weight'] = (data['weight']/5)**4 # put to a power > 1 to discourage direct paths
    for node, data in structure_graph.graph.nodes(data=True):
        site = structure_graph.structure[node]
        data["element"] = site.specie.name
        data["frac_coords"] = site.frac_coords
    return structure_graph, chgcar

def show_structure_graph(structure_graph, frac_coords=False, label=False):

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


    for i, j in structure_graph.graph.edges:
        x_y_z = np.stack([pos[i], pos[j]]).T
        ax.plot(*x_y_z, color="black")

    plt.show()


def get_shortest_path_and_weight(graph, source_target, weight="weight"):
    path = nx.shortest_path(graph, source=source_target[0], target=source_target[1], weight=weight)
    weight = nx.path_weight(graph, path, weight="weight")
    return source_target, path, weight

def sublattice_minimum_spanning_tree(graph, element:str):
        """Get a subgraph of minimum total weight that connects all nodes of an element."""
        nodes_subset = [node for node, data in graph.nodes(data=True) if data["element"] == element]
        # Create a complete graph with the shortest paths between nodes in the subset
        h = nx.complete_graph(nodes_subset)
        inputs = list(combinations(nodes_subset, 2))
        print(len(inputs), " inputs")
        shortest_path = partial(get_shortest_path_and_weight, graph)
        with Pool(10) as p:
            for h_edge, path, weight in tqdm(p.imap_unordered(shortest_path, inputs)):
                h[h_edge[0]][h_edge[1]]["path"] = path
                h[h_edge[0]][h_edge[1]]["weight"] = weight

        h_mst = nx.minimum_spanning_tree(h)
        graph_copy = nx.create_empty_copy(graph)
        
        for u,v,a in h_mst.edges(data=True):
            path = a["path"]
            for i in range(len(path)-1):
                graph_copy.add_edge(path[i], path[i+1], weight=graph[path[i]][path[i+1]]["weight"])
        
        return graph_copy

def linear_slice(f_i, f_f, jimage, chgcar_input, n=100):
    """
    INPUTS:
    f_i: starting fractional coordiates, in central unit cell
    f_f: final fractional coordinate, can be in neighboring cell
    jimage: (1, 0, 0) for unit cell one a vector away
    """
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

    f_i = (f_i + (1, 1, 1))/3
    f_f = (f_f + (1, 1, 1) + jimage)/3


    return chgcar.linear_slice(f_i, f_f, n=n)

def linear_slices(f_i, f_f, jimage, chgcar_input, n=100):
    """
    INPUTS:
    f_i: array of starting fractional coordiates, in central unit cell
    f_f: array of final fractional coordinates, can be in neighboring cell
    jimage: array of jimages (1, 0, 0) for unit cell one a vector away
    """
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

    out = []

    for i, f, image in zip(f_i, f_f, jimage):
        i = (i + (1, 1, 1))/3
        f = (f + (1, 1, 1) + image)/3
        out.append(chgcar.linear_slice(i, f, n=n))

    return out
