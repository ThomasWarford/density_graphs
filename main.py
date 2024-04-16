from structure_graph_utils import *
from networkx.exception import NetworkXNoPath # return bad metric if no path can be found!
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("data/df_material.csv", index_col="material_id")
    df = df[["formula", "sublattice_element"]]
    df = get_mst_slices_from_material_ids(df)
    df.to_csv("data/df_slices.csv")