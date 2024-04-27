from structure_graph_utils import *
from networkx.exception import NetworkXNoPath # return bad metric if no path can be found!
import pandas as pd

def make_array(json_string):
    try:
        return np.array(json.loads(json_string))
    except:
        return np.nan

if __name__ == "__main__":
    df = pd.read_csv("data/df_material.csv", index_col="material_id")
    df = df[["formula", "sublattice_element"]]
    chgcar_ids = [file.stem for file in CHGCAR_DIRECTORY.iterdir() if file.stem in df.index]
    df = df.loc[chgcar_ids]
    
    df = get_fractional_drop_mst_slices_from_material_ids(df)
    df.to_csv("data/df_mean_fractional_drop_slices.csv")

    # df = pd.read_csv("data/df_mean_fractional_drop.csv", index_col="material_id", converters={"slices": make_array})
    # print(df)