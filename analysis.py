import pandas as pd
from main import make_array
import matplotlib.pyplot as plt
import numpy as np

def plot_slices(slices, ax=None):
    if not ax:
        fig, ax = plt.subplots()
    for slice in slices:
        ax.plot(slice)
    return ax

def get_density_drops(slices):
    maxs = slices.max(axis=1)
    return (maxs - slices.min(axis=1)) / maxs

def plot_random_slices(df, n=5):
    sc_ids = np.random.choice(df[df.matched_to_a_sc==True].index, n)
    non_sc_ids = np.random.choice(df[df.matched_to_a_sc==False].index, n)

    fig, ax = plt.subplots(n, 2)

    for i in range(n):
        
        plot_slices(df.loc[non_sc_ids[i], "slices"], ax[i, 0])
        ax[i, 0].set_xlabel(non_sc_ids[i])
        plot_slices(df.loc[sc_ids[i], "slices"], ax[i, 1])
        ax[i, 1].set_xlabel(sc_ids[i])

def get_selection_cut_histograms_and_power(column, df_var, log_scale=False, apply_density=True, N_bins=100, ax=None, range_=(0,1), power=False):
    if not ax:
        fig, ax = plt.subplots()
        
    # range
    ax.hist(df_var[column].to_list(), bins=N_bins, range=range_, density=apply_density, alpha=0.7, label="All Flat-Band Materials")
    
    # plot histogram for materials matched to a superconductor
    df_matched = df_var[df_var["matched_to_a_sc"]]
    ax.hist(df_matched[column].to_list(), bins=N_bins, range=range_, density=apply_density, alpha=0.7, label="Matched to a SC")    
    
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency Density")
    ax.legend()
    
    if log_scale:
        ax.set_yscale("log")
        
    
    if power:
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

        cut_values = np.linspace(min(df_var[column]), max(df_var[column]), 100)
        cut_values = np.linspace(0.2, 1, 100)

        powers = np.zeros_like(cut_values)

        for i, cut_value in enumerate(cut_values):
            cut_mask = df_var[column]<cut_value
            signal_to_noise_with_cut = df_var[cut_mask].matched_to_a_sc.sum() / cut_mask.sum()
            signal_to_noise_without_cut = df_var.matched_to_a_sc.sum() / len(df_var)

            powers[i] = signal_to_noise_with_cut / signal_to_noise_without_cut
        color = "red"
        
        ax2.plot(cut_values, powers, label="Selection Cut Power", color=color)
        ax2.set_xlim(*range_)
        ax2.set_ylabel('Power', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
    return ax

