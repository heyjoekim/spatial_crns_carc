# --------------------------------------------------------------------------
# Python Code to fit URANOS/CRNS results
# --------------------------------------------------------------------------
# Author: Haejo Kim, Sam Tuttle
# Date Created: 6/20/2024
# Date last modified: 6/20/2024
# Note: python translation of original R code from Tuttle
# --------------------------------------------------------------------------

# Imports ------------------------------------------------------------------
import numpy as np
import pandas as pd
import xarray as xr
import scipy.optimize as opt

from lib.funcs import *
from lib.schron2017 import *

# Functions ----------------------------------------------------------------
def BiExp(x, a, b, c, d):
    return(a*np.exp(-x/b) + (c*np.exp(-x/d)))

def fit_crns_hist(func, hist_r, hist_counts, p0mat):
    popt, pcov = opt.curve_fit(func,
                               xdata = hist_r,
                               ydata = hist_counts,
                               p0 = p0mat)
    return(popt, pcov)

def our_BiExp_fit(r, popts):
    return(((30 / 0.8632525 * 192.3519) * np.exp(-r/1.6) + BiExp(r,*popts))* (1 - np.exp(-3.7*r)))

def scale_weights(popts):
    rs = np.arange(0,1400.1, 0.01)

    bi_exp_wts = our_BiExp_fit(rs, popts)

    weights_df = pd.DataFrame({
        'r': rs,
        'weight': bi_exp_wts
        })
    weights_df['cum_wts'] = weights_df['weight'].cumsum()
    weights_df['cum_totfrac'] = weights_df['weight'].cumsum() / np.sum(weights_df['weight'])
    weights_df['cum_totfrac_rev'] = 1 - weights_df['cum_totfrac']
    weights_df['wt_totfrac'] = weights_df.weight / 0.01 / np.sum(weights_df.weight)
    return(weights_df)


# Stuff --------------------------------------------------------------------
# # import nosnow test model run data
# nosnow = pd.read_csv('~/Documents/research/01.presentations/ESC2023/to_sam/neutron_count_hist_control.csv')
# nosnow.drop(['Unnamed: 0'], axis=1, inplace=True)
# 
# nosnow['mids'] = nosnow['bin_edges (left)']+2.5
# 
# mids = nosnow['mids'].iloc[3:].reset_index(drop=True)
# counts = nosnow['counts'].iloc[3:].reset_index(drop=True)
# 
# # Fit no snow run
# popt, pcov = fit_crns_hist(
#         BiExp,
#         mids,
#         counts, 
#         [150,115,170,16])
