"""
Python File with functions and other variables
    Author: Haejo Kim
    Affiliation: Syracuse University
    Date last edited: Apr. 30, 2024

"""


import numpy as np
import pandas as pd
import scipy

from lib.uranos import URANOS

# Other Functions
def mean_bias_error(xx, yy):
    # Calculate mean bias error
    mbe = np.mean(xx-yy)
    return(mbe)

# constants
crns_x = -211.265  # Detector x pos [m]
crns_y = -167.261  # Detector y pos [m]

# spline model
# make spline model:
HDPE_25mm = pd.read_csv('./DRFs/ResponseFunction_HDPE25mm.txt',
                        header=None,
                        sep='\t',
                        names=['E [MeV]','response'])
HDPE_25mm['eV'] = HDPE_25mm['E [MeV]']/1e-6

logx = np.log10(HDPE_25mm['E [MeV]'])
logy = np.log10(HDPE_25mm.response)
lin_interp = scipy.interpolate.interp1d(logx, logy, kind='cubic', fill_value='extrapolate')

log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))

def readUranosNC(path, run_dir, crns_x, crns_y):
    # read URANOS model output located at folderi
    #   path    : path to general location of model output files
    #   run_dir : working directory where Detector Hits file is located
    #   crns_x  : x direction of CRNS (center is (0,0))
    #   crns_y  : y direction of CRNS (center is (0,0))
    # Outputs:
    #   U       : URANOS Object
    #   U.Hits  : Hits DF
    U = URANOS(folder=path+run_dir)
    # read detector hits and drop dups
    U = U.read_hits().drop_multicounts()

    # recalculate distance to CRNS not located at center
    U.Hits['r'] = np.sqrt((U.Hits['x'] - crns_x)**2+(U.Hits['y'] - crns_y)**2)

    # weight hits using DRF
    U = U.weight_by_detector_response(method='drf',
                                      file = './DRFs/ResponseFunction_HDPE25mm.txt')
    U.Hits['Prob'] = log_interp(U.Hits['Energy_[MeV]'])/100

    # center x and y to CRNS location to calc angles
    U.Hits['rel_x'] = U.Hits['x'] - crns_x
    U.Hits['rel_y'] = U.Hits['y'] - crns_y

    U.Hits['w_new'] = np.arctan2(U.Hits['rel_y'], U.Hits['rel_x']) + np.pi
    U.Hits['w_deg_new'] = U.Hits['w_new']/np.pi*180
    return(U, U.Hits)


def calcNeutronCounts(hits_df):
    # calculate Filtered Neutron Counts
    return(hits_df['Prob'].sum())


def calcHist(df_hits, point, date):
    counts, bin_edges = np.histogram(df_hits['r'],
                                     bins=np.arange(0,500,5),
                                     weights=df_hits['Prob'])
    hist_nc = pd.DataFrame({
        'counts':counts,
        'bin_edge_left':bin_edges[:-1]
        })
    hist_nc.to_csv('./hists/{:q}_{}_hist.csv'.format(point, date))
    return(hist_nc)


def calcSWEfromNC(n_cals, n_raws):
    rho = (1.135988 - 0.05)
    theta_g = (0.142 + 0.0595)/rho

    a0 = 0.0808
    a1 = 0.3720
    a2 = 0.1150

    N_0 = n_cals / ((a0/(theta_g*rho+a1))+a1)
    N_wat = 0.24*N_0
    lamb = -4.8

    swes = lamb*np.log((n_raws-N_wat)/(n_cals-N_wat))
    return(swes)


def getINDS(xv, yv):
    xx = (xv+500)/2
    yy = np.abs(yv-500)/2
    return(xx-1, yy-1)


def spatialweighting(xind, yind):
    # create 500 by 500 of distances (in meters)
    xs = np.arange(2,1000.1,2)
    ys = np.arange(2,1000.1,2)

    # center the distances from CRNS location at (xind, yind)
    xs_centered = xs - 2*int(xind)
    ys_centered = 2*int(yind) - ys

    # compute distances
    XX, YY = np.meshgrid(xs_centered, ys_centered)
    dists = np.sqrt(XX**2 + YY**2)
    return(dists)


