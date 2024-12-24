# Weight functions import from R to Python from supplemental materials of
# Schron et al (2017)
# also note: I believe these are all the revised functions. They come to
# the conclusion that revised over conventional
## Horizontal Weights
import numpy as np
import pyproj


def WrX(r, x, y):
    # 0 < r < 1 m
    x00 = 3.7
    a00 = 8735; a01 = 22.689; a02 = 11720; a03 = 0.00978; a04 = 9306; a05 = 0.003632   
    a10 = 2.7925e-002; a11 = 6.6577; a12 = 0.028544; a13 = 0.002455; a14 = 6.851e-005; a15 = 12.2755
    a20 = 247970; a21 = 23.289; a22 = 374655; a23 = 0.00191; a24 = 258552 
    a30 = 5.4818e-002; a31 = 21.032; a32 = 0.6373; a33 = 0.0791; a34 = 5.425e-004 
    x0 = x00
    A0 = (a00*(1+a03*x)*np.exp(-a01*y)+a02*(1+a05*x)-a04*y)
    A1 = ((-a10+a14*x)*np.exp(-a11*y/(1+a15*y))+a12)*(1+x*a13)
    A2 = (a20*(1+a23*x)*np.exp(-a21*y)+a22-a24*y)
    A3 = a30*np.exp(-a31*y)+a32-a33*y+a34*x
    return((A0*(np.exp(-A1*r))+A2*np.exp(-A3*r))*(1-np.exp(-x0*r)))


def WrA(r,x,y):
    # 1 m < r < 50 m
    a00 = 8735; a01 = 22.689; a02 = 11720; a03 = 0.00978; a04 = 9306; a05 = 0.003632   
    a10 = 2.7925e-002; a11 = 6.6577; a12 = 0.028544; a13 = 0.002455; a14 = 6.851e-005; a15 = 12.2755
    a20 = 247970; a21 = 23.289; a22 = 374655; a23 = 0.00191; a24 = 258552 
    a30 = 5.4818e-002; a31 = 21.032; a32 = 0.6373; a33 = 0.0791; a34 = 5.425e-004 
    
    A0 = (a00*(1+a03*x)*np.exp(-a01*y)+a02*(1+a05*x)-a04*y)
    A1 = ((-a10+a14*x)*np.exp(-a11*y/(1+a15*y))+a12)*(1+x*a13)
    A2 = (a20*(1+a23*x)*np.exp(-a21*y)+a22-a24*y)
    A3 = a30*np.exp(-a31*y)+a32-a33*y+a34*x
    return(A0*(np.exp(-A1*r))+A2*np.exp(-A3*r))


def WrB(r,x,y):
    # 50 m < r < 350 m
    b00 = 39006; b01 = 15002337; b02 = 2009.24; b03 = 0.01181; b04 = 3.146; b05 = 16.7417; b06 = 3727
    b10 = 6.031e-005; b11 = 98.5; b12 = 0.0013826
    b20 = 11747; b21 = 55.033; b22 = 4521; b23 = 0.01998; b24 = 0.00604; b25 = 3347.4; b26 = 0.00475 
    b30 = 1.543e-002; b31 = 13.29; b32 = 1.807e-002; b33 = 0.0011; b34 = 8.81e-005; b35 = 0.0405; b36 = 26.74 
    
    B0 = (b00-b01/(b02*y+x-0.13))*(b03-y)*np.exp(-b04*y)-b05*x*y+b06
    B1 = b10*(x+b11)+b12*y
    B2 = (b20*(1-b26*x)*np.exp(-b21*y*(1-x*b24))+b22-b25*y)*(2+x*b23)
    B3 = ((-b30+b34*x)*np.exp(-b31*y/(1+b35*x+b36*y))+b32)*(2+x*b33) 
    return(B0*(np.exp(-B1*r))+B2*np.exp(-B3*r))


## Vertical
def D86(r, bd, y):
    # Penetration Depth 86%
    return(1/bd*(8.321+0.14249*(0.96655+np.exp(-0.01*r))*(20+y)/(0.0429+y)))

def Wd(d, r, bd, y):
    # Vertical weights (in cm)
    return(np.exp(-2*d/D86(r, bd, y)))


## Rescaled distance
def rscaled(r, p, Hveg, y):
    Fp = 0.4922/(0.86-np.exp(-p/1013.25))
    Fveg = 1-0.17*(1-np.exp(-0.41*Hveg))*(1+np.exp(-9.25*y))
    return(r / Fp / Fveg )  


# Convert CRNS positions from Lat/Lons to DTM
DTM = pyproj.Proj("+proj=utm +zone=12 +datum=NAD83 +units=m +no_defs")
CRNS = pyproj.Proj("+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0")
CRNSlon, CRNSlat = -109.955567, 47.059422
UTMx, UTMy = DTM(CRNSlon, CRNSlat)

