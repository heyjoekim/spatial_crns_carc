"""
uranos.py Lib for URANOS processing
    Functions specific for data processing of URANOS data
    Historically based on CoRNy <https://git.ufz.de/CRNS/cornish_pasdy/-/blob/master/corny/uranos.py>
    version: 1.0
"""

import numpy as np
from PIL import Image
from scipy.ndimage.measurements import label as scipy_img_label
import pandas
from glob import glob
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Design fancy function that can interpolate through log data
def _log_interp1d(xx, yy, kind='cubic'):
    """
    Interpolate over logscale
    """
    import scipy
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = scipy.interpolate.interp1d(logx, logy, kind=kind)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

class URANOS:
    """
    URANOS class for reading and plotting URANOS output data
    """
    variable_formats = dict({
        'Materials':'.0f',
        'Regions':'.0f',
        'region_id':'.0f',
        'center_mass':'',
        'center_geom':'',
        'area':'.1%',
        'SM':'.0%',
        'SM_diff':'+.0%',
        'Distance_min':'.1f',
        'Distance_com':'.1f',
        'Weights':'.1%',
        'Neutrons':'.0f',
        'Neutrons_diff':'+.0f',
        'Contributions':'.1%',
        'Contributions_diff':'+.1%',
        'Origins':'.1%',
        'Origins_err':'.1%',
        'Origins_below1':'.2%',
        'Density':'.2f',
        'freetext':''
    })
    
    variable_labels = dict({
        'Materials':      'Material code',
        'Regions':      'Regions (id)',
        'region_id':    'Regions (id)',
        'center_mass':  'Center of mass (grid)',
        'center_geom':  'Geometric center (grid)',
        'area':         'Area (%)',
        'SM':           'Soil Moisture (vol.%)',
        'SM_diff':           'Soil Moisture diff (vol.%)',
        'Distance_min': 'shortest Distance (m)',
        'Distance_com': 'Distance to center (m)',
        'Weights':      'Weights (%)',
        'Neutrons':     'estimated Neutrons (a.u.)',
        'Neutrons_diff':     'estimated Neutrons diff (a.u.)',
        'Contributions':'est. Contributions (%)',
        'Contributions_diff':'est. Contributions diff (%)',
        'Origins':      'sim. Contributions (%)',
        'Origins_err':      'sim. Contributions error (%)',
        'Origins_both':      'sim. Contributions (%)',
        'Density':      'sim. Neutron Density (a.u.)',
        'freetext':      'Materials'
    })
    
    def __init__(self, folder='', scaling=2, default_material=None,
                 hum=5, press=1013, verbose=False):
        """
        Initialization
        """    
        self.verbose = verbose
        self.folder = folder
        self.scaling = scaling # one pixel in the data is x meters in reality
        self.center = (249, 249)
        self.default_material = default_material
        self.hum = hum
        self.press = press
        # Initial approximation with sm=20%, will be updated by materials2sm
        #self.footprint_radius = get_footprint(0.2, self.hum, self.press)
    
    #######
    # Input
    #######
    
    def read_materials(self, filename, scaling=None):
        """
        Read Material PNG image
        """
        I = Image.open(self.folder+filename)
        I = self.convert_rgb2grey(I)
        A = np.array(I)
        self.Materials = A
        self._idim = A.shape 
        self.center = ((self._idim[0]-1)/2, (self._idim[1]-1)/2)
        if not scaling is None:
            self.scaling = scaling
        print('Imported map `.Materials` (%d x %d), center at (%.1f, %.1f).'
              % (self._idim[0], self._idim[1], self.center[0], self.center[1]))
        print('  One pixel in the data equals %d meters in reality (%d x %d)'
              % (self.scaling, self._idim[0]*self.scaling, self._idim[1]*self.scaling))
        print('  Material codes: %s' % (', '.join(str(x) for x in np.unique(self.Materials))))
        if self.default_material is None:
            self.default_material = self.Materials[0,0]
            print('  Guessing default material: %d' % self.default_material)
        return(self)
        
    def read_origins(self, filepattern='detectorOrigins*', pad=False):
        """
        Read URANOS origins matrix
        """
        U = None
        for filename in glob(self.folder + filepattern):
            print('  Reading %s' % filename)
            u = np.loadtxt(filename)
            if isinstance(U, np.ndarray):
                U += u
            else:
                U = u
        #U = np.loadtxt(self.folder+filename)
        if U is None:
            print('Error: no files found!')
            return(self)
        U = np.flipud(U)
        if pad:
            U = np.pad(U, ((0,1),(0,1)))
        self.Origins = U
        for i in self.region_data.index:
            U_i = U[self.Regions==i]
            self.region_data.loc[i, 'Origins'] = np.nansum(U_i/np.nansum(U))
            self.region_data.loc[i, 'Origins_err'] = np.nansum(U_i/np.nansum(U)) / np.sqrt(np.nansum(U_i))

        print('Imported URANOS origins as `.Origins` (%d x %d).' % (U.shape[0], U.shape[1]))
        return(self)

    def read_density(self, filepattern='densityMapSelected*', pad=False):
        """
        Read URANOS density matrix
        """
        U = None
        for filename in glob(self.folder + filepattern):
            print('  Reading %s' % filename)
            u = np.loadtxt(filename)
            if isinstance(U, np.ndarray):
                U += u
            else:
                U = u
            
        if U is None:
            print('Error: no files found!')
            return(self)
        
        U = np.flipud(U)
        if pad:
            U = np.pad(U, ((0,1),(0,1)))
        self.Density = U
        for i in self.region_data.index:
            self.region_data.loc[i, 'Density'] = np.mean(U[self.Regions==i])
        self.region_data['Density'] /= self.region_data['Density'].max()

        print('Imported URANOS density map as `.Density` (%d x %d).' % (U.shape[0], U.shape[1]))
        return(self)

    ############
    # Processing
    ############
    
    def material2sm(self, SM_gui=0.2):
        """
        Convert to soil moisture, assuming greyscale number between 2 and 139
        """
        
        self.SM = self.Materials/2/100
        self.SM[self.Materials == 100] = SM_gui  # water
        self.SM[self.Materials == 204] =  0.1  # concrete
        self.SM[self.Materials == 254] = 10.0  # water
        if self.Materials[(self.Materials<2)|(self.Materials>170)|(self.Materials!=204)|(self.Materials!=254)].any():
            print('Warning: some materials do not correspond to soil moisture, they are still naively converted using x/2/100.')
        print('Generated soil moisture map `.SM`, values: %s' % (', '.join(str(x) for x in np.unique(self.SM))))
        
        nearby_avg_sm = self.SM[self.m2grd(-25):self.m2grd(25), self.m2grd(-25):self.m2grd(25)]
        #self.footprint_radius = get_footprint(nearby_avg_sm.mean(), self.hum, self.press)
        self.nearby_avg_sm = nearby_avg_sm
        print('Nearby avg. sm is %.2f +/- %.2f.' % (nearby_avg_sm.mean(), nearby_avg_sm.std()))
        return(self)
    
    def generate_distance(self):
        """
        Distance matrix
        """
        D = np.zeros(shape=(self._idim[0],self._idim[1]))
        for i in range(self._idim[0]):
            for j in range(self._idim[1]):
                D[i,j] = np.sqrt(self.grd2m(i)**2 + self.grd2m(j)**2)
        self.Distance = D
        print('Generated distance map `.Distance`, reaching up to %.1f meters.' % (np.max(D)))
        return(self)
        
    
    def genereate_weights(self, approx=False, exclude_center=False):
        """
        Generate radial weights based on W_r
        """
        
        W = np.zeros(shape=(self._idim[0],self._idim[1]))
        for i in range(self._idim[0]):
            for j in range(self._idim[1]):
                r = 0.5 if self.Distance[i,j] == 0 else self.Distance[i,j] 
                if approx:
                    W[i,j] = Wr_approx(r)/r
                else:
                    W[i,j] = Wr(r, self.nearby_avg_sm.mean(), self.hum)/r
        if exclude_center:
            center_id = np.round(self.center).astype(int) 
            W[:,center_id[0]] = np.mean([W[:,center_id[0]-1], W[:,center_id[0]+1]])
            W[center_id[1],:] = np.mean([W[center_id[1]-1,:], W[center_id[0]+1,:]])
        W_sum = W.sum()
        self.Weights = W/W_sum
        print('Generated areal weights `.Weights`, ranging from %f to %f.' % (self.Weights.min(), self.Weights.max()))
        return(self)
        
    def find_regions(self, default_material=None):
        """
        Identidy connected regions based on the Materials map
        """
        if default_material is None:
            if self.default_material is None:
                default = np.unique(self.Materials)[0]
                print('Guessing default material code: %d' % default)
            else:
                default = self.default_material
        else:
            self.default_material = default_material
            default = default_material
        
        M = np.zeros(shape=(self._idim[0],self._idim[1]), dtype=np.uint8)
        for i in range(self._idim[0]):
            for j in range(self._idim[1]):
                M[i,j] = 0 if self.Materials[i,j] == default else 1
                
        #from scipy.ndimage.measurements import label
        L, ncomponents = scipy_img_label(M)
        self.Regions = L
        self.n_regions = ncomponents
        region_data = pandas.DataFrame(index=pandas.Series(np.arange(ncomponents+1), name='id'),
                                       columns=['Materials', 'center_mass', 'center_geom', 'area', 'SM',
                                                'Distance_min', 'Distance_com', 'Weights', 'Neutrons',
                                                'Contributions', 'Origins', 'Density'])
        region_data['Regions'] = region_data.index
        for i in region_data.index:
            region_data.loc[i, 'Materials'] = np.median(self.Materials[self.Regions==i])
            region_data.at[i, 'center_mass'] = self._get_region_center(i, method='mass')
            region_data.at[i, 'center_geom'] = self._get_region_center(i, method='geom')
            region_data.loc[i, 'area'] = len(self.Regions[self.Regions==i]) / self._idim[0]/self._idim[1]
            if hasattr(self, 'SM'):
                region_data.loc[i, 'SM'] = np.median(self.SM[self.Regions==i])
            if hasattr(self, 'Weights'):
                region_data.loc[i, 'Weights'] = np.sum(self.Weights[self.Regions==i])
            if hasattr(self, 'Distance'):
                region_data.loc[i, 'Distance_min'] = np.min(self.Distance[self.Regions==i])
                region_data.loc[i, 'Distance_com'] = self.Distance[region_data.loc[i, 'center_mass'][0], region_data.loc[i, 'center_mass'][1]]
        
        self.region_data = region_data
        print('Found %d regions, mapped to `.Regions`, DataFrame generated as `.region_data`.' % self.n_regions)
        return(self)
    
    def estimate_neutrons(self, method='Koehli.2021', N0=1000, bd=1.43):
        """
        Estimate neutrons from the input soil moisture map
        """
        
        N = np.zeros(shape=(self._idim[0],self._idim[0]))
        for i in range(self._idim[0]):
            for j in range(self._idim[1]):
                if method=='Desilets.2010':
                    N[i,j] = sm2N(self.SM[i,j], N0, off=0.0, bd=bd)
                elif method=='Koehli.2021':
                    N[i,j] = sm2N_Koehli(self.SM[i,j], h=self.hum, off=0.0, bd=bd, func='vers2', method='Mar21_uranos_drf', bio=0)*N0/0.77
        
        self.Neutrons = N
        print('Estimated neutrons from soil moisture, `.Neutrons` (%.0f +/- %.0f)' % (N.mean(), N.std()))
        self.Contributions = self.Weights*N/np.sum(self.Weights*N)
        print('Estimated their signal contributions, `.Contributions`')
        
        for i in self.region_data.index:
            if not 'Neutrons_diff' in self.region_data.columns:
                self.region_data['Neutrons_diff'] = 0
            self.region_data.loc[i, 'Neutrons_diff'] = np.mean(N[self.Regions==i]) - self.region_data.loc[i, 'Neutrons']
            self.region_data.loc[i, 'Neutrons'] = np.mean(N[self.Regions==i])
            
            if not 'Contributions_diff' in self.region_data.columns:
                self.region_data['Contributions_diff'] = 0
            self.region_data.loc[i, 'Contributions_diff'] = np.sum(self.Contributions[self.Regions==i]) - self.region_data.loc[i, 'Contributions'] 
            self.region_data.loc[i, 'Contributions'] = np.sum(self.Contributions[self.Regions==i])
            
        return(self)

    def modify(self, Region=0, SM=None):
        if not SM is None:
            self.SM[self.Regions==Region] = float(SM)
            if not 'SM_diff' in self.region_data.columns:
                self.region_data['SM_diff'] = 0
            self.region_data.loc[self.region_data.Regions==Region, 'SM_diff'] = float(SM) - self.region_data.loc[self.region_data.Regions==Region, 'SM']
            self.region_data.loc[self.region_data.Regions==Region, 'SM'] = float(SM)
        return(self)
    
    ##################
    # Helper functions
    ##################
    
    def m2grd(self, m):
        g = m / self.scaling + self.center[0]
        if np.isscalar(g):
            return(int(np.round(g)))
        else:
            return(np.round(g).astype(int))
    def grd2m(self, g):
        m = g * self.scaling - self.center[0]*self.scaling
        if np.isscalar(m):
            return(int(np.round(m)))
        else:
            return(np.round(m).astype(int))
    
    def convert_rgb2grey(self, obj):
        """
        Convert an Array of Image from RGB to Greyscale
        """
        if isinstance(obj, np.ndarray):
            A = np.zeros(shape=(obj.shape[0],obj.shape[1]), dtype=np.uint8)
            for i in range(obj.shape[0]):
                for j in range(obj.shape[1]):
                    A[i,j] = np.mean(obj[i,j][0:3])
            return(A)

        elif isinstance(obj, Image.Image):
            img = obj.convert('L')
            return(img)
        
        else:
            print('Error: provide either a numpy array of RGB arrays or an RGB image.')
            
    def _get_region_center(self, region_id, method='mass'):
        """
        Get the center of a region
        """
        indices = np.where(self.Regions==region_id)
        if method=='geom':
            yrange = (indices[0].min(), indices[0].max())
            xrange = (indices[1].min(), indices[1].max())
            center = (0.5*xrange[0]+0.5*xrange[1], 0.5*yrange[0]+0.5*yrange[1])
        elif method=='mass':
            center = (indices[1].mean(), indices[0].mean())
        return([int(np.round(x)) for x in center])
    
    ########
    # Output
    ########
    
    def plot(self, ax=None, image='SM', annotate=None, overlay=None, fontsize=10, title=None, contour=False,
             regions=None, extent=500, cmap='Greys', cmap_scale=2, x_marker=None, cross_alpha=0.5,
             label_offset=(0,0), step_fraction=0.2, colorbar=False, axis_labels=True, interpolation='none'):
        """
        Plot map, annotate, and overlay
        Avoid annotation with annotate='none'
        """
        
        # If no regions are provided, show all regions
        if regions is None:
            regions = np.arange(len(self.region_data))
        
        if annotate is None:
            annotate = image
            
        try:
            Var = getattr(self, image)
        except:
            print('Error: %s is no valid attribute.' % image)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(5,5))
        
        if title is None:
            title = 'Map of %s' % self.variable_labels[image]
            if annotate != image and annotate in self.variable_labels:
                title += '\nAnnotation: %s' % self.variable_labels[annotate]
            if overlay=='Origins':   title += '\nOverlay: sim. Neutron Origins (x)'
            if not x_marker is None: title += '\n\n'
        else:
            title = title
        ax.set_title(title, fontsize=fontsize)
        
        if cmap_scale != 1:
            if isinstance(cmap, str):
                from matplotlib.cm import get_cmap
                cmap = get_cmap(cmap)
            cmap = truncate_colormap(cmap, 0, 1/cmap_scale)

        
        rasterimg = ax.imshow(Var, interpolation=interpolation, cmap=cmap) #, vmax=np.max(Var)*cmap_scale)
        if contour:
            ax.contour(self.Regions, levels=0, colors='k', linewidths=1, antialiased=True)

        if colorbar:
            cax = ax.figure.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
            mycbar = ax.figure.colorbar(rasterimg, cax=cax)
            mycbar.ax.set_ylabel(self.variable_labels[image], rotation=-90, va="bottom")
            mycbar.ax.set_yticklabels(['{0:{1}}'.format(y, self.variable_formats[image]) for y in mycbar.get_ticks()])
            #mycbar.ax.set_yticklabels(['{0:.0f}'.format(y) for y in mycbar.get_ticks()])

        lox, loy = label_offset
        for i in regions:
            mask = (self.Regions == i)
            dataset = self.region_data.loc[i]
            
            cmx, cmy = dataset['center_mass']
            coords = (cmx+lox*i, cmy+loy*i)
            
            # all this clutter only for having 2 digits for origins < 1...
            fmt = self.variable_formats[annotate] if annotate in self.variable_formats else '%s'
            
            if annotate in ['Origins','Origins_err','Origins_both']:
                if dataset['Origins'] < 0.01:
                    fmt     = self.variable_formats['Origins_below1']
                    fmt_err = self.variable_formats['Origins_below1']
                else:
                    fmt     = self.variable_formats['Origins']
                    fmt_err = self.variable_formats['Origins_err']
            elif annotate == 'Contributions':
                if dataset['Contributions'] < 0.01:
                    fmt = self.variable_formats['Origins_below1']
                
            if annotate=='Origins_both':
                ax.annotate('{0:{1}}\n$\pm${2:{3}}'.format(dataset['Origins'], fmt, dataset['Origins_err'], fmt_err),
                    coords, fontsize=fontsize, ha='center', va='center')
            elif annotate in dataset:
                ax.annotate('{0:{1}}'.format(dataset[annotate], fmt),
                    coords, fontsize=fontsize, ha='center', va='center')

        #plt.title(r'$\theta_1=5\,\%$, $\theta_2=10\,\%$, $R=200\,$m')
        # Tick format in meters
        grid100 = np.arange(-extent*(1-step_fraction), extent*(1-step_fraction)+1, extent*step_fraction)
        ax.set_xticks(self.m2grd(grid100), ['%.0f' % x for x in grid100])
        ax.set_yticks(self.m2grd(grid100), ['%.0f' % x for x in grid100[::-1]])
        # Zoom to extend
        ax.set_xlim(self.m2grd(grid100-extent*step_fraction).min(), self.m2grd(grid100+extent*step_fraction).max())
        ax.set_ylim(self.m2grd(grid100+extent*step_fraction).max(), self.m2grd(grid100-extent*step_fraction).min())
        # Tick markers inside
        ax.tick_params(axis="y", direction='in', length=4)
        ax.tick_params(axis="x", direction='in', length=4)
        # Ticks an all sides
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        # Central lines
        #ax.axhline(m2grd(0), color='black', lw=1, ls='--', alpha=0.3)
        #ax.axvline(m2grd(0), color='black', lw=1, ls='--', alpha=0.3)
        ax.plot(self.m2grd(0),self.m2grd(0), color='black', marker='+', ms=10, mew=1, alpha=cross_alpha)
        
        if annotate=='pixels':
            for i in np.arange(self.m2grd(-extent)+1, self.m2grd(extent)):
                for j in np.arange(self.m2grd(-extent)+1, self.m2grd(extent)):
                    ax.text(j, i, int(Var[i, j]), ha="center", va="center", color="black", fontsize=7)
                    
        # Distance marker
        if not x_marker is None:
            ax.annotate('$R=%.0f\,$m' % x_marker, xy=(self.m2grd(x_marker), 0), xytext=(self.m2grd(x_marker), -35),
                        ha='center', va='center', arrowprops={'arrowstyle': 'wedge'}) 
            # for arrows, see https://matplotlib.org/stable/gallery/text_labels_and_annotations/fancyarrow_demo.html

        if not overlay is None:
            try:
                Var = getattr(self, overlay)
            except:
                print('Error: %s is no valid attribute.' % overlay)
            x,y = np.meshgrid(np.arange(Var.shape[1]), np.arange(Var.shape[0]))
            Var[np.where(Var < 1)] = np.nan
            #ax.imshow(X.Origins, interpolation='none')
            ax.scatter(x,y, c=Var, s=24, cmap='autumn', alpha=0.3, marker='x')
        
        # Footprint circle
        #fpc = plt.Circle((self.m2grd(0), self.m2grd(0)), self.footprint_radius/self.scaling, fc='none', ec='black', ls='--', alpha=0.3)
        #ax.add_patch(fpc)
        
        if axis_labels:
            ax.set_xlabel('x (in m)')
            ax.set_ylabel('y (in m)')
        
        return(ax)


    def histogram(self, ax=None, var='SM', layout=None, figsize=None):
        """
        Plot a histogram of a certain variable in .region_data 
        """
        if ax is None:
            if isinstance(var, (list, np.ndarray)):
                if figsize is None:
                    figsize = (5, len(var)+2)
                if layout is None:
                    rows = int(np.ceil(len(var)/2))
                    layout = (rows, 2)
                fig, ax = plt.subplots(nrows=layout[0], ncols=layout[1],
                    layout='tight', figsize=figsize)
            else:
                if figsize is None:
                    figsize = (5, 3)
                fig, ax = plt.subplots(figsize=figsize) #
        
        if not isinstance(ax, (list, np.ndarray)):
            ax = np.array([ax])
        if not isinstance(var, (list, np.ndarray)):
            var = [var]
            
        i = 0
        for ax_i in ax.flatten():
            if i < len(var):
                var_i = var[i]
                if not var_i in self.region_data:
                    print('Error: Variable is not in .region_data')
                    return()
                
                self.region_data[var_i].plot.barh(color='black', ax=ax_i)
                ax_i.grid(axis='x', color='black', alpha=0.3)
                ax_i.invert_yaxis()
                ax_i.set_yticks(range(len(self.region_data)+1))
                ax_i.set_ylabel('Region number')
                ax_i.set_xlabel(self.variable_labels[var_i])
                i += 1
            else:
                ax_i.set_visible(False)
        
        if isinstance(ax, (list, np.ndarray)) and len(ax)==1:
            return(ax[0])
        else:
            return(ax)
    
    def average_sm(self, N0=1000):
        """
        Calculate different approaches to get average CRNS soil moisture
        """
        print('Average soil moisture seen by the CRNS detector:')
        print('{0:.1%}             field mean (naive approach)'
              .format(self.SM.mean()))
        print('{0:.1%} SM-weighted field mean (lazy approach)'
              .format((self.Weights*self.SM).sum()))
        print('{0:.1%}  N-weighted field mean (correct approach)'
              .format(N2SM_Schmidt_single((self.Weights*self.Neutrons).sum()/N0*0.77, bd=1.43, hum=self.hum)))


    ########
    # Hits #
    ########
    def read_hits(self, file='detectorNeutronHitData.dat', soil_contact=False):
        """
        Read detector hits file
        """
        data = pandas.read_csv(self.folder + file, sep="\t")
        # data = data[data.Detector_ID.str.contains('Detector_ID') == False]
        data = data.apply(pandas.to_numeric)
        
        # Angle of origin
        data['x'] = data['x_at_interface_[m]']
        data['y'] = data['y_at_Interface_[m]']
        data['z'] = data['z_at_Interface_[m]']
        data['z_max'] = data['maximum_Depth_[m]']
        data['w'] = np.arctan2( data.y , data.x) + 3.141
        data['w_deg'] = data.w /3.141*180
        # Distance to origin
        data['r'] = np.sqrt(data.x**2 + data.y**2)
        
        # Angle of entry
        data['ex'] = data['previous_x_[m]']
        data['ey'] = data['previous_y_[m]']
        data['we'] = np.arctan2( data.ey , data.ex) + 3.141
        data['we_deg'] = data.we /3.141*180
        
        # Thermal mask
        data['thermal'] = False
        data.loc[data['Energy_[MeV]'] < 0.5e-6, 'thermal'] = True
        data['E'] = data['Energy_[MeV]']
        
        self.Hits = data
        if soil_contact:
            self.only_soil_contact()
            
        return(self)

    def only_soil_contact(self):
        """
        Filter only neutrons with soil contact
        """
        self.Hits = self.Hits[(self.Hits['Soil_Contact']==1) & (self.Hits['z_at_Interface_[m]']>0)]
        return(self)
        
    def drop_multicounts(self):
        """
        Drop multiple counts of the same neutron
        """
        self.Hits = self.Hits.drop_duplicates(subset=['Neutron_Number'], keep='last')
        return(self)
    


    def weight_by_detector_response(self, method='1/sqrt(E)', file=None):
        """
        Add weight column
        """
        if method == '1/sqrt(E)':
            self.Hits['Weight'] = 1/np.sqrt(self.Hits.E)
            # normalize
            self.Hits['Weight'] /= self.Hits.Weight.sum()
            # all other neutrons get no weight
            self.Hits['Weight'] = self.Hits.Weight.fillna(0)    
            
        elif method=='drf':
            if not file is None and os.path.exists(file):
                # Import energy response function
                DERF = pandas.read_csv(file, sep='\t', names=['E','response'], index_col=0)
                #DERF.plot(logx=True, grid=True, marker='o')
                # Generate sampling function with DERF
                self.drf = _log_interp1d(DERF.index, DERF.response)
                # cut 
                self.Hits = self.Hits[(self.Hits.E > DERF.index.min()) & (self.Hits.E < DERF.index.max())]
                # Weight of the neutron
                self.Hits['Weight'] = self.drf(self.Hits.E)
                # normalize
                self.Hits['Weight'] /= self.Hits.Weight.sum()
                # all other neutrons get no weight
                self.Hits['Weight'] = self.Hits.Weight.fillna(0)    
            else:
                print('! The file does not exist.')
        else:
            print('! Method should be either "1/sqrt(E)" or "drf"')
        
        return(self)
    
    def footprint_by_hits(self, var='r', quantile=0.865, thermal=False, weighted=True):
        """
        Calculate the footprint by neutron hits
        """
        
        data = self.Hits[self.Hits.thermal==thermal].sort_values(var)
        data['cumsum'] = data[var].cumsum()
        if weighted:
            data['neutron_count'] = data['Weight'].cumsum()
        else:
            data['neutron_count'] = range(len(data))
        footprint = data.loc[data.neutron_count < quantile * data.neutron_count.values[-1], var].values[-1]
        return(footprint)

    def plot_xy_hits(self, ax=None, thermal=False, footprint=True, quantile=0.865, weighted=True):
        """
        Make xy plot with origins and footprint
        """
        if ax is None:
            fig, ax = plt.subplots(1,1, layout='tight')
        
        data = self.Hits[self.Hits.thermal==thermal]
        ax.set_aspect('equal', adjustable='box')
        ax.scatter(data.x, data.y, s=10, color='C0', alpha=0.5, marker='.', edgecolor='none')
        
        if footprint:
            radius = self.footprint_by_hits('r', quantile=quantile, thermal=thermal, weighted=weighted)
            circle1 = plt.Circle((0, 0), radius, facecolor='none', edgecolor='k', ls='--')
            ax.add_patch(circle1)
        
        ax.plot(0,0, color='k', marker='+', ms=10, mew=1)
        ax.set_xlabel('x (in m)')
        ax.set_ylabel('y (in m)')
        return(ax)
    
    def plot_z_hits(self, ax=None, thermal=False, footprint=True, quantile=0.865, weighted=True):
        """
        Make z plot with footprint
        """
        if ax is None:
            fig, ax = plt.subplots(1,1, layout='tight')
        
        data = self.Hits[self.Hits.thermal==thermal]
        #ax.set_aspect('equal', adjustable='box')
        ax.scatter(data.x, -data.z, s=10, color='C0', alpha=0.5, marker='.', edgecolor='none')
        
        if footprint:
            radius = self.footprint_by_hits('r', quantile=quantile, thermal=thermal, weighted=weighted)
            depth  = self.footprint_by_hits('z', quantile=quantile, thermal=thermal, weighted=weighted)
            t = np.linspace(0, 2*3.141, 100)
            ax.plot(radius*np.cos(t) , depth*np.sin(t), color='k', ls='--')
        
        ax.plot(0,0, color='k', marker='+', ms=10, mew=1)
        ax.set_ylim(-data.z.max(), 0)
        ax.set_ylabel('Depth z (in m)')
        ax.set_xlabel('x (in m)')
        return(ax)
    
    def depth_distribution(self, ax=None, var='z', weighted=True):
        """
        Histogram of z
        """
        if ax is None:
            fig, ax = plt.subplots(1,1, layout='tight')
            
        data = self.Hits[self.Hits.thermal==False]
        if weighted:
            weights = data.Weight
        else:
            weights = np.arange(0,len(data))
        
        hist_kw = dict(kind='hist', ax=ax, bins=np.arange(0,1,0.01),
            weights=weights, density=True)
        data[var].plot(color='C0', alpha=0.2, **hist_kw, label='Epithermal')
        data[var].plot(color='C0', histtype='step', **hist_kw)
        
        data = self.Hits[self.Hits.thermal==True]
        if weighted:
            weights = data.Weight
        else:
            weights = np.arange(0,len(data))
        
        hist_kw = dict(kind='hist', ax=ax, bins=np.arange(0,1,0.01),
            weights=weights, density=True)
        data[var].plot(color='C1', alpha=0.2, **hist_kw, label='Thermal')
        data[var].plot(color='C1', histtype='step', **hist_kw)
        
        ax.set_xlim(0,1)
        ax.set_xlabel(var + ' (in m)')
        ax.grid(color='k', alpha=0.1)
        
        h, l = ax.get_legend_handles_labels()
        ax.legend(handles=[h[0], h[2]], labels=[l[0], l[2]])
        return(ax)
    
    def distance_distribution(self, ax=None, var='r', weighted=True):
        """
        Histogram of r
        """
        if ax is None:
            fig, ax = plt.subplots(1,1, layout='tight')
            
        data = self.Hits[self.Hits.thermal==False]
        if weighted:
            weights = data.Weight
        else:
            weights = np.arange(0,len(data))
        
        hist_kw = dict(kind='hist', ax=ax, bins=np.arange(0,300,5),
            weights=weights, density=True)
        data[var].plot(color='C0', alpha=0.2, **hist_kw, label='Epithermal')
        data[var].plot(color='C0', histtype='step', **hist_kw)
        
        data = self.Hits[self.Hits.thermal==True]
        if weighted:
            weights = data.Weight
        else:
            weights = np.arange(0,len(data))
        
        hist_kw = dict(kind='hist', ax=ax, bins=np.arange(0,300,5),
            weights=weights, density=True)
        data[var].plot(color='C1', alpha=0.2, **hist_kw, label='Thermal')
        data[var].plot(color='C1', histtype='step', **hist_kw)
        
        ax.set_xlim(0,300)
        ax.set_xlabel(var + ' (in m)')
        ax.grid(color='k', alpha=0.1)
        
        h, l = ax.get_legend_handles_labels()
        ax.legend(handles=[h[0], h[2]], labels=[l[0], l[2]])
        return(ax)

    def plot_angle_of_origin(self, ax=None, thermal=False, polar=False, normalize='all'):
        """
        Plot angular origins
        """
        
        from matplotlib.ticker import FormatStrFormatter

        data = self.Hits[self.Hits.thermal==thermal]
        
        if polar:
            from matplotlib.cm import get_cmap
            mesh_kw = dict(cmap=get_cmap('pink_r', 9), vmin=0)
            
            if ax is None:
                fig, ax = plt.subplots(1,1, layout='tight', figsize=(5,5), subplot_kw=dict(polar=True))
                ax.set_title('Intensity relative to the total ring intensity (in %)')
                self._polar_plot(ax, data, 'r', 'w', 'Weight', r=[0, 10, 25, 50, 100, 200],
                    normalize=normalize, **mesh_kw)

        else:
            if ax is None:
                fig, ax = plt.subplots(1,1, layout='tight', figsize=(7,5))
            
            hist_kw = dict(bins=18, color='C0', histtype= u'step', lw=2)
            for r in [0, 10, 50, 100]:
                s = data[data.r > r] 
                s.w_deg.hist(weights=s.Weight, label='$r>%3.0f$ m' % r, alpha=0.2+0.8*r/100, **hist_kw)    
            
            ax.set_title('Angle of origin')
            ax.set_xlabel('Angle of origin of detected neutrons (°)')
            ax.set_xlim(0,360)
            xticks = np.arange(0,360+1,45)
            ax.set_xticks(xticks)
            ax.set_xticklabels(['%i°' % x for x in xticks])
            
            ax.set_ylabel('Histogram (fraction of total counts)')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f%%'))
        
            ax.grid(b=True, color='black', alpha=0.2)
            ax.legend(edgecolor='white', framealpha =1, borderpad=1, frameon=False)
            #wy=0.001, wmin=45, wmax=135
            #ax.text(np.mean([wmin,wmax]), wy*1.25, window_text, ha='center')
            #ax.plot([wmin,wmax],[wy, wy], color='black', lw=1, ls='--')
        
        return(ax)
    
    def _polar_plot(self, ax, data, r_col, a_col, z_col, da=37, r=[0,1],
        title='', normalize='r', colorbar=False, **mesh_kw):
        """
        General setup for polar plots
        """
        # Set domain
        angls = np.linspace(0, 2*np.pi, da)
        radii = np.array(r)
        n_angls = len(angls)
        n_radii = len(radii)
        dw = np.median(np.diff(angls))
        z = np.random.uniform(0.0, 0.0, (n_angls-1, n_radii-1))
        z.shape
        
        # Generate data
        for ir in range(n_radii-1):
            z_r = []
            for ia in range(0,n_angls-1):
                #print('ir=%3.f, ia=%3.f, w=%.3f' % (ir, ia, angls[ia]), end='\r')
                wfilter = (data[a_col] > angls[ia]) & (data[a_col] <= angls[ia+1]) & (data[r_col] > radii[ir]) & (data[r_col] <= radii[ir+1])
                z_r.append(data.loc[wfilter, z_col].sum())
    
            if normalize=='r':
                z_r /= np.sum(z_r)
            for ia in range(0,n_angls-1):
                z[ia, ir] = z_r[ia] * 100
        
        if normalize=='all':
            z /= z.sum()
        
        # Plot
        ax.set_title = title
        ax.grid(False)
        e = ax.pcolormesh(angls, radii, z.T, **mesh_kw)
        if colorbar:
            #draw_colorbar(e, label=title)
            cbar = ax.figure.colorbar(e, pad=0.08, shrink=0.7)
            #mpl.cm.ScalarMappable(norm=norm, cmap='Blues'),
            #ax=ax, pad=.05, extend='both', fraction=fraction)
    
    
        ax.set_xticks(angls[:-1])
        ax.set_rgrids(radii[1:-1], ["$%d\,$m" % r for r in radii[1:-1]], angle=-60, ha='center', va='top')
        ax.grid(axis='y', which='major', color='black', alpha=0.3)
        ax.scatter(0,0, marker='+', lw=1, s=100, color='k')


"""
Additional functions from CoRNy exdata.uranos
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def ReadURANOSmatrix(filename, size):
    """
    Read a single Matrix file
    """
    return(np.loadtxt(filename, dtype="int", delimiter="\t", usecols=range(size-1)))

def ReadURANOS(filename, size=500):
    """
    Read list of matrix files
    """
    #clear_output(wait=True)
    print("Reading file: %s" % filename)

    # Auto-detect zip archives
    from os.path import splitext
    basename, ext = splitext(filename)

    if ext == '.zip':

        from io import BytesIO
        from zipfile import ZipFile
        Z = ZipFile(filename)

        # Generate an empty matrix and add all files
        A = [[0 for x in range(size-1)] for y in range(size-1)]
        # Loop through files
        num_matrices = 0
        for fileitem in Z.namelist():
            if 'densityMap' in str(fileitem) and str(fileitem).endswith('.csv'):
                F = Z.read(fileitem)
                A += ReadURANOSmatrix(BytesIO(F), size)
                num_matrices += 1
        
        A = A / num_matrices
                    
    # or just read a single file without unzip
    else:
        A = ReadURANOSmatrix(filename, size)

    #clear_output(wait=True)
    return(A)


def icenter(matrix):
    """
    Determine the center coordinates
    """
    return([matrix.shape[1]//2, 1+matrix.shape[0]//2])

def ccrop(matrix, rx=0, ry=0, shift=[0,0], dropx=0, dropy=0):
    """
    Function to crop inner center matrix with radius rx,ry, from big matrix
    """
    c = icenter(matrix)
    startx = c[1] - rx + shift[0]
    starty = c[0] - ry + shift[1]
    return matrix[starty:starty+1+ry*2-dropy,startx:(startx+1+rx*2-dropx)]


def extractgrid(matrix, center=None, shift=[0]):
    """
    Function to crop inner center matrix with radius rx,ry, from big matrix
    """
    if center==None:
        center = icenter(matrix)
    A = []
    for sx in shift:
        for sy in shift:
            A.append(matrix[center[0]+sx, center[1]+sy])
    return(A)


def asum(A, B):
    """
    Sum up the contents of two arrays
    """
    return([sum(x) for x in zip(*[A,B])])

def neutronmap(matrix,             # input matrix
               zrange=[None,None], # range of the color axis
               interpolation=None, # 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
               cmap="coolwarm",    # color map, https://matplotlib.org/examples/color/colormaps_reference.html
               cbar=False,         # color bar is off by default
               cbarlabel="Simulated neutrons", # color bar label
               text=True,          # Show text annotations
               contourdata=None,   # URANOS input image file for contour annotations
               contourlevels=np.arange(0,255,50) # contour levels
              ):
    """
    Old function to generate heatmap
    """
    xdim = matrix.shape[1]
    ydim = matrix.shape[0]

    fig, ax = plt.subplots(figsize = (15, 10))
    im = ax.imshow(matrix, cmap=cmap, vmin=zrange[0], vmax=zrange[1], interpolation=interpolation)

    if cbar:
        mycbar = ax.figure.colorbar(im, ax=ax)
        mycbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set_xticks(np.arange(xdim))
    ax.set_yticks(np.arange(ydim))
    ax.set_xticklabels(np.arange(xdim)-xdim//2)
    ax.set_yticklabels(np.arange(ydim)*-1+ydim//2)

    # Create text annotations.
    if text:
        for i in range(ydim):
            for j in range(xdim):
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")

    # Contours from an image source
    if not contourdata is None:
        ax.contour(Image2Array(contourdata, xdim//2, ydim//2),
                   levels=contourlevels, colors='black', linewidths=1)

    plt.tight_layout()
    plt.show()


def Image2Array(filename, rx, ry):
    """
    Load URANOS input image and convert to 2d array
    """
    from PIL import Image
    I = Image.open(filename)
    I = I.convert('L')      # Convert to L=greyscale, 1=black&white
    IA = np.asarray(I)       # Creates an array, white pixels==True and black pixels==False
    IA = np.flip(IA, 0)
    IA = IA[0:-1,0:-1]
    IA = ccrop(IA, rx, ry)
    return(IA)


def stats(x, precision=1, raw=False):
    """
    Statistics summary of a matrix
    """

    # Convert potential lists to arrays
    if isinstance(x, list):
        x = np.asarray(x)

    # Format string
    s = "%."+str(precision)+"f"
    if raw:
        return([float("%0.1f" % x.mean()), float("%0.1f" % x.std())])
    else:
        s = s+" +/- "+s
        print(s % (x.mean(), x.std()))

def collect_results(files, size=500, norm_id=0,
                    radius=[200,200], dropx=0, dropy=0,
                    repeat_shift=None, repeat_center=[0,0]):
    """
    Note:
        Set norm_id = -1 if no normalization should be done
    Usage:
        files = [
            'a.csv',
            'b.csv',
            'c.csv']
        data, cntr = collect_results(files, size=1000, radius=[1,499])
    """

    # Read files and fill arrays
    data = []
    #cntr = []
    i = 0
    norm = 1
    if isinstance(files, str):
        files = [files]
        
    for filename in files:

        if os.path.isfile(filename):
            # Load whole matrix
            matrix = ReadURANOS(filename, size) # read file to matrix

            if not repeat_shift is None:
                # extract pixels in a repeating grid
                #repeat_shift = np.arange(-200, 201, 10)
                excerpt  = extractgrid(matrix, center=asum(icenter(matrix), repeat_center), shift=repeat_shift) # start at center
            else:
                # Extract area from centre
                excerpt = ccrop(matrix, radius[0], radius[1], dropx=dropx, dropy=dropy)

            # Normalize all data relative to one
            if i==norm_id:
                norm = np.mean(excerpt)
            excerpt = [x / norm for x in excerpt]

            data.append(np.asarray(excerpt).flatten()) # append flat array to data
            #cntr.append(ccrop(matrix/norm).item())      # append center pixel
        else:
            data.append(np.array([np.nan]))
            
        i += 1

    if norm_id>=0:
        print("Mean values: 1, " + ', '.join(["%+.2f%%" % (100*(d.mean()-1)) for d in data[1:]]))
    else:
        print("Mean values: " + ', '.join(["%.1f" % d.mean() for d in data]))
    return(data)

def boxplot_results(data, labels=None, width=0.3, x_offset=0, ax=None,
                    title="Detector count average", y_label='Relative neutron counts',
                    y_ticks=None, color='white', figsize=(7,5)):
    """
    Usage:
        ax = boxplot_results(data, y_ticks=np.arange(0,1.5,0.1))
    """

    if ax is None:
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(wspace=0.2)
        ax = fig.add_subplot()

    ax.set_title(title)
    xpos = np.arange(1,len(data) +1)
    box = ax.boxplot(data, showfliers=False, medianprops=dict(linestyle='--', color='black'),
                     positions=xpos+x_offset, widths=width, patch_artist=True)
    for patch in box['boxes']: patch.set_facecolor(color)

    if labels is None:
        labels_default = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        labels = [labels_default[i] for i in range(len(data))]
    elif len(labels) != len(data):
        print('ERROR: numbers of labels (%d) and provided data (%d) are different!' % (len(labels), len(data)))

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)

    if y_ticks:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(["%.1f" % y for y in y_ticks])
        ax.set_ylim(y_ticks[0], y_ticks[-1])
    ax.set_ylabel(y_label)
    ax.grid(b=True, axis='y', color='k', alpha=0.2, zorder=0)

    return(ax, box)
    #leg = plt.legend([box1["boxes"][0], box2["boxes"][0]], ['10 %', '50 %'],
    #                 loc='upper right', title='Soil water content', framealpha=1,
    #                 facecolor='white', edgecolor='white')
    #leg._legend_box.align = "left" # title alignment

def barplot_results(data, labels=None, width=0.3, x_offset=0, ax=None,
                    title="Detector count average", y_label='Relative neutron counts',
                    y_ticks=None, color='C0', figsize=(7,5),
                    text=None):
    """
    Usage:
        ax = barplot_results(data)
    """

    if ax is None:
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(wspace=0.2)
        ax = fig.add_subplot()

    ax.set_title(title)
    xpos = np.arange(1,len(data) +1)

    bar = plt.bar(xpos+x_offset, [np.mean(x) for x in data], width=width, color=color)
    if not text is None:
        for i in xpos:
            plt.text(i, np.mean(data[i-1])+0.02, text[i-1], horizontalalignment='center', color=color)

    if labels is None:
        labels_default = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        labels = [labels_default[i] for i in range(len(data))]
    elif len(labels) != len(data):
        print('ERROR: numbers of labels (%d) and provided data (%d) are different!' % (len(labels), len(data)))

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)


    if y_ticks:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(["%.1f" % y for y in y_ticks])
        ax.set_ylim(y_ticks[0], y_ticks[-1])
    ax.set_ylabel(y_label)
    ax.grid(b=True, axis='y', color='k', alpha=0.2, zorder=0)

    return(ax, bar)
    #leg = plt.legend([box1["boxes"][0], box2["boxes"][0]], ['10 %', '50 %'],
    #                 loc='upper right', title='Soil water content', framealpha=1,
    #                 facecolor='white', edgecolor='white')
    #leg._legend_box.align = "left" # title alignment

    
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Truncates a cmap between 0 and 1.
    Usage:
        new_cmap = truncate_colormap(cmap, 0, 0.5)
    Thanks to:
        https://stackoverflow.com/a/18926541/2575273
    """
    import matplotlib.colors as colors

    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    
    return(new_cmap)