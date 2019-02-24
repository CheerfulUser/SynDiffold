# Import the TESS PRF modelling from DAVE
import numpy as np
import sys
sys.path
sys.path.append('./dave/diffimg/')
import tessprf as prf

from PS_image_download import *

from scipy import interpolate

from glob import glob
from astropy.io import fits
from astropy.wcs import WCS

from scipy.ndimage.filters import convolve

def Interp_PRF(X,Y,Camera,CCD):
    pathToMatFile = './data/prf/'
    obj = prf.TessPrf(pathToMatFile)
    PRF = obj.getPrfAtColRow(X, Y, 1,Camera,CCD)
    x2 = np.arange(0,PRF.shape[1]-1,0.01075)
    y2 = np.arange(0,PRF.shape[0]-1,0.01075)

    x = np.arange(0,PRF.shape[1],1)
    y = np.arange(0,PRF.shape[0],1)
    X, Y = np.meshgrid(x,y)

    x=X.ravel()              #Flat input into 1d vector
    y=Y.ravel()

    z = PRF
    z = z.ravel()
    x = list(x[np.isfinite(z)])
    y = list(y[np.isfinite(z)])
    z = list(z[np.isfinite(z)])

    znew = interpolate.griddata((x, y), z, (x2[None,:], y2[:,None]), method='cubic')
    kernal = znew[300:700,300:700]
    return kernal

def Get_TESS_image(Path, Sector, Camera, CCD, Time = None):
    """
    Grabs a TESS FFI image from a directed path.
    Inputs
    ------
    Path: str
        Path to FFIs
    Sector: int
        Sector of the FFI
    Camera: int
        Camera of the FFI
    CCD: int
        CCD of the FFI
        
    Returns
    -------
    tess_image: array
        TESS image
    tess_wcs
        WCS of the TESS image
        
    Raises
    ------
    FileExistsError
        The file specified by the parameters does not exist.
        
    """
    if Time == None:
        File = "{Path}tess*-s{Sec:04d}-{Camera}-{CCD}*.fits".format(Path = Path, Sec = Sector, Camera = Camera, CCD = CCD)
    else:
        File = "{Path}tess{Time}-s{Sec:04d}-{Camera}-{CCD}*.fits".format(Path = Path, Time = Time, Sec = Sector, Camera = Camera, CCD = CCD)

    file = glob(File)
    if len(file) > 0:
        if (len(file) > 1):
            file = file[0]
        tess_hdu = fits.open(file)
        tess_wcs = WCS(tess_hdu[1].header)
        tess_image = tess_hdu[1].data
        return tess_image, tess_wcs
    else:
        raise FileExistsError("TESS file does not exist: '{}'".format(File))
        pass


def Run_convolution(Path,Camera,CCD,PSsize=1000):
	tess_image, tess_wcs = Get_TESS_image(Path,1,Camera,CCD)
	x = tess_image.shape[1]/2
	y = tess_image.shape[0]/2
	keranl = Interp_PRF(x,y,Camera,CCD)
	ra, dec = tess_wcs.all_pix2world(x,y,1)
	print('({},{})'.format(ra,dec))
	size = PSsize
	fitsurl = geturl(ra, dec, size=size, filters="i", format="fits")
	if len(fitsurl) > 0:
		fh = fits.open(fitsurl[0])
		ps = fh[0].data

		test = convolve(ps,kernal)
		np.save('test_PS_TESS.npy',test)
		return 'Convolved'
	else:
		return 'No PS images for RA = {}, DEC = {}'.format(ra,dec)