# Import the TESS PRF modelling from DAVE
import numpy as np
import sys
sys.path
sys.path.append('./dave/diffimg/')
import tessprf as prf

from scipy import interpolate

from glob import glob
from astropy.io import fits
from astropy.wcs import WC

from scipy.ndimage.filters import convolve




def testRowFrac(datapath):        
    """Test that changing column fraction moves flux around"""

    obj = prf.TessPrf(datapath)
    
    img1 = obj.getPrfAtColRow(123.0, 456, 1,1,1)
    
    for frac in np.linspace(0, .9, 11):
        img2 = obj.getPrfAtColRow(123.0, 456.0 + frac, 1,1,1)
        delta = img2 - img1
        
        prfPlot(img1, delta)
        
        #For TESS, PRFs are 13x13. Check the flux near the centre
        #is moving from lower columns to higher ones
        assert delta[6,6] >= 0, delta[6,6]
        assert delta[7,6] >= 0, delta[7,6]
        assert delta[5,6] <= 0, delta[5,6]



def Interp_PRF(X,Y,Camera,CCD):
    pathToMatFile = './data/prf/'
    obj = prf.TessPrf(pathToMatFile)
    PRF = obj.getPrfAtColRow(123.0, 456, 1,Camera,CCD)
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
	prf = Interp_PRF(x,y,Camera,CCD)
	ra, dec = tess_wcs.all_pix2world(x,y,1)

	size = PSsize
	fitsurl = geturl(ra, dec, size=size, filters="i", format="fits")
	fh = fits.open(fitsurl[0])

	ps = fh[0].data

	test = convolve(ps,prf)


	np.save('test_PS_TESS.npy',test)

	return 'Convolved'