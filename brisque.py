import numpy as np
import cv2
from scipy.stats import gennorm
from scipy.special import gamma

def LocalVariance(im, kernelWidth, sigma):
    # Compute local mean and variance
    mean = cv2.GaussianBlur(im, (7, 7), 7/6)

    # Variance = sqrt ( average(im^2) - (average(im))^2 )
    variance = np.sqrt(cv2.GaussianBlur(im * im, (7, 7), 7/6) - mean**2)
    return (mean, variance)

def MscnCoefficients(im, mean, variance):
    outputIm = (im - mean) / (variance + 1)
    return outputIm

def GetMscnOrientations(mscn):
    if (len(mscn.shape) < 2):
        return None, None, None, None

    rows = mscn.shape[0]
    cols = mscn.shape[1]

    h = np.zeros_like(mscn)
    v = np.zeros_like(mscn)
    d1 = np.zeros_like(mscn)
    d2 = np.zeros_like(mscn)
    for i in range(0, rows-1):
        for j in range(1, cols-1):
            h[i, j] = mscn[i, j] * mscn[i, j+1]
            v[i, j] = mscn[i, j] * mscn[i+1, j]
            d1[i, j] = mscn[i, j] * mscn[i+1, j+1]
            d2[i, j] = mscn[i, j] * mscn[i+1, j-1]

    # Fill in border (missing) values
    """
    h[:, 0] = h[:, 1]
    v[:, 0] = v[:, 1]
    d1[:, 0] = d1[:, 1]
    d2[:, 0] = d2[:, 1]
    """
    return h, v, d1, d2

"""
Fit symmetric GGD to MSCN coefficients using scipy.stats.gennorm (MLE).
Returns: (beta, alpha, loc)
  - beta = shape parameter (same as BRISQUE beta)
  - alpha = scale parameter (same as BRISQUE alpha)
  - loc  = location (should be ~0 for MSCN; optionally fixed to 0)
"""
def FitGGD(mscn):
    x = mscn.ravel()
    x = x[np.isfinite(x)]

    # fix loc=0 because MSCN is approximately zero-mean
    beta, loc, scale = gennorm.fit(x, floc=0)

    # scipy's gennorm parameterization: pdf ~ beta/(2*scale*Gamma(1/beta)) * exp(-(abs(x-loc)/scale)**beta)
    alpha = scale
    return beta, alpha, loc

"""
Fits an AGGD to a pairwise product of MSCN coefficients (aka "orientation")
Input: 
Returns: (eta, beta, alpha_left, alpha_right)
  - alpha = shape parameter 
  - beta_l = scale parameter, left-side (for negative values)
  - beta_r  = scale parameter, right-side scale (for positive values)
  - mu = asymmetry parameter
  
  References: 
  [1] Anish Mittal; Anush K. Moorthy; Alan C. Bovik; Blind/Referenceless Image Spatial Quality Evaluator, 
  Department of Electrical and Computer Engineering University of Texas at Austin, 2011 
  [2] Lasmar NE; Stitou Y; Berthoumieu Y; Multiscale skewed heavy tailed model for texture analysis. 
  IEEE Int’l Conf. Image Proc. (ICIP), 2009
  """
def FitAGGD(x):
    # r = m_1 ^ 2 / mu_2 (2nd order absolute moment squared / 1st order moment)
    # gamma = beta_l / beta_r
    # Estimate beta_l =
    # rho(alpha) = # generalized gaussian ratio function

    # Split pairwise product into left and right halves where left: x < 0 and right: x >= 0
    left = x[x < 0]
    right = x[x >= 0]

    # Calculate gamma using:
    # g = sqrt ( 1 / N_l * sum ( x_k^2 ) for k = 1 to k = 0 - N_l ) /
    #         sqrt ( 1 / N_r * sum ( x_k^2 ) for k = 1 to k = 0 - N_r )
    # Ref [2] Equation 6
    N_l = len(left)
    N_r = len(right)
    g_l = np.sqrt( 1/N_l * np.sum(left * left))
    g_r = np.sqrt( 1/N_r * np.sum(right * right))
    g = g_l / g_r

    # Estimate r using:
    # r = sum ( x_k ) ^ 2 / sum (x_k ^ 2) for k = 1 to k = N_l + N_r
    # Ref [2] Equation 7
    # Also note: r = R * rho
    r = np.sum(x) ** 2 / np.sum(x * x)

    # Calculate R using gamma and r
    # r is also = R * rho
    # Ref [2] Equation 8
    R  = r * ( (g ** 3 + 1) * (g + 1) ) / ( (g ** 2 + 1) ** 2)

    # Calculate alpha_l, alpha_r using:
    # rho(R) = Gamma(2/R)^2/(Gamma(1/R)*Gamma(3/R)) - Ref [2] Equation 5
    # alpha = inverse_rho(R) = (Gamma(1/R)*Gamma(3/R)) / Gamma(2/R)^2 - Ref [2] Equation 9 + 5
    # Gamma = the gamma function
    alpha = ( gamma(1/R) * gamma(3/R) ) / gamma(2/R)**2

    # Calculate beta_l and beta_r using g and alpha
    # Ref[2] Equation 10
    beta_l = g_l * np.sqrt( gamma(3/alpha) / gamma(1/alpha) )
    beta_r = g_r * np.sqrt( gamma(3/alpha) / gamma(1/alpha) )

    # Calculate mu (asymmetry parameter) using:
    mu = ( beta_r - beta_l ) * ( gamma(2/alpha) / gamma (1/alpha) )

    return alpha, beta_l, beta_r, mu

def FitAGGDLookup(orientation):
    x = orientation.ravel()

    # Separate into positive and negative component
    left = x[ x < 0 ]
    right = x[ x >= 0 ]

    # Compute mean of the square for left and right sides
    # Determines width (mean) of each side
    sqm_left = np.mean(left**2)
    sqm_right = np.mean(right**2)

    # Compute mean-abs and mean-sq for Beta calculation
    # The ratio helps determine shape (is there a tail or is it narrowly centered around the peak?)
    mean_abs = np.mean(np.abs(orientation))
    mean_sq = np.mean(orientation * orientation)

    # Compute moment ratio
    R = (mean_abs ** 2) / (mean_sq + 1e-12)

    # Pre-compute theoretical ratio for a range of Beta values
    betas = np.linspace(0.2, 10, 200)
    rbeta = (gamma(2 / betas) ** 2) / (gamma(1 / betas) * gamma(3 / betas))

    # Choose the Beta whose moment ratio is closest to ours
    beta = betas[np.argmin(np.abs(rbeta - R))]

    # Compute alpha_l and alpha_r
    alpha_l = np.sqrt(sqm_left * gamma(1 / beta) / gamma(3 / beta))
    alpha_r = np.sqrt(sqm_right * gamma(1 / beta) / gamma(3 / beta))

    # Compute eta (asymmetry)
    eta = (alpha_r - alpha_l) * (gamma(2 / beta) / gamma(1 / beta))

    return eta, alpha_l, alpha_r, beta

"""
Returns a 16 element list representing the 16 features defined in the BRISQUE method
"""
def GetFeatureVector16x1(mscn):

    # Fit GGD to mscn to obtain first 2 features
    (shape, scale, loc) = FitGGD(mscn)

    # Get pairwise products (orientations)
    (h, v, d1, d2) = GetMscnOrientations(mscn)

    # Fit AGGD to each of 4 orientations (4 params / orientation = 16 features)
    (h_alpha, h_beta_l, h_beta_r, h_mu) = FitAGGD(h)
    (v_alpha, v_beta_l, v_beta_r, v_mu) = FitAGGD(v)
    (d1_alpha, d1_beta_l, d1_beta_r, d1_mu) = FitAGGD(d1)
    (d2_alpha, d2_beta_l, d2_beta_r, d2_mu) = FitAGGD(d2)

    return [shape, scale,
            h_alpha, h_beta_l, h_beta_r, h_mu,
            v_alpha, v_beta_l, v_beta_r, v_mu,
            d1_alpha, d1_beta_l, d1_beta_r, d1_mu,
            d2_alpha, d2_beta_l, d2_beta_r, d2_mu]

def GetFeatureVector36x1(mscn):
    features = np.zeros((36,))

    # Downsample mscn
    newRows = int(mscn.shape[0] / 2)
    newCols = int(mscn.shape[1] / 2)
    mscn_downsampled = cv2.resize(mscn, (newRows, newCols), interpolation=cv2.INTER_LINEAR)

    features[:18] = GetFeatureVector16x1(mscn)
    features[18:] = GetFeatureVector16x1(mscn_downsampled)
    return features

"""
Implements the "BRISQUE" metric - Blind/Referenceless Image Spatial Quality Evaluator
The score ranges from 0 - 100 (0 = best quality, 100 = worst quality)
Input: Image to be evaluated
Returns: score
"""
def GetBrisqueScore(img):
    gray = img
    if (len(img.shape) > 2):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    (mean, variance) = LocalVariance(gray, 7, 7/6)
    mscn = MscnCoefficients(gray, mean, variance)

    #(h, v, d1, d2) = GetMscnOrientations(mscn)
    #orientations = [h, v, d1, d2]
    featureVector = GetFeatureVector36x1(mscn)

    cv2.imshow("average img", mean.astype(np.uint8))
    cv2.waitKey(0)

    return