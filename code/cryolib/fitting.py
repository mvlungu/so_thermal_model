import numpy as np
import numexpr as ne
import scipy.optimize as opt

from numpy import pi
from scipy import special

### DEFINE ROUTINES ###

# A polynomial basis

def poly_basis(x, n):
  a = np.arange(0,n+1).reshape(n+1,1)
  A = ne.evaluate('x**a').T

  return A

# A 2D polynomial basis

def poly_basis_2D(x, y, n):
  a = np.array([])
  b = np.array([])

  for i in range(n+1):
    for j in range(n+1):
      if (i+j) < (n+1):
        a = np.append(a,i)
        b = np.append(b,j)

  a = a.reshape(len(a),1)
  b = b.reshape(len(b),1)
  A = ne.evaluate('(x**a)*(y**b)').T

  return A

# A 3D polynomial basis

def poly_basis_3D(x, y, z, n):
  a = np.array([])
  b = np.array([])
  c = np.array([])

  for i in range(n+1):
    for j in range(n+1):
      for k in range(n+1):
        if (i+j+k) < (n+1):
          a = np.append(a,i)
          b = np.append(b,j)
          c = np.append(c,k)

  a = a.reshape(len(a),1)
  b = b.reshape(len(b),1)
  c = b.reshape(len(c),1)
  A = ne.evaluate('(x**a)*(y**b)*(z**c)').T

  return A

# A Legendre polynomial basis

def legendre_basis(x, n):
  A = np.zeros((len(x),n+1))
  xx = x.copy()
  xx -= xx.min()
  xx /= xx.max()/2
  xx -= 1

  for i in range(n+1):
    A[:,i] = special.eval_legendre(i,xx)

  return A

# A 2D Chebyshev polynomial basis

def chebyshev_basis_2D(x,y,n):
  a = np.array([])
  b = np.array([])

  for i in range(n+1):
    for j in range(n+1):
      if (i+j) < (n+1):
        a = np.append(a,i)
        b = np.append(b,j)

  A = np.zeros((len(x),len(a)))
  xx = x.copy() - x.min()
  xx /= xx.max()/2
  xx -= 1
  yy = y.copy() - y.min()
  yy /= yy.max()/2
  yy -= 1

  for i in range(len(a)):
    A[:,i] = special.eval_chebyt(a[i],xx)*special.eval_chebyt(b[i],yy)

  return A

# A Fourier mode basis

def fourier_basis(x, n, P = 2*pi):
  A = np.zeros((len(x),2*n+1))
  w = np.arange(1,n+1).reshape(n,1)
  w = ne.evaluate('w*(2*pi/P)*x').T

  A[:,0] = 1
  A[:,1:n+1] = ne.evaluate('cos(w)')
  A[:,n+1:] = ne.evaluate('sin(w)')

  return A

# A standard linear least-squares fit. Model is a matrix of basis vectors. The 
# optional argument Sigma may be an uncertainty matrix, a vector of 
# uncertainties for each data point, or a single uncertainty for everything.

def linear_lsq(y, model, sigma = None, approximate = False):
  I = np.diag(np.ones(model.shape[1]))

  if type(sigma) in ['list', np.ndarray]:
    sigma = np.array(sigma)

    if len(sigma.shape) == 1 or any(np.array(sigma.shape) == 1):
      NA = (np.diag(1.0/sigma**2)).dot(model)
    else:
      NA = np.linalg.solve(sigma,np.eye(sigma.shape[0])).dot(model)

  elif sigma is not None:
    NA = model/(sigma**2)

  else:
    NA = model

  if approximate:
    try:
      Sigma_p = np.linalg.solve(model.T.dot(NA),I)
    except np.linalg.LinAlgError:
      Sigma_p = np.linalg.lstsq(model.T.dot(NA),I)[0]
  else:
      Sigma_p = np.linalg.solve(model.T.dot(NA),I)

  p = Sigma_p.dot(y.T.dot(NA))

  if sigma is None:
    return p
  else:
    return p, Sigma_p

# A least-squares wrapper for the scipy routine. Optionally mask the data
# to do a fit on a subset of the data. Add an uncertainty estimate (sigma) to 
# compute the reduced Chi^2 statistic.

def leastsq(X, Y, model, p0, sigma = None, mask = None, vary = None, \
            mkwargs = {}, **kwargs):
  p0 = np.array(p0)

  # Let all model parameters vary by default

  if vary is None:
    vary = np.ones_like(p0, dtype = bool)
  else:
    vary = np.array(vary, dtype = bool)

  # Should we compute a chi^2?

  if sigma is None:
    do_chi2 = False
  else:
    do_chi2 = True

  # Mask the data?

  if mask is not None:
    mask = np.array(mask, dtype = bool)

    if len(Y.shape) == 1:
      Y = Y[mask]
    else:
      Y = Y[:,mask]

    if sigma is not None:
      if type(sigma) in [list, np.ndarray]:
        sigma = np.array(sigma)
        sigma = sigma[mask]

  # Compute the least-squares fit 

  resfunc = FixedResiduals(vary, p0)
  fit = opt.leastsq(resfunc, p0[vary], args = (model, X, Y, mkwargs, mask), \
                    full_output = do_chi2, **kwargs)

  pfit = p0.copy()
  pfit[vary] = fit[0]

  if do_chi2:
    res = fit[2]['fvec']
    red_chi2 = (res**2/sigma**2).sum()/(len(res)-len(p0[vary])-1)
    return pfit, red_chi2

  else:
    return pfit

# Returns residuals given a model

def residuals(p, model, X, Y, mkwargs, mask = None):
  X = np.array(X)
  Y = np.array(Y)

  try:
    Z = model(X, p, **mkwargs)
  except TypeError:
    Z = model(*X, p = p, **mkwargs)

  if len(Y.shape) > 1:
    dim = Y.shape[0]
  else:
    Y = Y.reshape(1,Y.size)
    Z = Z.reshape(1,Z.size)
    dim = 1

  if mask is None:
    mask = np.ones(Y.shape[1], dtype = bool)

  y = Y[0]
  z = Z[0][mask]

  for i in range(1,dim):
    y = np.append(y, Y[i])
    z = np.append(z, Z[i][mask])

  res = y - z
  #print p ### REMOVE ME
  return res

def get_chi2(res, cov):
  res = res.flatten()

  if len(cov.shape) == 1 or np.any(cov.shape == 1):
    chi2 = ((res/cov)**2).sum()
  else:
    res = res.reshape(res.size,1)
    chi2 = res.T.dot(np.linalg.inv(cov)).dot(res)[0,0]

  return chi2

# A new MCMC solver... what can it do?

def mcmc(X, Y, cov, model, p0, nstep, scale = None, mask = None, vary = None, \
         mkwargs = {}):
  p0 = np.array(p0)

  # Let all model parameters vary by default

  if vary is None:
    vary = np.ones_like(p0, dtype = bool)
  else:
    vary = np.array(vary, dtype = bool)

  # Mask the data?

  if mask is not None:
    mask = np.array(mask, dtype = bool)

    if len(Y.shape) == 1:
      Y = Y[mask]
    else:
      Y = Y[:,mask]

    if len(cov.shape) == 1:
      cov = cov[mask]
    else:
      cov = cov[mask,mask]

  # Set the default step scale to 1

  if scale is None:
    scale = np.ones_like(p0)
  else:
    scale = np.array(scale)

  # Initial position

  chain = np.zeros((nstep,vary.sum()+1))
  resfunc = FixedResiduals(vary, p0)

  p = p0[vary]
  res = resfunc(p, model, X, Y, mkwargs, mask = None)
  chi2 = get_chi2(res,cov)
  chain[0] = np.append(p,chi2)

  # Run through the chain

  for i in range(1,nstep):
    p_new = p + scale[vary]*np.random.randn(len(p))
    res = resfunc(p_new, model, X, Y, mkwargs, mask = None)
    chi2_new = get_chi2(res,cov)
    dchi2 = chi2_new - chi2

    if dchi2 < 0:
      p = p_new
      chi2 = chi2_new

    else:
      prob = np.exp(-0.5*dchi2)

      if np.random.rand() < prob:
        p = p_new
        chi2 = chi2_new

    chain[i] = np.append(p,chi2)

  return chain


### DEFINE CLASSES ###

# Residuals wrapper that allows certain parameters to be kept fixed

class FixedResiduals:

  def __init__(self, vary, p0):
    self.vary = np.array(vary)
    self.p0 = np.array(p0)

  def __call__(self, p, *args, **kwargs):
    parg = self.p0.copy()
    parg[self.vary] = p

    return residuals(parg, *args, **kwargs)
