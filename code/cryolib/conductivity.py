import numpy as np
import os
import scipy

from scipy.integrate import quad as integrate

### FUNCTIONS ###

# Generate the conductivity model for a given material

def conductivity_model(material):
  param = np.loadtxt(os.path.dirname(__file__)+'/conductivity.txt', dtype = str)
  pmat = np.array([mat.lower() for mat in param[:,0]])
  idx = (pmat == material.lower())

  mtype = param[idx,1][0]
  Tlow = (param[idx,3].astype(float))[0]
  Thigh = (param[idx,4].astype(float))[0]
  p = (param[idx,5:].astype(float))[0]

  # Polynomial

  if mtype == 'Poly':

    def model(T):
      k = 0

      for i in range(len(p)):
        k += p[i]*T**i

      return k

    return model, Tlow, Thigh

  # NIST formulation

  if mtype == 'NIST':

    def model(T):
      k = 0

      for i in range(9):
        k += p[i]*np.log10(T)**i

      return 10**k

    return model, Tlow, Thigh

  # NIST formulation with power-law extension

  if mtype == 'NIST_power':

    def model(T):
      k = 0

      if type(T) in [list, np.ndarray]:
        T = np.array(T)

        mask1 = (T > p[9])
        mask2 = (T <= p[9])*(T > p[10])
        mask3 = (T <= p[10])

        k *= np.ones_like(T)

        if mask1.sum():
          for i in range(9):
            k[mask1] += p[i]*np.log10(T[mask1])**i

          k[mask1] = 10**k[mask1]

        k[mask2] = p[12]*T[mask2]**p[11]
        k[mask3] = p[15]*T[mask3]**p[14]

        return k

      else:

        if T > p[9]:
          for i in range(9):
            k += p[i]*np.log10(T)**i

          return 10**k

        elif T > p[10]:
          return p[12]*T**p[11]

        else:
          return p[15]*T**p[14]

    return model, Tlow, Thigh

  # NIST formulation for OFHC Copper

  if mtype == 'NIST_OFHC':

    def model(T):
      k = p[0] + p[2]*T**0.5 + p[4]*T + p[6]*T**1.5 + p[8]*T**2
      k /= 1 + p[1]*T**0.5 + p[3]*T + p[5]*T**1.5 + p[7]*T**2

      return k

    return model, Tlow, Thigh

  # Chebyshev polynomials of ln(T) giving ln(k)

  if mtype == 'Cheby_ln':

    def model(T):
      k = 0
      x = ((np.log(T) - p[1]) - (p[2]-np.log(T)))/(p[2] - p[1])

      for i in range(3,3+int(p[0])):
        k += p[i]*np.cos((i-3)*np.arccos(x))

      return np.exp(k)

    return model, Tlow, Thigh

  raise ValueError('%s is not a valid model type.' %mtype)

# Get conductivty for a given material

def get_conductivity(material, T):
  model, Tlow, Thigh = conductivity_model(material)

  if type(T) in [list, np.ndarray]:
    T = np.array(T)

    if (T < Tlow).sum() or (T > Thigh).sum():
      pass
      #print 'WARNING: Temperatures are outside model\'s range of validity: ' \
      #      + '%.3f - %.3f K' %(Tlow,Thigh)

  elif T < Tlow or T > Thigh:
    pass
    #print 'WARNING: Temperature is outside model\'s range of validity: ' \
    #      '%.3f - %.3f K' %(Tlow,Thigh)

  return model(T)

# Get integrated conductivty for a given material

def get_integrated_conductivity(material, T1, T2):
  model, Tlow, Thigh = conductivity_model(material)

  if type(T1) in [list, np.ndarray]:
    T1 = np.array(T1)

    if (T1 < Tlow).sum() or (T1 > Thigh).sum():
      pass
      #print 'WARNING: Low temperatures are outside model\'s range of ' \
      #      + 'validity: %.3f - %.3f K' %(Tlow,Thigh)

    if type(T2) in [list, np.ndarray]:
      T2 = np.array(T2)

      if (T2 < Tlow).sum() or (T2 > Thigh).sum():
        pass
        #print 'WARNING: High temperatures are outside model\'s range of ' \
        #      + 'validity: %.3f - %.3f K' %(Tlow,Thigh)

      K = np.zeros_like(T1)

      for i in range(len(T1)):
        K[i] = integrate(model, T1[i], T2[i],  epsabs = 0.0, epsrel = 1E-7)[0]

      return K

    else:
      K = np.zeros_like(T1)

      for i in range(len(T1)):
        K[i] = integrate(model, T1[i], T2,  epsabs = 0.0, epsrel = 1E-7)[0]

      return K

  elif type(T2) in [list, np.ndarray]:
    T2 = np.array(T2)

    if (T2 < Tlow).sum() or (T2 > Thigh).sum():
      pass
      #print 'WARNING: High temperatures are outside model\'s range of ' \
      #      + 'validity: %.3f - %.3f K' %(Tlow,Thigh)

    K = np.zeros_like(T2)

    for i in range(len(T2)):
      K[i] = integrate(model, T1, T2[i],  epsabs = 0.0, epsrel = 1E-7)[0]

    return K

  else:
    if T1 < Tlow or T1 > Thigh:
      pass
      #print 'WARNING: Low Temperature is outside model\'s range of validity: ' \
      #      + '%.3f - %.3f K' %(Tlow,Thigh)

    if T2 < Tlow or T2 > Thigh:
      pass
      #print 'WARNING: High Temperature is outside model\'s range of ' \
      #      + 'validity: %.3f - %.3f K' %(Tlow,Thigh)

    return integrate(model, T1, T2,  epsabs = 0.0, epsrel = 1E-7)[0]
