import numpy as np
import os

### FUNCTIONS ###

def emissivity_model(material):
  param = np.loadtxt(os.path.dirname(__file__)+'/emissivity.txt', dtype = str)
  pmat = np.array([mat.lower() for mat in param[:,0]])
  idx = (pmat == material.lower())

  mtype = param[idx,1][0]
  Tlow = (param[idx,3].astype(float))[0]
  Thigh = (param[idx,4].astype(float))[0]
  p = (param[idx,5:].astype(float))[0]

  # Linear model

  if mtype == 'Poly':

    def model(T):
      e = 0

      for i in range(len(p)):
        e += p[i]*T**i

      return e

    return model, Tlow, Thigh

  raise ValueError('%s is not a valid model type.' %mtype)

# Get emissivity for a given material

def get_emissivity(material, T):

  if 'MLI' in material:
    N =  int(material.split('_')[-1])
    material = '_'.join(material.split('_')[1:-1])

    return get_mli_emissivity('Al_com', 'Black', material, N, T)

  model, Tlow, Thigh = emissivity_model(material)

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

# Get the emissivity for planar multilayer shielding

def get_mli_emissivity(material_front, material_back, material_covered, N, T):
  model1, Tlow1, Thigh1 = emissivity_model(material_front)
  model2, Tlow2, Thigh2 = emissivity_model(material_back)
  model3, Tlow3, Thigh3 = emissivity_model(material_covered)

  Tlow = np.max([Tlow1,Tlow2,Tlow3])
  Thigh = np.min([Thigh1,Thigh2,Thigh3])

  e_f = model1(T)
  e_b = model2(T)
  e_c = model3(T)

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

  return 1/(N * ((1/e_f) + (1/e_b) - 1) + (1/e_c))
