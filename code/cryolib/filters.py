import dill as pickle
import matplotlib as mpl
import numexpr as ne
import numpy as np
import os
import scipy

import ConfigParser

from scipy.integrate import quad as integrate
from scipy.interpolate import interp1d as interpolate

import fitting
import conductivity as thermC
import emissivity as thermE

### CONSTANTS ###

c = 2.99792458E8     # m s^-1
kb = 1.38064852E-23  # m^2 kg s^-2 K^-1
h = 6.62607004E-34   # m^2 kg s^-1

c1 = 2*np.pi*h/c**2  # W m^-2 Hz^-1
c2 = h/kb            # K Hz^-1


### FUNCTIONS ###

# Reflect vector on a plane

def reflect(direction, plane):
  out = 2*(np.matmul(plane,direction)*plane)
  out = direction - out.reshape(len(direction),3,1)

  return out

# Rotation matrix that takes +z to the normal of a plane

def rotation_from_zplane(plane):
  if len(plane.shape) == 3:
    nx,ny,nz = plane.T[0]/np.linalg.norm(plane.T, axis=1)
  else:
    nx,ny,nz = plane/np.linalg.norm(plane)

  R = np.tile(np.diag(np.ones(3)),(len(nx),1,1))

  mask = (nz == -1)

  if mask.sum():
    R[mask,[0,1,2],[0,1,2]] *= -1

  mask = np.logical_not(mask)

  v = np.zeros_like(R)
  v[:,0,2] += nx
  v[:,1,2] += ny
  v[:,2,:2] += np.vstack([-nx,-ny]).T

  R[mask] += v[mask]+np.matmul(v,v)[mask]/(1+nz[mask].reshape(mask.sum(),1,1))

  return np.squeeze(R)

# Planck's law (W m^-2 Hz^-1)

c1 = 2*np.pi*h/c**2
c2 = h/kb

def Bv(v,T):
  a = c2*v/T

  if np.any(a > 700):
    if type(a) == np.ndarray:
      a[a > 700] = 700.0
    else:
      a = 700.0

  return (c1*v**3)/(np.exp(a) - 1)

# Planck integral (W m^-2)

def Bvint(param):
  T,flo,fhi = param

  return integrate(Bv, flo, fhi, args = T, epsabs = 0.0, epsrel = 1E-7)[0]

# Power conducted across concentric annuli to a fixed edge

def Pc(idx,T,stack,i):
  E = stack.elements[i]
  r,dr = [E.radius,E.ring_width]
  AL = 2*np.pi*np.append(np.arange(dr,r-1E-14,dr),2*r-dr/2.0)*E.thickness/dr

  return -AL*thermC.get_integrated_conductivity(E.material, abs(T), \
          np.append(abs(T[1:]),E.temperature))

# Power radiated to/from surfaces

def Pr(idx,T,stack):
  F = stack.transfer_factors
  Pr = np.zeros(len(idx))

  flo = np.repeat(stack.frequencies[:-1],len(T))
  fhi = np.repeat(stack.frequencies[1:],len(T))
  Ti = np.tile(abs(T),len(F))
  Iint = np.array(map(Bvint, np.array([Ti,flo,fhi]).T))

  I = stack.radiosity.copy()
  I[:,idx] = Iint.reshape(len(F),len(T))

  for f in range(len(F)):
    Pr += -(F[f,idx].T*(I[f,idx]-np.tile(I[f,:],(len(idx),1)).T)).sum(0)

  return Pr

# Total power entering concentric annuli

def Ptot(idx,T,stack,i,comm = None):
  E = stack.elements[i]
  F = stack.transfer_factors
  I = stack.radiosity.copy()

  if comm is None:
    rank = 0
    nproc = 1
  else:
    rank = comm.rank
    nproc = comm.size

  Pr = np.zeros(len(idx))

  flo = np.repeat(stack.frequencies[:-1],len(T))
  fhi = np.repeat(stack.frequencies[1:],len(T))
  Ti = np.tile(abs(T),len(F))
  Iint = np.zeros(len(Ti))

  for j in range(rank,len(Iint),nproc):
    Iint[j] = Bvint(np.array([Ti[j],flo[j],fhi[j]]).T)

  if comm is not None:
    comm.Barrier()
    Iint = comm.allreduce(Iint)

  I[:,idx] = Iint.reshape(len(F),len(T))

  for f in range(len(F)):
    Pr += (F[f,idx].T*(I[f,idx]-np.tile(I[f,:],(len(idx),1)).T)).sum(0)

  r,dr = [E.radius,E.ring_width]
  AL = 2*np.pi*np.append(np.arange(dr,r-1E-14,dr),2*r-dr/2.0)*E.thickness/dr

  Pc = np.diff(np.append(0,AL*thermC.get_integrated_conductivity(E.material, \
               abs(T),np.append(abs(T[1:]),E.temperature))))

  Pdiff = Pc-Pr

  #if not rank:
  #  print abs(T)
  #  print Pdiff/Pr
  #  print

  return Pdiff


### CLASSES ###

# An optical stack composed of multiple geometric objects

class OpticalStack(object):

  id = None
  size = None

  elements = None
  types = None
  names = None

  stages = None
  stagenum = None

  nsurfaces = None
  temperatures = None
  frequencies = None

  transfer_factors = None
  radiosity = None
  power = None

  nrays = None
  maxiterF = None
  maxiterT = None

  # Initialization

  def __init__(self, id, stages):
    self.id = id
    self.size = 0
    self.elements = np.array([])
    self.types = np.array([])
    self.names = np.array([])
    self.stages = stages
    self.stagenum = np.array([], dtype = int)
    self.nsurfaces = np.array([], dtype = int)
    self.temperatures = np.array([])

  # Add an element

  def add_element(self, element, type, name, stagenum):
    self.size += 1
    self.elements = np.append(self.elements,element)
    self.types = np.append(self.types,type)
    self.names = np.append(self.names,name)
    self.stagenum = np.append(self.stagenum,stagenum)

    if type == 'filter':
      self.nsurfaces = np.append(self.nsurfaces,element.nrings)
      self.temperatures = np.append(self.temperatures, \
                                    np.ones(element.nrings)*element.temperature)
    else:
      self.nsurfaces = np.append(self.nsurfaces,1)
      self.temperatures = np.append(self.temperatures,element.temperature)

  # Initialize from config file

  @classmethod
  def from_config(cls,fp):
    model = ConfigParser.ConfigParser()
    model.read(fp)

    spectra = {}
    sections = model.sections()
    idx = np.array([]).astype(float)

    for section in sections:

      if section.lower() == 'model':
        modelID = model.get(section,'identifier').split('#')[0].strip()
        Nrings = int(model.get(section,'numrings').split('#')[0])
        Nrays = int(model.get(section,'numrays').split('#')[0])
        Niter = int(model.get(section,'maxiterations').split('#')[0])

        freq = model.get(section,'frequencies').split('#')[0].strip()[1:-1]
        freq = np.array(freq.split(',')).astype(float)
        freq = 10**np.arange(freq[0],freq[1]+1E-14,freq[2])
        freq = np.append(0.0,np.append(freq,1E16))
 
      elif section.lower() == 'stages':
        options = model.options(section)
        stages = np.zeros(len(options))

        for opt in options:
          stages[int(opt[5:])] = float(model.get(section,opt).split('#')[0])

      elif section.lower() == 'spectra':
        spectrafp = model.get(section,'directory').split('#')[0].strip()+'/'
        options = model.options(section)

        for opt in options:
          if opt != 'directory':
            spectrafn = model.get(section,opt).split('#')[0].strip()
            spectra[opt.upper()] = np.loadtxt(spectrafp+spectrafn)
            spectra[opt.upper()][:,0] *= c*100

    self = cls(modelID,stages)
    self.nrays = Nrays
    self.maxiterF = Niter
    self.maxiterT = Niter
    self.frequencies = freq

    for section in sections:

      if section.lower() in ['model','stages','spectra']:
        continue

      name = model.get(section,'name').split('#')[0].strip()
      snum = int(model.get(section,'stage').split('#')[0])
      idx = np.append(idx,int(model.get(section,'index').split('#')[0]))

      M = model.get(section,'material').split('#')[0].strip()

      if 'temperature' in model.options(section):
        T = float(model.get(section,'temperature').split('#')[0])
      else:
        T = stages[snum]

      S = model.get(section,'shape').split('#')[0].strip()

      O = model.get(section,'origin').split('#')[0].strip()[1:-1].split(',')
      O = np.array(O).astype(float)/100

      N = model.get(section,'direction').split('#')[0].strip()[1:-1].split(',')
      N = np.array(N).astype(float)

      R = float(model.get(section,'radius').split('#')[0])/100
      H = float(model.get(section,'thickness').split('#')[0])/1000

      if 'filter' in section.lower():
        assert S.lower() == 'layereddisk'

        spec = model.get(section,'spectrum').split('#')[0].strip().upper()

        LD = LayeredDisk(O,N,R,Nrings,H,M,T,spectra[spec])
        CL = Cylinder(O,N,R,H,0.0,'Black',T)
        self.add_element(LD, 'filter', name, snum)
        self.add_element(CL, 'clamp', name.replace('lter','lter Clamp').replace('ens','ens Clamp'), snum)

        idx = np.append(idx,idx[-1]+0.5)

      elif 'window' in section.lower():
        assert S.lower() == 'disk'

        DK = Disk(O,N,R,H,M,T)
        CL = Cylinder(O,N,R,H,0.0,'Black',T)
        self.add_element(DK, section.split()[0].lower(), name, snum)
        self.add_element(CL, 'clamp', name.replace('ndow','ndow Clamp'), snum)

        idx = np.append(idx,idx[-1]+0.5)

      elif S.lower() == 'disk':
        DK = Disk(O,N,R,H,M,T)
        self.add_element(DK, section.split()[0].lower(), name, snum)

      elif S.lower() == 'cylinder':
        L = float(model.get(section,'length').split('#')[0])/100

        CL = Cylinder(O,N,R,L,H,M,T)
        self.add_element(CL, section.split()[0].lower(), name, snum)

      elif S.lower() == 'ring':
        W = float(model.get(section,'width').split('#')[0])/100

        RG = Ring(O,N,R,W,H,M,T)
        self.add_element(RG, section.split()[0].lower(), name, snum)

      elif S.lower() == 'hemisphere':
        HS = HemiSphere(O,N,R,H,M,T)
        self.add_element(HS, section.split()[0].lower(), name, snum)

    idx = idx.argsort()
    self.elements = self.elements[idx]
    self.types = self.types[idx]
    self.names = self.names[idx]
    self.stagenum = self.stagenum[idx]

    temperatures = np.zeros_like(self.temperatures)
    nsurfaces = self.nsurfaces[idx]

    for i in range(self.size):
      idx1 = nsurfaces[:i].sum()
      idx2 = idx1+nsurfaces[i]
      idx3 = self.nsurfaces[:idx[i]].sum()
      idx4 = idx3+self.nsurfaces[idx[i]]
      temperatures[idx1:idx2] = self.temperatures[idx3:idx4]

    self.nsurfaces = nsurfaces
    self.temperatures = temperatures

    return self

  # Compute radiational transfer factors between elements

  def compute_transfer_factors(self, comm = None, niter = None):

    if comm is None:
      rank = 0
      nproc = 1
    else:
      rank = comm.rank
      nproc = comm.size
      comm.Barrier()

    emask = (self.types == 'filter')
    emask += (self.types == 'wall')
    emask += (self.types == 'window')
    emask += (self.types == 'clamp')
    emask += (self.types == 'exit')
    emask += (self.types == 'entrance')

    if niter is None:
      niter = self.maxiterF
    else:
      self.maxiterF = niter

    freq = self.frequencies

    N = np.arange(emask.size)[emask].astype(int)
    ABe = np.zeros((len(freq)-1, self.nsurfaces[emask].sum(), \
                    self.nsurfaces.sum()))

    if comm is None:
      rank = 0
      nproc = 1
    else:
      rank = comm.rank
      nproc = comm.size

    for f in range(len(freq)-1):
      R = np.array([])
      T = np.array([])
      A = np.array([])

      for i in range(self.size):

        if callable(self.elements[i].reflectivity):
          r = integrate(self.elements[i].reflectivity, freq[f], freq[f+1], \
                        epsabs = 0.0, epsrel = 1E-7, limit = 100000)[0] \
                        / (freq[f+1]-freq[f])
        else:
          r = self.elements[i].reflectivity

        if callable(self.elements[i].transmissivity):
          t = integrate(self.elements[i].transmissivity, freq[f], freq[f+1], \
                        epsabs = 0.0, epsrel = 1E-7, limit = 100000)[0] \
                        / (freq[f+1]-freq[f])
        else:
          t = self.elements[i].transmissivity

        if callable(self.elements[i].transmissivity):
          a = integrate(self.elements[i].absorptivity, freq[f], freq[f+1], \
                        epsabs = 0.0, epsrel = 1E-7, limit = 100000)[0] \
                        / (freq[f+1]-freq[f])
        else:
          a = self.elements[i].absorptivity

        R = np.append(R,np.repeat(r,self.nsurfaces[i]))
        T = np.append(T,np.repeat(t,self.nsurfaces[i]))
        A = np.append(A,np.repeat(a,self.nsurfaces[i]))

      ii = np.repeat(np.arange(emask.sum()),self.nsurfaces[emask])
      jj = np.hstack([range(self.nsurfaces[emask][j]) for j in \
                      range(emask.sum())])
      idx = zip(ii,jj)

      if not rank:
        print

      for ij in range(rank,len(idx),nproc):
        i,j = idx[ij]
        element = self.elements[emask][i]

        if self.types[emask][i] == 'filter':
          rays = element.generate_rays(self.nrays, j, 'both')
          area = 2*element.area[j]
        elif self.types[emask][i] == 'window':
          rays = element.generate_rays(self.nrays,'both')
          area = 2*element.area
        else:
          rays = element.generate_rays(self.nrays)
          area = element.area

        for k in range(niter):
          nearest = i*np.ones(rays.n, dtype = int)
          D = np.inf*np.ones(rays.n)

          for m in range(self.size):
            d = self.elements[m].intersect(rays)[0].flatten()
            d[d == 0] = np.inf
            mask = np.vstack([D,d]).argmin(0)
            D = np.vstack([D,d])[mask,np.arange(rays.n)]
            nearest = np.vstack([nearest, \
                                 m*np.ones(rays.n)])[mask,np.arange(rays.n)]

          for m in np.unique(nearest).astype(int):
            ii = self.nsurfaces[:m].sum()
            mask_in = (nearest == m)
            amask = self.elements[m].interact(rays,R[ii],T[ii],A[ii],mask_in)
            nearest = nearest[np.logical_not(amask)]

          if rays.n == 0:
            break

        numit = k
        idx1 = self.nsurfaces[emask][:i].sum()+j

        for k in range(self.size):

          if self.types[k] == 'filter':
            for l in range(self.elements[k].nrings):
              idx2 = self.nsurfaces[:k].sum()+l
              ABe[f,idx1,idx2] = area*self.elements[k].nabs[l]/self.nrays

              #print (freq[f]+freq[f+1])/2/1E9, idx1, idx2, \
              #      '%d:%d -> %d:%d =' %(N[i],j,k,l), \
              #      area*self.elements[k].nabs[l]/self.nrays

          else:
            idx2 = self.nsurfaces[:k].sum()
            ABe[f,idx1,idx2] = area*self.elements[k].nabs/self.nrays

            #print (freq[f]+freq[f+1])/2/1E9, idx1, idx2, \
            #      '%d:%d -> %d:0 =' %(N[i],j,k), \
            #      area*self.elements[k].nabs/self.nrays

          self.elements[k].nabs *= 0

        print '%24s (%2d/%2d): Band = %7.1e-%7.1e Hz, Iterations = %4d ' \
              %(self.names[i], j+1, self.nsurfaces[emask][i], freq[f], \
                freq[f+1], numit)

      if comm is not None:
        comm.Barrier()
        ABe[f] = comm.allreduce(ABe[f])

      for i in range(emask.sum()):
        element = self.elements[emask][i]

        for j in range(self.nsurfaces[emask][i]):
          idx = self.nsurfaces[emask][:i].sum()+j
          ABe[f,idx,:] *= A[idx]

    self.transfer_factors = ABe

    if comm is not None:
      comm.Barrier()

  # Get the total radiosity of each element

  def get_radiosity(self, T = None):

    if T is None:
      Iint = np.zeros((self.frequencies.size-1,self.nsurfaces.sum()))

      for f in range(self.frequencies.size-1):
        for i in range(self.temperatures.size):
          Iint[f,i] = integrate(Bv,self.frequencies[f],self.frequencies[f+1], \
                                args = self.temperatures[i], epsabs = 0.0, \
                                epsrel = 1E-7)[0]

      self.radiosity = Iint

    else:
      Iint = np.zeros((self.frequencies.size-1,len(T)))

      for f in range(self.frequencies.size-1):
        for i in range(len(T)):
          Iint[f,i] = integrate(Bv,self.frequencies[f],self.frequencies[f+1], \
                                args = T[i], epsabs = 0.0, epsrel = 1E-7)[0]

    return Iint

  # Find the stack's thermal equilibrium

  def thermalize(self, comm = None, niter = None):

    if comm is None:
      rank = 0
      nproc = 1
    else:
      rank = comm.rank
      nproc = comm.size
      comm.Barrier()

    F = self.transfer_factors.copy()
    mask = np.hstack([np.repeat(self.types[i] != 'clamp', self.nsurfaces[i]) \
                      for i in range(self.size)])

    if niter is None:
      niter = self.maxiterT
    else:
      self.maxiterT = niter

    for f in range(len(F)):
      Favg = (F[f]+F[f].T)/2
      Fclamp1 = F[f].copy()
      Fclamp2 = F[f].T.copy()

      for i in range(len(F[f])):
        if mask[i]:
          self.transfer_factors[f,i,:] = (Favg[:,i]+Favg[i,:])/2
          self.transfer_factors[f,:,i] = (Favg[:,i]+Favg[i,:])/2
        else:
          self.transfer_factors[f,:,i] = Fclamp1[:,i]
          self.transfer_factors[f,i,:] = Fclamp2[i,:]

    for i in range(niter):
      T = self.temperatures
      Tpr = self.temperatures.copy()

      if not rank:
        print

      for j in range(self.size):
        self.get_radiosity()
        idx = range(self.nsurfaces[:j].sum(),self.nsurfaces[:j+1].sum())

        if self.types[j] == 'filter':
          Tfit = T[idx].copy()
          TT = T[idx].copy()

          mkwargs = {'stack':self,'i':j,'comm': comm}

          Prel = np.inf*TT

          while np.any(abs(Prel) > 1E-3):
            Tfit = abs(fitting.leastsq(idx, np.zeros(len(idx)), Ptot, TT, \
                                       mkwargs = mkwargs, maxfev=10000, \
                                       ftol = 1E-15, xtol = 1E-15))

            Prel = -Ptot(idx,Tfit,self,j,comm)/Pr(idx,Tfit,self)

            if not rank:
              if self.elements[j].temperature < 0.5:
                TT = TT*0+0.2+0.8*np.random.rand()
              else:
                TT = 1+150*np.random.rand(len(TT))

            if comm is not None:
              comm.Barrier()
              TT = comm.bcast(TT)

          T[idx] = Tfit.copy()
          self.get_radiosity()

          if not rank:
            print '%18s: Iteration = %2d, ' %(self.names[j],i+1) \
                + 'Temperature = %7.3f K, ' %(T[idx][0]) \
                + 'Load = %9.2e W' %(Pr(idx,T[idx],self).sum())

      if all(abs(T/Tpr-1) <= 2E-8):
        break

    self.get_radiosity()
    self.power = np.zeros(self.size)

    for i in range(self.size):
      idx = range(self.nsurfaces[:i].sum(),self.nsurfaces[:i+1].sum())
      self.power[i] = Pr(idx,self.temperatures[idx],self).sum()

    self.transfer_factors = F

  # Plot the stack

  def plot(self, ax, color = 'default'):

    if color == 'default':

      for i in range(self.size):
        if self.types[i] in ['wall','clamp']:
          self.elements[i].plot(ax)
        elif self.types[i] == 'window':
          self.elements[i].plot(ax,'b')
        elif 'ir' in self.names[i].lower():
          self.elements[i].plot(ax,'r')
        elif 'lp' in self.names[i].lower():
          self.elements[i].plot(ax,'g')

    elif color == 'thermal':
      norm = mpl.colors.Normalize(0,300)
      s = mpl.cm.ScalarMappable(norm,'spectral')

      for i in range(self.size)[::-1]:
        if self.types[i] != 'filter':
          continue

        idx = range(self.nsurfaces[:i].sum(),self.nsurfaces[:i+1].sum())
        c = s.to_rgba(self.temperatures[idx])
        self.elements[i].plot(ax,c,0.8)

      cfig = mpl.pyplot.figure()
      cax = cfig.add_subplot(111)
      cbm = cax.imshow([np.append(0,self.temperatures)], cmap = 'spectral')
      mpl.pyplot.close(cfig)

      return cbm

  # Pickle the stack

  def dump(self, outfp):
    out = open(outfp,'w')
    pickle.dump(self,out,2)
    out.close()

# A bundle of rays

class RayBundle(object):

  # Parameters

  n = None
  origin = None
  direction = None

  # Initialization

  def __init__(self, origin, direction):
    self.n = len(origin)
    self.origin = origin
    self.direction = direction

  # Remove rays using a mask

  def reduce(self, mask):
    self.n = mask.sum()
    self.origin = self.origin[mask]
    self.direction = self.direction[mask]

  def plot(self, ax, length, color = 'b'):
    t = np.arange(2)*length

    XX = self.origin[:,0]+self.direction[:,0]*t
    YY = self.origin[:,1]+self.direction[:,1]*t
    ZZ = self.origin[:,2]+self.direction[:,2]*t

    for i in range(0,len(XX)):
      ax.plot(XX[i,:],YY[i,:], ZZ[i,:], color, lw = 0.2)

# A standard disk

class Disk(object):

  # Geometric Parameters

  origin = None
  plane = None
  radius = None
  area = None
  thickness = None

  # Material properties

  material = None
  temperature = None
  reflectivity = None
  transmissivity = None
  absorptivity = None

  # Interaction

  nabs = None

  # Initialization

  def __init__(self, origin, plane, radius, thickness, material, temperature, \
               spectra = None):
    self.origin = np.array(origin).reshape(3,1)
    self.plane = (np.array(plane)/np.linalg.norm(plane)).reshape(3,1)
    self.radius = radius
    self.area = np.pi*radius**2

    self.material = material
    self.temperature = temperature
    self.thickness = thickness

    if spectra is None:

      if material.lower() == 'black':
        self.reflectivity = 0.0
        self.transmissivity = 0.0
        self.absorptivity = 1.0
      elif material.lower() == 'transparent':
        self.reflectivity = 0.0
        self.transmissivity = 1.0
        self.absorptivity = 0.0
      else:
        self.reflectivity = 1-thermE.get_emissivity(material,temperature)
        self.transmissivity = 0.0
        self.absorptivity = thermE.get_emissivity(material,temperature)

    else:
      f,T,A = spectra.T

      self.transmissivity = interpolate(f,T,'linear', bounds_error = False, \
                                        fill_value = (T[0],T[-1]))
      self.absorptivity = interpolate(f,A,'linear', bounds_error = False, \
                                      fill_value = (A[0],A[-1]))
      self.reflectivity = lambda freq: 1 - self.absorptivity(freq) - \
                                       self.transmissivity(freq)

    self.nabs = 0.0

  # Generate ray bundle from surface

  def generate_rays(self, n, side = 'both'):

    # Sample the surface

    rho = np.random.rand(n)*(self.radius**2)
    alpha = np.random.rand(n)*2*np.pi

    X = np.zeros((n,3,1))
    X[:,0] = (np.sqrt(rho)*np.cos(alpha)).reshape(n,1)
    X[:,1] = (np.sqrt(rho)*np.sin(alpha)).reshape(n,1)
    X[:,2] = np.zeros_like(rho).reshape(n,1)

    R = rotation_from_zplane(self.plane)
    X = np.matmul(R,X) + self.origin.reshape(3,1)

    # Generate random directions

    u1 = np.random.rand(n)
    u2 = np.random.rand(n)
    theta = 0.5*np.arccos(1-2*u1)
    phi = 2*np.pi*u2

    N = np.zeros_like(X)
    N[:,0] = (np.sin(theta)*np.cos(phi)).reshape(n,1)
    N[:,1] = (np.sin(theta)*np.sin(phi)).reshape(n,1)
    N[:,2] = (np.cos(theta)).reshape(n,1)

    N = np.matmul(R,N)

    # Front (back) is in the +N (-N) direction

    if side == 'front':
      rays = RayBundle(X+self.thickness*self.plane,N)
    elif side == 'back':
      rays = RayBundle(X,-N)
    elif side == 'both':
      X = np.vstack([X[:len(X)/2],X[len(X)/2:]+self.thickness*self.plane])
      N = np.vstack([-N[:len(X)/2],N[len(X)/2:]])
      rays = RayBundle(X,N)

    return rays

  # Intersection

  def intersect(self, rays):
    N = self.plane.reshape(3,)
    D = np.inf*np.ones((rays.n,1))
    norm = np.matmul(N,rays.direction)

    for side in ['front','back']:

      if side == 'front':
        O = (self.origin+self.thickness*self.plane).reshape(3,)
        d = np.matmul(N,self.origin+self.thickness*self.plane-rays.origin)/norm
      elif side == 'back':
        O = self.origin.reshape(3,)
        d = np.matmul(N,self.origin-rays.origin)/norm

      d[abs(d) < 1E-14] = 0

      X = rays.origin + rays.direction*d.reshape(rays.n,1,1)
      x,y,z = X[:,:,0].T
      Ox,Oy,Oz = O

      r = ne.evaluate('sqrt((x-Ox)**2+(y-Oy)**2+(z-Oz)**2)').reshape(rays.n,1)
      mask = ((r > self.radius)+(d < 0)).flatten()

      d[mask] = np.inf
      D = np.hstack([D,d]).min(1).reshape(rays.n,1)

    mask = (D != np.inf).flatten()
    X[mask] = rays.origin[mask] \
              + rays.direction[mask]*D[mask].reshape(mask.sum(),1,1)

    return D,X

  # Interaction

  def interact(self, rays, R, T, A, mask_in = None):

    # Check intersection

    mask,X = self.intersect(rays)
    mask = (mask != np.inf).flatten()

    if mask_in is not None:
      mask *= mask_in

    # Deterimine absorption, reflection, transmission

    u = np.random.rand(mask.sum())
    amask = np.zeros_like(mask)
    amask[mask] = (A >= u)
    rmask = np.zeros_like(mask)
    rmask[mask] = np.logical_not(amask[mask])
    rmask[mask] *= ((R+A) >= u)
    tmask = np.zeros_like(mask)
    tmask[mask] = np.logical_not(rmask[mask])*np.logical_not(amask[mask])

    # Transmit rays

    if tmask.sum() != 0:
      rays.origin[tmask] = X[tmask]

    # Reflect rays

    if rmask.sum() != 0:
      rays.origin[rmask] = X[rmask]
      rays.direction[rmask] = reflect(rays.direction[rmask], \
                                      self.plane.reshape(3,))

    # Absorb rays

    if amask.sum() != 0:
      self.nabs += amask.sum()
      rays.reduce(np.logical_not(amask))

    return amask

  def plot(self, ax, color = 'k', alpha = 0.2):
    phi = np.linspace(0,2*np.pi,100)
    z = np.linspace(0,self.thickness,100)

    phi,z = np.meshgrid(phi,z)
    mshape = z.shape
    phi = phi.flatten()
    z = z.flatten()
    x = self.radius*np.cos(phi)
    y = self.radius*np.sin(phi)

    R = rotation_from_zplane(self.plane)
    X = np.vstack([x,y,z])
    X = np.matmul(R,X)

    X += self.origin
    x,y,z = X
    x = x.reshape(mshape)
    y = y.reshape(mshape)
    z = z.reshape(mshape)

    ax.plot_surface(x, y, z, color = color, lw = 0, alpha = alpha)

    r = np.linspace(0,self.radius,100)
    phi = np.linspace(0,2*np.pi,100)

    r,phi = np.meshgrid(r,phi)
    mshape = r.shape
    r = r.flatten()
    phi = phi.flatten()
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    z1 = np.zeros_like(x)
    z2 = z1+self.thickness

    R = rotation_from_zplane(self.plane)
    X1 = np.vstack([x,y,z1])
    X2 = np.vstack([x,y,z2])
    X1 = np.matmul(R,X1)
    X2 = np.matmul(R,X2)

    X1 += self.origin
    x,y,z = X1
    x = x.reshape(mshape)
    y = y.reshape(mshape)
    z = z.reshape(mshape)

    ax.plot_surface(x, y, z, color = color, lw = 0, alpha = alpha)

    X2 += self.origin
    x,y,z = X2
    x = x.reshape(mshape)
    y = y.reshape(mshape)
    z = z.reshape(mshape)

    ax.plot_surface(x, y, z, color = color, lw = 0, alpha = alpha)

# A standard ring

class Ring(object):

  # Geometric Parameters

  origin = None
  plane = None
  radius = None
  ring_width = None
  area = None
  thickness = None

  # Material properties

  material = None
  temperature = None
  reflectivity = None
  transmissivity = None
  absorptivity = None

  # Interaction

  nabs = None

  # Initialization

  def __init__(self, origin, plane, radius, ring_width, thickness, material, \
               temperature, spectra = None):
    self.origin = np.array(origin).reshape(3,1)
    self.plane = (np.array(plane)/np.linalg.norm(plane)).reshape(3,1)
    self.radius = radius
    self.ring_width = ring_width
    self.area = np.pi*((radius+ring_width)**2 - radius**2)

    self.material = material
    self.temperature = temperature
    self.thickness = thickness

    if spectra is None:

      if material.lower() == 'black':
        self.reflectivity = 0.0
        self.transmissivity = 0.0
        self.absorptivity = 1.0
      elif material.lower() == 'transparent':
        self.reflectivity = 0.0
        self.transmissivity = 1.0
        self.absorptivity = 0.0
      else:
        self.reflectivity = 1-thermE.get_emissivity(material,temperature)
        self.transmissivity = 0.0
        self.absorptivity = thermE.get_emissivity(material,temperature)

    else:
      f,T,A = spectra.T

      self.transmissivity = interpolate(f,T,'linear', bounds_error = False, \
                                        fill_value = (T[0],T[-1]))
      self.absorptivity = interpolate(f,A,'linear', bounds_error = False, \
                                      fill_value = (A[0],A[-1]))
      self.reflectivity = lambda freq: 1 - self.absorptivity(freq) - \
                                       self.transmissivity(freq)

    self.nabs = 0.0

  # Generate ray bundle from surface

  def generate_rays(self, n, side = 'front'):

    # Sample the surface

    r1 = self.radius
    r2 = r1 + self.ring_width

    rho = r1**2 + np.random.rand(n)*(r2**2-r1**2)
    alpha = np.random.rand(n)*2*np.pi

    X = np.zeros((n,3,1))
    X[:,0] = (np.sqrt(rho)*np.cos(alpha)).reshape(n,1)
    X[:,1] = (np.sqrt(rho)*np.sin(alpha)).reshape(n,1)
    X[:,2] = np.zeros_like(rho).reshape(n,1)

    R = rotation_from_zplane(self.plane)
    X = np.matmul(R,X) + self.origin.reshape(3,1)

    # Generate random directions

    u1 = np.random.rand(n)
    u2 = np.random.rand(n)
    theta = 0.5*np.arccos(1-2*u1)
    phi = 2*np.pi*u2

    N = np.zeros_like(X)
    N[:,0] = (np.sin(theta)*np.cos(phi)).reshape(n,1)
    N[:,1] = (np.sin(theta)*np.sin(phi)).reshape(n,1)
    N[:,2] = (np.cos(theta)).reshape(n,1)

    N = np.matmul(R,N)

    # Front (back) is in the +N (-N) direction

    if side == 'front':
      rays = RayBundle(X+self.thickness*self.plane,N)
    elif side == 'back':
      rays = RayBundle(X,-N)
    elif side == 'both':
      X = np.vstack([X[:len(X)/2],X[len(X)/2:]+self.thickness*self.plane])
      N = np.vstack([-N[:len(X)/2],N[len(X)/2:]])
      rays = RayBundle(X,N)

    return rays

  # Intersection

  def intersect(self, rays):
    N = self.plane.reshape(3,)
    D = np.inf*np.ones((rays.n,1))
    norm = np.matmul(N,rays.direction)

    for side in ['front','back']:

      if side == 'front':
        O = (self.origin+self.thickness*self.plane).reshape(3,)
        d = np.matmul(N,self.origin+self.thickness*self.plane-rays.origin)/norm
      elif side == 'back':
        O = self.origin.reshape(3,)
        d = np.matmul(N,self.origin-rays.origin)/norm

      d[abs(d) < 1E-14] = 0

      X = rays.origin + rays.direction*d.reshape(rays.n,1,1)
      x,y,z = X[:,:,0].T
      Ox,Oy,Oz = O

      r = ne.evaluate('sqrt((x-Ox)**2+(y-Oy)**2+(z-Oz)**2)').reshape(rays.n,1)
      mask = ((r < self.radius) + (r > self.radius+self.ring_width) + \
              (d < 0)).flatten()

      d[mask] = np.inf
      D = np.hstack([D,d]).min(1).reshape(rays.n,1)

    mask = (D != np.inf).flatten()
    X[mask] = rays.origin[mask] \
              + rays.direction[mask]*D[mask].reshape(mask.sum(),1,1)

    return D,X

  # Interaction

  def interact(self, rays, R, T, A, mask_in = None):

    # Check intersection

    mask,X = self.intersect(rays)
    mask = (mask != np.inf).flatten()

    if mask_in is not None:
      mask *= mask_in

    # Deterimine absorption, reflection, transmission

    u = np.random.rand(mask.sum())
    amask = np.zeros_like(mask)
    amask[mask] = (A >= u)
    rmask = np.zeros_like(mask)
    rmask[mask] = np.logical_not(amask[mask])
    rmask[mask] *= ((R+A) >= u)
    tmask = np.zeros_like(mask)
    tmask[mask] = np.logical_not(rmask[mask])*np.logical_not(amask[mask])

    # Transmit rays

    if tmask.sum() != 0:
      rays.origin[tmask] = X[tmask]

    # Reflect rays

    if rmask.sum() != 0:
      rays.origin[rmask] = X[rmask]
      rays.direction[rmask] = reflect(rays.direction[rmask], \
                                      self.plane.reshape(3,))

    # Absorb rays

    if amask.sum() != 0:
      self.nabs += amask.sum()
      rays.reduce(np.logical_not(amask))

    return amask

  def plot(self, ax, color = 'k', alpha = 0.2):
    phi = np.linspace(0,2*np.pi,100)
    z = np.linspace(0,self.thickness,100)

    phi,z = np.meshgrid(phi,z)
    mshape = z.shape
    phi = phi.flatten()
    z = z.flatten()
    x1 = self.radius*np.cos(phi)
    x2 = (self.radius+self.ring_width)*np.cos(phi)
    y1 = self.radius*np.sin(phi)
    y2 = (self.radius+self.ring_width)*np.sin(phi)

    R = rotation_from_zplane(self.plane)
    X1 = np.vstack([x1,y1,z])
    X2 = np.vstack([x2,y2,z])
    X1 = np.matmul(R,X1)
    X2 = np.matmul(R,X2)

    X1 += self.origin
    x,y,z = X1
    x = x.reshape(mshape)
    y = y.reshape(mshape)
    z = z.reshape(mshape)

    ax.plot_surface(x, y, z, color = color, lw = 0, alpha = alpha)

    X2 += self.origin
    x,y,z = X2
    x = x.reshape(mshape)
    y = y.reshape(mshape)
    z = z.reshape(mshape)

    ax.plot_surface(x, y, z, color = color, lw = 0, alpha = alpha)

    r = np.linspace(self.radius,self.radius+self.ring_width,100)
    phi = np.linspace(0,2*np.pi,100)

    r,phi = np.meshgrid(r,phi)
    mshape = r.shape
    r = r.flatten()
    phi = phi.flatten()
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    z1 = np.zeros_like(x)
    z2 = z1+self.thickness

    R = rotation_from_zplane(self.plane)
    X1 = np.vstack([x,y,z1])
    X2 = np.vstack([x,y,z2])
    X1 = np.matmul(R,X1)
    X2 = np.matmul(R,X2)

    X1 += self.origin
    x,y,z = X1
    x = x.reshape(mshape)
    y = y.reshape(mshape)
    z = z.reshape(mshape)

    ax.plot_surface(x, y, z, color = color, lw = 0, alpha = alpha)

    X2 += self.origin
    x,y,z = X2
    x = x.reshape(mshape)
    y = y.reshape(mshape)
    z = z.reshape(mshape)

    ax.plot_surface(x, y, z, color = color, lw = 0, alpha = alpha)

# A multi-layered disk composed of concentric rings

class LayeredDisk(object):

  # Geometric Parameters

  origin = None
  plane = None
  radius = None
  area = None
  thickness = None

  # Layering parameters

  nrings = None
  ring_width = None

  # Material properties

  material = None
  temperature = None
  reflectivity = None
  transmissivity = None
  absorptivity = None

  # Interaction

  nabs = None

  # Initialization

  def __init__(self, origin, plane, radius, nrings, thickness, material, \
               temperature, spectra = None):
    self.origin = np.array(origin).reshape(3,1)
    self.plane = (np.array(plane)/np.linalg.norm(plane)).reshape(3,1)
    self.radius = radius
    self.nrings = nrings
    self.ring_width = radius/nrings
    self.area = np.pi*(((np.arange(self.nrings)+1)*self.ring_width)**2 - \
                      (np.arange(self.nrings)*self.ring_width)**2)

    self.material = material
    self.temperature = temperature
    self.thickness = thickness

    if spectra is None:

      if material.lower() == 'black':
        self.reflectivity = 0.0
        self.transmissivity = 0.0
        self.absorptivity = 1.0
      elif material.lower() == 'transparent':
        self.reflectivity = 0.0
        self.transmissivity = 1.0
        self.absorptivity = 0.0
      else:
        self.reflectivity = 1-thermE.get_emissivity(material,temperature)
        self.transmissivity = 0.0
        self.absorptivity = thermE.get_emissivity(material,temperature)

    else:
      f,T,A = spectra.T

      self.transmissivity = interpolate(f,T,'linear', bounds_error = False, \
                                        fill_value = (T[0],T[-1]))
      self.absorptivity = interpolate(f,A,'linear', bounds_error = False, \
                                      fill_value = (A[0],A[-1]))
      self.reflectivity = lambda freq: 1 - self.absorptivity(freq) - \
                                       self.transmissivity(freq)

    self.nabs = np.zeros(nrings, dtype = float)

  # Generate ray bundle from surface

  def generate_rays(self, n, ring_num, side = 'both'):

    # Define inner and outer radii (ring_num is zero-indexed)

    r1 = ring_num*self.ring_width
    r2 = r1+self.ring_width

    # Sample the surface

    rho = r1**2 + np.random.rand(n)*(r2**2-r1**2)
    alpha = np.random.rand(n)*2*np.pi

    X = np.zeros((n,3,1))
    X[:,0] = (np.sqrt(rho)*np.cos(alpha)).reshape(n,1)
    X[:,1] = (np.sqrt(rho)*np.sin(alpha)).reshape(n,1)
    X[:,2] = np.zeros_like(rho).reshape(n,1)

    R = rotation_from_zplane(self.plane)
    X = np.matmul(R,X) + self.origin.reshape(3,1)

    # Generate random directions

    u1 = np.random.rand(n)
    u2 = np.random.rand(n)
    theta = 0.5*np.arccos(1-2*u1)
    phi = 2*np.pi*u2

    N = np.zeros_like(X)
    N[:,0] = (np.sin(theta)*np.cos(phi)).reshape(n,1)
    N[:,1] = (np.sin(theta)*np.sin(phi)).reshape(n,1)
    N[:,2] = (np.cos(theta)).reshape(n,1)

    N = np.matmul(R,N)

    # Front (back) is in the +N (-N) direction

    if side == 'front':
      rays = RayBundle(X+self.thickness*self.plane,N)
    elif side == 'back':
      rays = RayBundle(X,-N)
    elif side == 'both':
      X = np.vstack([X[:len(X)/2],X[len(X)/2:]+self.thickness*self.plane])
      N = np.vstack([-N[:len(X)/2],N[len(X)/2:]])
      rays = RayBundle(X,N)

    return rays

  # Intersection

  def intersect(self, rays):
    N = self.plane.reshape(3,)
    D = np.inf*np.ones((rays.n,1))
    R = D.copy()
    norm = np.matmul(N,rays.direction)

    for side in ['front','back']:

      if side == 'front':
        O = (self.origin+self.thickness*self.plane).reshape(3,)
        d = np.matmul(N,self.origin+self.thickness*self.plane-rays.origin)/norm
      elif side == 'back':
        O = self.origin.reshape(3,)
        d = np.matmul(N,self.origin-rays.origin)/norm

      d[abs(d) < 1E-14] = 0

      X = rays.origin + rays.direction*d.reshape(rays.n,1,1)
      x,y,z = X[:,:,0].T
      Ox,Oy,Oz = O

      r = ne.evaluate('sqrt((x-Ox)**2+(y-Oy)**2+(z-Oz)**2)').reshape(rays.n,1)
      mask = ((r > self.radius)+(d < 0)).flatten()

      d[mask] = np.inf
      mask = np.hstack([D,d]).argmin(1)
      D = np.hstack([D,d]).min(1).reshape(rays.n,1)
      R = np.hstack([R,r])[np.arange(rays.n), mask].reshape(rays.n,1)

    mask = (D != np.inf).flatten()
    X[mask] = rays.origin[mask] \
              + rays.direction[mask]*D[mask].reshape(mask.sum(),1,1)

    return D,X,R

  # Interaction

  def interact(self, rays, R, T, A, mask_in = None):

    # Check intersection

    mask,X,r = self.intersect(rays)
    mask = (mask != np.inf).flatten()

    if mask_in is not None:
      mask *= mask_in

    # Deterimine absorption, reflection, transmission

    u = np.random.rand(mask.sum())
    amask = np.zeros_like(mask)
    amask[mask] = (A >= u)
    rmask = np.zeros_like(mask)
    rmask[mask] = np.logical_not(amask[mask])
    rmask[mask] *= ((R+A) >= u)
    tmask = np.zeros_like(mask)
    tmask[mask] = np.logical_not(rmask[mask])*np.logical_not(amask[mask])

    # Transmit rays

    if tmask.sum() != 0:
      rays.origin[tmask] = X[tmask]

    # Reflect rays

    if rmask.sum() != 0:
      rays.origin[rmask] = X[rmask]
      rays.direction[rmask] = reflect(rays.direction[rmask], \
                                      self.plane.reshape(3,))

    # Absorb rays

    if amask.sum() != 0:
      bins = np.arange(0,self.radius+1E-14,self.ring_width)
      self.nabs += np.histogram(r[amask], bins)[0]
      rays.reduce(np.logical_not(amask))

    return amask

  def plot(self, ax, color = 'k', alpha = 0.2):
    phi = np.linspace(0,2*np.pi,self.nrings*10)
    z = np.linspace(0,self.thickness,self.nrings*10)

    phi,z = np.meshgrid(phi,z)
    mshape = z.shape
    phi = phi.flatten()
    z = z.flatten()
    x = self.radius*np.cos(phi)
    y = self.radius*np.sin(phi)

    R = rotation_from_zplane(self.plane)
    X = np.vstack([x,y,z])
    X = np.matmul(R,X)

    X += self.origin
    x,y,z = X
    x = x.reshape(mshape)
    y = y.reshape(mshape)
    z = z.reshape(mshape)

    ax.plot_surface(x, y, z, color = color[-1], lw = 0, alpha = alpha)

    r = np.linspace(0,self.radius,self.nrings*10)
    phi = np.linspace(0,2*np.pi,self.nrings*10)

    if type(color) == str:
      color = np.repeat(color, self.nrings)

    r,phi = np.meshgrid(r,phi)
    mshape = r.shape
    r = r.flatten()
    phi = phi.flatten()
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    z1 = np.zeros_like(x)
    z2 = z1+self.thickness

    idx = (r.reshape(mshape)/self.ring_width).astype(int)
    idx[idx > self.nrings-1] = self.nrings-1
    c = color[idx]

    R = rotation_from_zplane(self.plane)
    X1 = np.vstack([x,y,z1])
    X2 = np.vstack([x,y,z2])
    X1 = np.matmul(R,X1)
    X2 = np.matmul(R,X2)

    X1 += self.origin
    x,y,z = X1
    x = x.reshape(mshape)
    y = y.reshape(mshape)
    z = z.reshape(mshape)

    ax.plot_surface(x, y, z, facecolors = c, lw = 0, alpha = alpha)

    X2 += self.origin
    x,y,z = X2
    x = x.reshape(mshape)
    y = y.reshape(mshape)
    z = z.reshape(mshape)

    ax.plot_surface(x, y, z, facecolors = c, lw = 0, alpha = alpha)

# An uncapped cylinder

class Cylinder(object):

  # Geometric Parameters

  origin = None
  plane = None
  radius = None
  height = None
  thickness = None

  # Material properties

  material = None
  temperature = None
  reflectivity = None
  transmissivity = None
  absorptivity = None

  # Interaction

  nabs = None

  # Initialization

  def __init__(self, origin, plane, radius, height, thickness, material, \
               temperature, spectra = None):
    self.origin = np.array(origin).reshape(3,1)
    self.plane = (np.array(plane)/np.linalg.norm(plane)).reshape(3,1)
    self.radius = radius
    self.height = height
    self.area = 2*np.pi*radius*height

    self.material = material
    self.temperature = temperature
    self.thickness = thickness

    if spectra is None:

      if material.lower() == 'black':
        self.reflectivity = 0.0
        self.transmissivity = 0.0
        self.absorptivity = 1.0
      elif material.lower() == 'transparent':
        self.reflectivity = 0.0
        self.transmissivity = 1.0
        self.absorptivity = 0.0
      else:
        self.reflectivity = 1-thermE.get_emissivity(material,temperature)
        self.transmissivity = 0.0
        self.absorptivity = thermE.get_emissivity(material,temperature)

    else:
      f,T,A = spectra.T

      self.transmissivity = interpolate(f,T,'linear', bounds_error = False, \
                                        fill_value = (T[0],T[-1]))
      self.absorptivity = interpolate(f,A,'linear', bounds_error = False, \
                                      fill_value = (A[0],A[-1]))
      self.reflectivity = lambda freq: 1 - self.absorptivity(freq) - \
                                       self.transmissivity(freq)

    self.nabs = 0.0

  # Generate ray bundle from surface

  def generate_rays(self, n, side = 'inner'):

    # Sample the surface

    hh = np.random.rand(n)*self.height
    alpha = np.random.rand(n)*2*np.pi

    X = np.zeros((n,3,1))
    X[:,0] = (self.radius*np.cos(alpha)).reshape(n,1)
    X[:,1] = (self.radius*np.sin(alpha)).reshape(n,1)
    X[:,2] = hh.reshape(n,1)

    R = rotation_from_zplane(self.plane)
    X = np.matmul(R,X) + self.origin.reshape(3,1)

    # Generate random directions

    u1 = np.random.rand(n)
    u2 = np.random.rand(n)
    theta = 0.5*np.arccos(1-2*u1)
    phi = 2*np.pi*u2

    N = np.zeros_like(X)
    N[:,0] = (np.sin(theta)*np.cos(phi)).reshape(n,1)
    N[:,1] = (np.sin(theta)*np.sin(phi)).reshape(n,1)
    N[:,2] = (np.cos(theta)).reshape(n,1)

    Ns = X - self.origin
    Ns -= (np.matmul(self.plane.flatten(), X-self.origin).flatten()
         * self.plane).T.reshape(len(X),3,1)
    Ns /= self.radius
    R = rotation_from_zplane(Ns)

    N = np.matmul(R,N)

    # Front (back) is in the +N (-N) direction

    if side == 'outer':
      rays = RayBundle(X+self.thickness*Ns,N)
    elif side == 'inner':
      rays = RayBundle(X,-N)
    elif side == 'both':
      X = np.vstack([X[:len(X)/2],X[len(X)/2:]+self.thickness*Ns[len(X)/2:]])
      N = np.vstack([-N[:len(X)/2],N[len(X)/2:]])
      rays = RayBundle(X,N)

    return rays

  # Intersection

  def intersect(self, rays):

    ### ONLY WORKS FOR ZERO THICKNESS ###

    N = self.plane.reshape(3,)

    a = rays.direction - (np.matmul(N,rays.direction)*N).reshape(rays.n,3,1)
    b = rays.origin - self.origin \
        - (np.matmul(N,rays.origin-self.origin)*N).reshape(rays.n,3,1)
    e = np.matmul(np.transpose(b,(0,2,1)),b) - self.radius**2
    b = 2*np.matmul(np.transpose(a,(0,2,1)),b)
    a = np.matmul(np.transpose(a,(0,2,1)),a)

    d = ne.evaluate('b**2 - 4*a*e')
    d[abs(d) < 1E-14] = 0
    mask = (d >= 0).flatten()
    dm,bm,am = (d[mask],b[mask],a[mask])
    bm = ne.evaluate('-bm/(2*am)')
    am = ne.evaluate('sqrt(dm)/(2*am)')
    dm = ne.evaluate('bm-am')
    am = am[dm < 1E-14]
    bm = dm[dm < 1E-14]
    dm[dm < 1E-14] = ne.evaluate('bm+2*am')
    d[mask] = dm
    d[abs(d) < 1E-14] = 0

    X = rays.origin + rays.direction*d.reshape(rays.n,1,1)

    mask = (d < 0).reshape(rays.n,1)
    mask += (np.matmul(N,X-self.origin) < -1E-14)
    mask += (np.matmul(N,X-(self.origin+(N*self.height).reshape(3,1))) > 1E-14)
    mask = mask.flatten()

    N = X - self.origin
    N -= (np.matmul(self.plane.flatten(), X-self.origin).flatten()
         * self.plane).T.reshape(len(X),3,1)
    N /= self.radius

    d[mask] = np.inf

    return d,X,N

  # Interaction

  def interact(self, rays, R, T, A, mask_in = None):

    # Check intersection

    mask,X,N = self.intersect(rays)
    mask = (mask != np.inf).flatten()

    if mask_in is not None:
      mask *= mask_in

    # Deterimine absorption, reflection, transmission

    u = np.random.rand(mask.sum())
    amask = np.zeros_like(mask)
    amask[mask] = (A >= u)
    rmask = np.zeros_like(mask)
    rmask[mask] = np.logical_not(amask[mask])
    rmask[mask] *= ((R+A) >= u)
    tmask = np.zeros_like(mask)
    tmask[mask] = np.logical_not(rmask[mask])*np.logical_not(amask[mask])

    # Transmit rays

    if tmask.sum() != 0:
      rays.origin[tmask] = X[tmask]

    # Reflect rays

    if rmask.sum() != 0:
      rays.origin[rmask] = X[rmask]
      rays.direction[rmask] = reflect(rays.direction[rmask], \
                                      np.transpose(N[rmask],(0,2,1)))

    # Absorb rays

    if amask.sum() != 0:
      self.nabs += amask.sum()
      rays.reduce(np.logical_not(amask))

    return amask

  def plot(self, ax, color = 'k', alpha = 0.2):
    phi = np.linspace(0,2*np.pi,100)
    z = np.linspace(0,self.height,100)

    phi,z = np.meshgrid(phi,z)
    mshape = z.shape
    phi = phi.flatten()
    z = z.flatten()
    x = self.radius*np.cos(phi)
    y = self.radius*np.sin(phi)

    R = rotation_from_zplane(self.plane)
    X = np.vstack([x,y,z])
    X = np.matmul(R,X)

    X += self.origin
    x,y,z = X
    x = x.reshape(mshape)
    y = y.reshape(mshape)
    z = z.reshape(mshape)

    ax.plot_surface(x, y, z, color = color, lw = 0, alpha = alpha)

# An uncapped hemisphere

class HemiSphere(object):

  # Geometric Parameters

  origin = None
  plane = None
  radius = None
  area = None
  thickness = None

  # Material properties

  material = None
  temperature = None
  reflectivity = None
  transmissivity = None
  absorptivity = None

  # Interaction

  nabs = None

  # Initialization

  def __init__(self, origin, plane, radius, thickness, material, temperature, \
               spectra = None):
    self.origin = np.array(origin).reshape(3,1)
    self.plane = (np.array(plane)/np.linalg.norm(plane)).reshape(3,1)
    self.radius = radius
    self.area = 2*np.pi*radius**2

    self.material = material
    self.temperature = temperature
    self.thickness = thickness

    if spectra is None:

      if material.lower() == 'black':
        self.reflectivity = 0.0
        self.transmissivity = 0.0
        self.absorptivity = 1.0
      elif material.lower() == 'transparent':
        self.reflectivity = 0.0
        self.transmissivity = 1.0
        self.absorptivity = 0.0
      else:
        self.reflectivity = 1-thermE.get_emissivity(material,temperature)
        self.transmissivity = 0.0
        self.absorptivity = thermE.get_emissivity(material,temperature)

    else:
      f,T,A = spectra.T

      self.transmissivity = interpolate(f,T,'linear', bounds_error = False, \
                                        fill_value = (T[0],T[-1]))
      self.absorptivity = interpolate(f,A,'linear', bounds_error = False, \
                                      fill_value = (A[0],A[-1]))
      self.reflectivity = lambda freq: 1 - self.absorptivity(freq) - \
                                       self.transmissivity(freq)

    self.nabs = 0.0

  # Generate ray bundle from surface

  def generate_rays(self, n, side = 'inner'):

    # Sample the surface

    phi = 2*np.pi*np.random.rand(n)
    theta = np.arccos(2*np.random.rand(n)-1)
    theta[theta > np.pi/2] = np.pi - theta[theta > np.pi/2]

    X = np.zeros((n,3,1))

    x = self.radius*np.sin(theta)*np.cos(phi)
    y = self.radius*np.sin(theta)*np.sin(phi)
    z = self.radius*np.cos(theta)

    X[:,0] = (self.radius*np.sin(theta)*np.cos(phi)).reshape(n,1)
    X[:,1] = (self.radius*np.sin(theta)*np.sin(phi)).reshape(n,1)
    X[:,2] = (self.radius*np.cos(theta)).reshape(n,1)

    R = rotation_from_zplane(self.plane)
    X = np.matmul(R,X) + self.origin.reshape(3,1)

    # Generate random directions

    u1 = np.random.rand(n)
    u2 = np.random.rand(n)
    theta = 0.5*np.arccos(1-2*u1)
    phi = 2*np.pi*u2

    N = np.zeros_like(X)
    N[:,0] = (np.sin(theta)*np.cos(phi)).reshape(n,1)
    N[:,1] = (np.sin(theta)*np.sin(phi)).reshape(n,1)
    N[:,2] = (np.cos(theta)).reshape(n,1)

    Ns = X - self.origin
    Ns /= self.radius
    R = rotation_from_zplane(Ns)

    N = np.matmul(R,N)

    # Front (back) is in the +N (-N) direction

    if side == 'outer':
      rays = RayBundle(X+self.thickness*Ns,N)
    elif side == 'inner':
      rays = RayBundle(X,-N)
    elif side == 'both':
      X = np.vstack([X[:len(X)/2],X[len(X)/2:]+self.thickness*Ns[len(X)/2:]])
      N = np.vstack([-N[:len(X)/2],N[len(X)/2:]])
      rays = RayBundle(X,N)

    return rays

  # Intersection

  def intersect(self, rays):
    N = self.plane.reshape(3,)

    e = rays.origin - self.origin
    b = np.matmul(np.transpose(rays.direction,(0,2,1)),e)
    e = np.matmul(np.transpose(e,(0,2,1)),e)-self.radius**2

    d = ne.evaluate('b**2 - e')
    d[abs(d) < 1E-14] = 0
    dmask = (d >= 0).flatten()
    dm,bm = (d[dmask],b[dmask])
    dm = np.hstack([ne.evaluate('-bm-sqrt(dm)'),ne.evaluate('-bm+sqrt(dm)')])
    dm = dm.reshape(len(dm),1,2)
    dm[abs(dm) < 1E-14] = 0

    X = rays.origin[dmask] + rays.direction[dmask]*dm.reshape(len(dm),1,2)

    mask = (np.matmul(N,X-self.origin) < -1E-14).reshape(len(dm),1,2)
    mask += (dm < 1E-14)
    dm[mask] = np.inf
    dm = dm.min(2).reshape(len(dm),1,1)
    dm[dm == np.inf] = 0
    d[dmask] = dm
    d[abs(d) < 1E-14] = 0

    X = rays.origin + rays.direction*d.reshape(rays.n,1,1)

    mask = (d <= 0).reshape(rays.n,1)
    mask = mask.flatten()

    N = X - self.origin
    N /= self.radius

    d[mask] = np.inf

    return d,X,N

  # Interaction

  def interact(self, rays, R, T, A, mask_in = None):

    # Check intersection

    mask,X,N = self.intersect(rays)
    mask = (mask != np.inf).flatten()

    if mask_in is not None:
      mask *= mask_in

    # Deterimine absorption, reflection, transmission

    u = np.random.rand(mask.sum())
    amask = np.zeros_like(mask)
    amask[mask] = (A >= u)
    rmask = np.zeros_like(mask)
    rmask[mask] = np.logical_not(amask[mask])
    rmask[mask] *= ((R+A) >= u)
    tmask = np.zeros_like(mask)
    tmask[mask] = np.logical_not(rmask[mask])*np.logical_not(amask[mask])

    # Transmit rays

    if tmask.sum() != 0:
      rays.origin[tmask] = X[tmask]

    # Reflect rays

    if rmask.sum() != 0:
      rays.origin[rmask] = X[rmask]
      rays.direction[rmask] = reflect(rays.direction[rmask], \
                                      np.transpose(N[rmask],(0,2,1)))

    # Absorb rays

    if amask.sum() != 0:
      self.nabs += amask.sum()
      rays.reduce(np.logical_not(amask))

    return amask

  def plot(self, ax, color = 'k', alpha = 0.2):
    phi = np.linspace(0,2*np.pi,100)
    theta = np.linspace(0,np.pi/2,100)

    phi,theta = np.meshgrid(phi,theta)
    mshape = phi.shape
    phi = phi.flatten()
    theta = theta.flatten()
    x = self.radius*np.sin(theta)*np.cos(phi)
    y = self.radius*np.sin(theta)*np.sin(phi)
    z = self.radius*np.cos(theta)

    R = rotation_from_zplane(self.plane)
    X = np.vstack([x,y,z])
    X = np.matmul(R,X)

    X += self.origin
    x,y,z = X
    x = x.reshape(mshape)
    y = y.reshape(mshape)
    z = z.reshape(mshape)

    ax.plot_surface(x, y, z, color = color, lw = 0, alpha = alpha)
