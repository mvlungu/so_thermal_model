#!/usr/bin/python -u

import matplotlib as mpl
from matplotlib import rc
rc('text', usetex=True)
mpl.use('Agg')
import pylab

img=mpl.image.imread('../stacks/pb2/transfer_matrix_00.png')
norm = mpl.colors.LogNorm(vmin=1e-12, vmax=1)

mpl.pyplot.gcf().set_size_inches(7,6)
mpl.pyplot.imshow(img, interpolation = 'none', cmap = 'jet', norm = norm)
cb = mpl.pyplot.colorbar(shrink = 0.92)
cb.set_label(r'\rm{Transfer Factor (m}$^2$\rm{)}', fontsize = 12)
mpl.pyplot.xlabel(r'\rm{Surface Number}', fontsize = 12)
mpl.pyplot.ylabel(r'\rm{Surface Number}', fontsize = 12)
mpl.pyplot.title(r'\rm{%.2e Hz - %.2e Hz}' %(0.00e0, 1e10), \
                    fontsize = 14)
mpl.pyplot.tight_layout()


