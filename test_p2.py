from mpi4py import MPI
import datetime
from neicio.readstation import readStation
from neicio.shake import ShakeGrid
import numpy as np
import time
from matplotlib import cm
from neicio.gmt import GMTGrid
import sys
sys.path.append('/home/sverros/')
from Correlation.setup import initialize
from Correlation.loop_p2 import compute
from Correlation.realizations_p import realizations
#from Correlation.plotting import plot
from Correlation.optimization import optimize

comm = MPI.COMM_WORLD
size = comm.Get_size()
my_rank = comm.Get_rank()

t = time.time()

# Variable of interest
voi = 'PGA'

# Specify the radius of interest
r = 25
n_stations = 0
intensity_factor = 1
num_realizations = 1

# Get shakemap for desired variable, PGA, uncertainty grid and stationdata
# Selected Stations: Units in pctg
shakemap = ShakeGrid(
    '/home/sverros/Correlation/Inputs/grid%i.xml'%(n_stations), variable = '%s' % voi)

# Uncertainty Data: Units in ln(pctg)
uncertainty = ShakeGrid(
    '/home/sverros/Correlation/Inputs/uncertainty%i.xml'%(n_stations), 
    variable= 'STD%s' % voi)

# Station Data: Units in pctg
stationlist = '/home/sverros/Correlation/Inputs/stationlist%i.xml'%(n_stations)
stationdata = readStation(stationlist)


if my_rank == 0:
    print 'Calling initialize'
variables = initialize(shakemap, uncertainty, stationdata, True)

if my_rank == 0:
    print variables['M']*variables['N'], 'data points'
    print variables['K'], 'stations'

# Determine the row distribution 
if variables['K'] == 0:
    n_rows = np.floor(variables['M']/size)*np.ones(size)
    n_rows[0] += np.mod(variables['M'], size)
else:
    n_rows = optimize(variables, size)

if my_rank == 0:
    print 'row distribution', n_rows

# Each core determines its lower and upper limits
low_limit = int(sum(n_rows[0:my_rank]))
up_limit = int(sum(n_rows[0:my_rank+1]))

# Compute the arrays grid_arr, mu_arr, sigma_arr
output = compute(variables, low_limit, up_limit, r, voi, intensity_factor)

time_end = time.time() - t
print my_rank, 'completed in time:', time_end, 'sec'

comm.Barrier()
if my_rank == 0:    
    real = realizations(num_realizations, variables['N'], variables['M'], size, 
                        variables['uncertaintydata'], variables['data'], 
                        output['grid_arr'], output['mu_arr'], output['sigma_arr'])
#    plot(real, variables, voi, shakemap, stationdata, real['ACCUM_ARRAY'])
