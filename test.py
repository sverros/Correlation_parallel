#######################################################
# Parallel code for computing the spatial correlation for a ShakeMap,
# adding to a ShakeMap grid, and computing multiple realizations
# VARIABLES:
#     voi - variable of interest, i.e. PGA
#     r - radius of influence
#     num_realization- integer for desired number of realizations
#     corr_model- JB2009 or GA2010
#     vs_corr- Vs30 correlated bool, see JB2009
#     input data- grid.xml, uncertainty.xml, and stationlist.xml
#         stored in Inputs directory
# mpi4py is used for parallelization
# File may be run using:
# mpiexec -n # python test.py
# where # is the desired number of processors
#######################################################
from mpi4py import MPI
from neicio.readstation import readStation
from neicio.shake import ShakeGrid
import numpy as np
import time
from matplotlib import cm
import matplotlib.pyplot as plt
from neicio.gmt import GMTGrid
import sys
sys.path.append('/home/sverros/Correlation_parallel')
from setup import initialize
from loop import main
from realizations import realizations
#from plotting import plot

comm = MPI.COMM_WORLD
size = comm.Get_size()
my_rank = comm.Get_rank()

voi = 'PGA'
r = [25]
num_realizations = 10
corr_model = 'JB2009'
vscorr = True
plot_on = False
compute_loss = False

for R in range(0, np.size(r)):

    total_time_start = time.time()
    radius = r[R]
    
    # Get shakemap for desired variable, PGA, uncertainty grid and stationdata
    shakemap = ShakeGrid('Inputs/grid.xml', variable = '%s' % voi)
    
    # Uncertainty Data: Units in ln(pctg)
    unc_INTRA = ShakeGrid('Inputs/uncertainty.xml', variable = 'GMPE_INTRA_STD%s' % voi)
    unc_INTER = ShakeGrid('Inputs/uncertainty.xml', variable = 'GMPE_INTER_STD%s' % voi)
    
    # Station Data: Units in pctg
    stationlist = 'Inputs/stationlist.xml'
    stationdata = readStation(stationlist)

    # Initialize the grid
    if my_rank == 0:
        print 'Calling initialize'
    variables = initialize(shakemap, unc_INTRA, unc_INTER, stationdata)
    if my_rank == 0:
        print 'Radius:', radius
        print variables['K'], 'stations', variables['M']*variables['N'], 'data points'
        
    initialization_time = time.time() - total_time_start

    # Compute the random vector
    rand = np.random.randn(variables['N']*variables['M'])

    # Compute the grid, mu, and sigma arrays
    if my_rank == 0:
        print 'Calling main'
    out = main(variables, radius, voi, rand, corr_model, vscorr)
    
    main_time = time.time() - total_time_start - initialization_time

    if num_realizations == 1:
        # Master will compute this single realization
        if my_rank == 0:
            print 'Computing realizations'
            data = realizations(1, my_rank, radius, variables['N'], 
                            variables['M'], out['grid_arr'], out['mu_arr'], 
                            out['sigma_arr'], variables['uncertaintydata'], variables['data'], 
                            compute_loss, shakemap, voi)
    else:
        # Master broadcasts the arrays to the other cores
        if my_rank == 0:
            grid_arr = out['grid_arr']
            mu_arr = out['mu_arr']
            sigma_arr = out['sigma_arr']
        else:
            grid_arr = None
            mu_arr = None
            sigma_arr = None

        grid_arr = comm.bcast(grid_arr, root = 0)
        mu_arr = comm.bcast(mu_arr, root = 0)
        sigma_arr = comm.bcast(sigma_arr, root = 0)
        my_reals = np.arange(my_rank, num_realizations, size) 
        # Each core does a set of realizations
        data = realizations(np.size(my_reals), my_rank, radius, variables['N'], 
                            variables['M'], grid_arr, mu_arr, sigma_arr, 
                            variables['uncertaintydata'], variables['data'], 
                            compute_loss, shakemap, voi)

    realization_time = time.time() - total_time_start - initialization_time - main_time

    if plot_on == True:
        if my_rank == 0:
            print 'Plotting results'
            plot(data, variables, voi, shakemap, stationdata)

    if my_rank == 0:
        print 'Total time', time.time() - total_time_start
        print 'Init  time', initialization_time
        print 'Main  time', main_time
        print 'Realz time', realization_time
        
    comm.Barrier()
