from mpi4py import MPI
import numexpr as ne
import numpy as np
from scipy import linalg
import math
import matplotlib.pyplot as plt
import datetime
from neicio.readstation import readStation
from neicio.shake import ShakeGrid
from openquake.hazardlib.correlation import JB2009CorrelationModel
from openquake.hazardlib.correlation import BaseCorrelationModel
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.geo import Point
from openquake.hazardlib.geo.geodetic import geodetic_distance
from openquake.hazardlib.imt import from_string
import time
from matplotlib import cm
from neicio.gmt import GMTGrid
import time

def compute(var, ll, ul, r, voi, intensity_factor):
    # Computes the vectors for multiple realizations
    # INPUTS: var - dict output from the initialize function
    #         ll - lower limit row, different for each core
    #         ul - upper limit row, different for each core
    #         r  - radius of influence
    #         voi - variable of interest i.e. PGA
    #         intensity factor - multiplier on non-native observations
    # OUTPUTS: mu_arr - array of calculated Sig12*Sig11inv for each point
    #          grid_arr - array of indices on which the point depends
    #          sigma_arr - array of calculated sigma for each point

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    size = comm.Get_size()

    M = var['M']
    N = var['N']
    K = var['K']

    cor_model = BaseCorrelationModel
    JB_cor_model = JB2009CorrelationModel(cor_model)

    if my_rank == 0:
        ll_e = np.zeros(size)
        ul_e = np.zeros(size)
    else:
        ll_e = None
        ul_e = None

    # Master gathers upper and lower limits from each core
    ll_e = comm.gather(ll, root = 0)
    ul_e = comm.gather(ul, root = 0)

    total_entries = (ul - ll)*N

    if my_rank == 0:
        # Master initialized entire arrays
        grid_arr = [None]*(M*N)
        mu_arr = [None]*(M*N)
        sigma_arr = np.zeros([M*N,1])
    else:
        # Other cores initialize subsets of the arrays
        grid_arr = [None]*(total_entries)
        mu_arr = [None]*(total_entries)
        sigma_arr = np.zeros([total_entries,1])

    # Get spcing of horozontal and vertical points
    ld  = set_up_grid_dist(M,N,var['site_collection_SM'])

    for i in range(ll, ul):
        
        # Find the number of points in radius horozontally and vertically for each row
        vhva = calc_vert_hor(i, r, ld['l'], ld['d'])
        
        # Calculate the full distance matrix for each row
        dist = calc_full_dist(vhva['vert'], vhva['hor'], N, var['site_collection_SM'])

        first_time_per_row = 1

        for j in range(0,N):

            num = i*N+j - ll*N
            
            # Find the reduced distance matrix 
            dist_calc = reduce_distance(
                j, vhva['vert'], vhva['hor'], vhva['added_vert'], N, 
                dist['distance_matrix'], dist['grid_indices'])

            # Include stations in distance matrix and find the 
            # indices of the points within the radius
            out = inc_stations(j, i, N, K, r, var['site_collection_SM'], 
                               var['site_collection_station'], dist_calc['dist_mat'], 
                               dist_calc['inc_ind'], dist_calc['inc_indices'])

            if np.size(dist_calc['inc_indices']) == 1:
                # Correlation value is not dependent on anything, result is random
                grid_arr [num]= np.zeros(0)
                mu_arr   [num]= np.zeros(0)
                sigma_arr[num]= 1
            else:
                # Check if reduced distance matrix is full distance matrix
                if ((vhva['vert'] == 1 and dist_calc['num_indices'] == vhva['hor']+1)or(vhva['vert'] != 1 and dist_calc['num_indices'] == 2*vhva['hor']+1)) and (np.size(out['inc_sta_indices']) == 0):
                    # If this is the first full distance matrix per row, calculate base case
                    if first_time_per_row == 1:
                        base = calculate_corr(
                            out['dist_mat'], voi, JB_cor_model, var, out['inc_sta_indices'],
                            intensity_factor)
                        first_time_per_row = 0
                    # Use the base case matrices
                    grid_arr [num] = np.array(dist_calc['inc_ind'][0:-1])
                    mu_arr   [num] = np.array((base['Sig12'].T*base['Sig11inv']).T)
                    sigma_arr[num] = base['R']
                else:
                    # Need to compute the correlation matrix for this individual case
                    other = calculate_corr(
                        out['dist_mat'], voi, JB_cor_model, var, out['inc_sta_indices'],
                        intensity_factor)
                    grid_arr [num] = np.array(dist_calc['inc_ind'][0:-1])
                    mu_arr   [num] = np.array((other['Sig12'].T*other['Sig11inv']).T)
                    sigma_arr[num] = other['R']

            if np.mod(i*N+j,5000) == 0:
                print 'Finishing step:', i*N+j
    t = time.time()
    xi = np.zeros(2)
    sd_rk = np.array([my_rank])

    if my_rank != 0:
        # Send my_rank to the master 
        comm.send(sd_rk, dest = 0, tag = 1)
        print my_rank, 'sending to master'
        for i in range(0,np.size(grid_arr)):
            xi[0] = np.size(grid_arr[i])
            xi[1] = np.size(mu_arr[i])
            # Send the sizes of the grid and mu arrays
            comm.Send(xi, dest = 0, tag = my_rank*111)
            # Send the grid, mu, and sigma arrays
            comm.Send([grid_arr[i], MPI.INTEGER], dest = 0, tag = i*my_rank*11)
            comm.Send([mu_arr[i], MPI.DOUBLE], dest = 0, tag = i*my_rank*12)
            comm.Send([sigma_arr[i], MPI.DOUBLE], dest = 0, tag = i*my_rank*13)
    else:
        for i in range(1,size):
            current_pt = np.zeros(1)
            # Recieve the rank of a cpu ready to send
            current_pt = comm.recv(source = MPI.ANY_SOURCE, tag = 1)
            
            print 'master recieving from', current_pt[0]
            for j in range(0,N*(ul_e[current_pt[0]] - ll_e[current_pt[0]])):
                pt = j + ul_e[current_pt[0]-1]*N
                # Recieve the sizes of the grid and mu arrays
                comm.Recv(xi, source = current_pt[0], tag = (current_pt[0])*111)
                grid_arr[pt] = np.zeros([xi[0],1])
                mu_arr[pt] = np.zeros([xi[1],1])
                # Recieve the grid, mu, and sigma arrays
                comm.Recv([grid_arr [pt], MPI.INTEGER],source = current_pt[0], tag = 11*(current_pt[0])*j)
                comm.Recv([mu_arr   [pt], MPI.DOUBLE], source = current_pt[0], tag = 12*(current_pt[0])*j)
                comm.Recv([sigma_arr[pt], MPI.DOUBLE], source = current_pt[0], tag = 13*(current_pt[0])*j)

    if my_rank != 0:
        grid_arr = None
        mu_arr = None
        sigma_arr = None
        
    print 'total sending time for ', my_rank,  'is', time.time() - t

    return {'grid_arr':grid_arr, 'mu_arr':mu_arr, 'sigma_arr':sigma_arr}


def calculate_corr(dist_mat, voi, JB_cor_model, var, inc_sta_indices,intensity_factor = 0.9):
    #####
    # Calculates correlation model for distance matrix and voi
    # IN: dist_mat- reduced distance matrix
    #     voi- variable of interest
    #     JB_cor_model- correlation model from correlation in oq-hazardlib
    #OUT: Sig12, Sig11inv- partitions of correlation matrix
    #     R - Sqrt of sigma
    #####
    correlation_model = JB_cor_model._get_correlation_model(dist_mat, from_string(voi))
    
    intensity = var['intensity'][inc_sta_indices]
    if np.size(intensity) != 0:
        for i in range(0,np.size(intensity)):
            if intensity[i] == 1:
                correlation_model[i,i+1:] = correlation_model[i,i+1:].copy()*intensity_factor
                correlation_model[i+1:,i] = correlation_model[i+1:,i].copy()*intensity_factor

    Sig11 = np.mat(correlation_model[0:-1, 0:-1])
    Sig12 = np.mat(correlation_model[0:-1, -1]).T
    Sig22 = np.mat(correlation_model[-1,-1])

    Sig11inv = np.mat(np.linalg.pinv(Sig11))
    sigma = Sig22 - (Sig12.T*Sig11inv*Sig12)

    R = np.sqrt(max(sigma,0))
    
    if sigma < 0:
        print abs(sigma)

    return {'Sig12':Sig12, 'Sig11inv':Sig11inv, 'R':R}

def inc_stations(j, i, N, K, r, site_collection_SM, 
                 site_collection_station, dist_mat, inc_ind, inc_indices):
    #####
    # If there are stations included within the radius for a point, this function will add those stations to the 
    # distance matrix and determine the array of points included in the radius, x
    # IN: i,j- current points row and column 
    #     N,K - number of points in row and total number of stations
    #     r- radius 
    #     site_collection_SM/station- site collections for ShakeMap and station data
    #     dist_mat- reduced distance matrix
    #     X- array of previously calculated correlation values
    #     inc_ind- indices of included points
    #     inc_indices- total number of points in the top most row of distance matrix
    #OUT: dist_mat- reduced distance matrix, modified to include stations
    #     x- array of points in X included in radius and stations included 
    #     inc_sta_indices- indices of stations included in the radius
    #####

    num = i*N+j
    
    # Compute the distances for all stations to the grid point we're looking at
    dist_sta_sit = np.array(geodetic_distance(
            site_collection_SM.lons[j+i*N], site_collection_SM.lats[j+i*N],
            site_collection_station.lons[0:K], site_collection_station.lats[0:K]))
        
    # Find which of those stations are in the radius we are considering
    inc_sta_indices = np.where(dist_sta_sit < r)
    if np.size(inc_sta_indices) != 0:
            
        station_distance_matrix = np.zeros(
            [np.size(inc_sta_indices), np.size(inc_sta_indices)+np.size(inc_ind)])
        # Calculate distance between each included station and all included grid points, 
        # then calculate the distance
        # from each included station to every other included station
        for eta in range(0, np.size(inc_sta_indices)):
            for beta in range(0,np.size(inc_ind)):
                station_distance_matrix[eta,np.size(inc_sta_indices) + beta] = geodetic_distance(
                    site_collection_station.lons[inc_sta_indices[0][eta]], 
                    site_collection_station.lats[inc_sta_indices[0][eta]], 
                    site_collection_SM.lons[inc_ind[beta]], 
                    site_collection_SM.lats[inc_ind[beta]])
            for beta in range(0, np.size(inc_sta_indices)):
                station_distance_matrix[eta, beta] = geodetic_distance(
                    site_collection_station.lons[inc_sta_indices[0][eta]], 
                    site_collection_station.lats[inc_sta_indices[0][eta]],
                    site_collection_station.lons[inc_sta_indices[0][beta]], 
                    site_collection_station.lats[inc_sta_indices[0][beta]])
            
        # Concatenate the station distance matrix with the modified distance matrix, dist_mat
        dist_mat = np.concatenate((
                station_distance_matrix[:, np.size(inc_sta_indices):], dist_mat), axis=0)
        dist_mat = np.concatenate((
                station_distance_matrix.T, dist_mat), axis=1)
            
        # x: vector of previously calculated covariance values
        #x = np.concatenate((np.zeros([np.size(inc_sta_indices),1]),X[inc_ind,0]), axis = 0)
        #x = np.mat(x[0:-1])
            
        # x: vector of previously calculated covariance values
        #x = X[inc_ind,0]
        #x = np.mat(x[0:-1])
    
    return {'dist_mat':dist_mat, 'inc_sta_indices':inc_sta_indices}

def reduce_distance(j, vert, hor, added_vert, N, distance_matrix, grid_indices):
    # Find which columns/rows in the distance matrix to keep
    # IN: j- points column
    #     vert- number of rows included in the radius
    #     hor- number of columns included in radius
    #     added_vert- number of rows in between first row and first included row 
    #     N- number of points in row
    #     distance_matrix- full distance matrix
    #     grid_indices- indices included in full distance matrix
    #OUT: dist_mat- reduced distance matrix
    #     inc_indices- number of points in top most row of dist_mat
    #     inc_ind - indices of points included in dist_mat
    #     num_indices- number of points in top most row of distance_matrix

    inc_ind = [None]*(vert*(2*hor+1))
    n_grid_indices = 0
    num_indices = 0
    
    if j < hor:
        # On left end of grid
        inc_indices = range(0,np.size(grid_indices))
        if vert == 1:
            num_indices = (j+1)
            inc_indices = np.where(np.mod(inc_indices,hor+1) >=hor+1 - num_indices)
        else:
            num_indices = (j+hor+1)
            inc_indices = np.where(np.mod(inc_indices,2*hor+1) >=2*hor+1 - num_indices)
            
        for k in range(0,vert):
            if k == vert-1:
                for eta in range(0,j+1):
                    inc_ind[n_grid_indices] = [eta+N*k+ N*added_vert]
                    n_grid_indices += 1
            else:
                for eta in range(0,j+hor+1):
                    inc_ind[n_grid_indices] = [eta+N*k+N*added_vert]
                    n_grid_indices += 1
        del inc_ind[n_grid_indices:]
            
    elif j > N-hor-1:
        # On right end of grid
        inc_indices = range(0,np.size(grid_indices))
        if vert == 1:
            num_indices = (1+hor)
            inc_indices = np.where(np.mod(inc_indices,hor+1) >=hor+1 - num_indices)
        else:
            num_indices = (N - j+hor)
            inc_indices = np.where(np.mod(inc_indices,2*hor+1) <=num_indices-1)
        for k in range(0,vert):
            if k == vert-1:
                for eta in range(j-hor,j+1):
                    inc_ind[n_grid_indices] = [eta+N*k+N*added_vert]
                    n_grid_indices += 1
            else:
                for eta in range(j-hor,N):
                    inc_ind[n_grid_indices] = [eta+N*k+N*added_vert]
                    n_grid_indices += 1
        del inc_ind[n_grid_indices:]
            
    else:
        # In the middle of the grid, all points included
        inc_indices = range(0,np.size(grid_indices))
        if vert == 1:
            num_indices = (1+hor)
            inc_indices = np.where(np.mod(inc_indices,hor+1) >=hor+1 - num_indices)
        else:
            num_indices = (2*hor+1)
            inc_indices = np.where(np.mod(inc_indices,2*hor+1) >=2*hor+1 - num_indices)
        for k in range(0,vert):
            if k == vert-1:
                for eta in range(j-hor,j+1):
                    inc_ind[n_grid_indices] = [eta+N*k+N*added_vert]
                    n_grid_indices += 1
            else:
                for eta in range(j-hor,j+hor+1):
                    inc_ind[n_grid_indices] = [eta+N*k+N*added_vert]
                    n_grid_indices += 1
        del inc_ind[n_grid_indices:]

    inc_indices = np.array(inc_indices).flatten()
        
    # dist_mat: the distance matrix modified for the current point
    dist_mat = distance_matrix[:,inc_indices]
    dist_mat = dist_mat[inc_indices, :]

    return {'dist_mat':dist_mat, 'inc_indices':inc_indices, 'inc_ind':inc_ind, 'num_indices':num_indices}

def calc_full_dist(vert, hor, N, site_collection_SM):
    #####
    # Calculates full distance matrix. Called once per row.
    # IN: vert- number of included rows
    #     hor- number of columns within radius 
    #     N- number of points in row
    #     site_collection_SM- site collection for ShakeMap data
    #OUT: grid_indices- indices of points included in distance matrix
    #     distance_matrix- full distance matrix
    #####

    # gathers indices for full distance matrix for each row
    grid_indices = [None]*(vert*(2*hor+1))
    n_grid_indices = 0
    for k in range(0, vert):
        if k == vert-1:
            for j in range(0,hor+1):
                grid_indices[n_grid_indices] = [j + N*k]
                n_grid_indices += 1
        else:
            for j in range(0,2*hor+1):
                grid_indices[n_grid_indices] = [j + N*k]
                n_grid_indices += 1 
    del grid_indices[n_grid_indices:]

    distance_matrix = np.zeros([np.size(grid_indices), np.size(grid_indices)])
    grid_indices = np.vstack(grid_indices)
    
    # Create full distance matrix for row
    for k in range(0, np.size(grid_indices)):
        distance_matrix[k, k:] = geodetic_distance(
                   site_collection_SM.lons[grid_indices[k ]], site_collection_SM.lats[grid_indices[k]],
                   site_collection_SM.lons[grid_indices[k:]], site_collection_SM.lats[grid_indices[k:]]).flatten()
    
    distance_matrix = distance_matrix + distance_matrix.T
    
    return {'grid_indices':grid_indices, 'distance_matrix':distance_matrix}

def calc_vert_hor(i, r, l, d):
    ######
    # Calculates the number of vertical of horozontal points in the full distance matrix
    # IN: i- current row
    #     r- radius
    #     l,d- vectors of distances between points vertically and horozontally
    #OUT: vert- number of rows in radius above current point
    #     hor- number of columns in radius to the left of current point
    #     added_vert- number of rows in between first row and the first row included in radius
    ######
    hor = int(np.floor(r/l[1]))
    
    # number of data points in vertical direction
    k = i
    vert_r = 0
    vert = 0
    while (vert_r+d[k] <= r)&(k>-1):
        vert_r += d[k]
        k -= 1
        vert += 1
    
    # adjusts for the unincluded rows
    if i+1 > vert:
        added_vert = i - vert+1
    else:
        added_vert = 0
    return {'vert':vert, 'hor':hor, 'added_vert':added_vert}

def set_up_grid_dist(M,N, site_collection_SM):
    ######
    # Calculates the vertical and horozontal spacing between points for each row
    # IN: M,N- number of points in grid vertically and horozontally
    #     site_collection_SM- site collection for ShakeMap data
    # OUT: l,d- vectors of distances between points vertically and horozontally
    ######

    l = np.zeros([M-1])
    d = np.zeros([M])
    
    # Calculate vertical and horozonal spacing between points for each row
    l[:] = geodetic_distance(
        site_collection_SM.lons[range(N,N*M,N)],   site_collection_SM.lats[range(N,N*M,N)],
        site_collection_SM.lons[range(0,N*M-N,N)], site_collection_SM.lats[range(0,N*M-N,N)])
    d[:] = geodetic_distance(
        site_collection_SM.lons[range(0, M*N, N)], site_collection_SM.lats[range(0, M*N, N)],
        site_collection_SM.lons[range(1, M*N, N)], site_collection_SM.lats[range(1, M*N, N)])
    return {'l':l, 'd':d}
