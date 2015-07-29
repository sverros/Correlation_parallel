from numba import jit
import numpy as np

#@jit
def realizations(num_realizations, my_rank, radius, N,M, grid_arr, mu_arr, sigma_arr, uncertaintydata, DATA):

    if num_realizations == 0:
        return
    else:
        #h = open('random_arrays_100', 'r')
        g = open(('corr_output_R%i_rank%i' % (radius, my_rank)), 'w')
        s = str(num_realizations)
        g.write(s+'\n')

        for j in range(0, num_realizations):
            rand_arr = np.random.randn(M*N)
            X = np.zeros([M*N,1])
            for i in range(0,M*N):
                nzeros = np.size(mu_arr[i]) - np.size(grid_arr[i])
                x = np.append(np.zeros(nzeros), X[np.array(grid_arr[i], dtype = 'i')])
                mu = np.dot(mu_arr[i], x)
                X[i] = mu + rand_arr[i] * sigma_arr[i]
                s = str(X[i])
                g.write(s+'\n')

            COR = np.reshape(X, [M,N])
            X = np.multiply(COR, uncertaintydata)
            DATA_NEW = DATA * np.exp(X)
                
            if np.mod(j+1, 25) == 0:
                print "Done with", j+1, "of", num_realizations, "iterations."

        g.close()
#        h.close()

        return {'cor':COR, 'data':DATA, 'data_new':DATA_NEW}
