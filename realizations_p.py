import numpy as np
import time
def realizations(num_realizations, N,M, size, uncertaintydata, DATA, grid_arr, mu_arr, sigma_arr):

    t_realizations = 0
    t_start = time.time()

    for j in range(0, num_realizations):

        t_s_r = time.time()
        X = np.zeros([M*N,1])
        rand_arr = np.random.randn(M*N)
        X[0] = rand_arr[0]

        for i in range(1,M*N):
            nzeros = (np.size(mu_arr[i]) - np.size(grid_arr[i]))
            x = np.append(np.zeros(nzeros), [X[int(k)] for k in grid_arr[i]])
            mu = np.dot([float(st) for st in mu_arr[i]], x)
            X[i] = mu + rand_arr[i] * float(sigma_arr[i])

        COR = np.reshape(X, [M,N])
        X = np.multiply(COR, uncertaintydata)
        DATA_NEW = DATA * np.exp(X)

        if j == 0:
            ACCUM_ARRAY = DATA_NEW.copy()
        else:
            ACCUM_ARRAY += DATA_NEW
        
        if np.mod(j+1, 25) == 0:
            print "Done with", j+1, "of", num_realizations, "iterations."
        t_realizations += time.time() - t_s_r

    ACCUM_ARRAY = ACCUM_ARRAY / num_realizations
    t_total = time.time() - t_start
    avg_t = t_realizations/num_realizations

    print 'total realization time', t_total, 'sec'
    print 'avg realization time', avg_t, 'sec'


    return {'ACCUM_ARRAY':ACCUM_ARRAY, 'COR':COR, 'DATA_NEW':DATA_NEW, 'DATA':DATA}
