from numba import jit
import numpy as np
#from model_losses import model_loss, MMI_conv, initialize_loss

def realizations(num_realizations, my_rank, radius, N,M, grid_arr, mu_arr, sigma_arr, uncertaintydata, DATA, compute_loss, shakemap, voi):

    if num_realizations == 0:
        return
    else:
        if compute_loss == True:
            popgrid, isogrid, ratedict = initialize_loss(shakemap)
            ccodes = [None]*num_realizations
            fats = [None]*num_realizations

        g = open(('corr_R%i_rank%i_CC0_JB' % (radius, my_rank)), 'w')
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

            if compute_loss == True:
                mmi_data = MMI_conv(DATA_NEW, voi)
                ccodes[j], fats[j] = model_loss(mmi_data, popgrid, isogrid, ratedict)
                print fats
            if np.mod(j+1, 25) == 0:
                print "Done with", j+1, "of", num_realizations, "iterations."

        g.close()


        return {'cor':COR, 'data':DATA, 'data_new':DATA_NEW}
