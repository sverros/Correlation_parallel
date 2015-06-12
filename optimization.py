import numpy as np

def optimize(var, size):

    # Groups the station latitudes into rows
    row_lats = np.zeros(var['M'])
    for i in range(0,var['M']):
        row_lats[i] = var['site_collection_SM'].lats[i*var['N']]
    
    sta_lats = var['site_collection_station'].lats

    # Determines 'weight' for each row, beginning with 1 and adding 5 for each 
    # additional station in the row
    weights = np.ones(var['M'])
    for i in range(0,var['K']):
        for j in range(1,var['M']):
            if ((sta_lats[i] >= row_lats[j])and(sta_lats[i] <= row_lats[j-1]))or(
                (sta_lats[i] <= row_lats[j])and(sta_lats[i] >= row_lats[j-1])):
                weights[j] += 5
                break

    # Approximate goal weight for each core
    total = np.floor(sum(weights)/size)

    n_rows = np.zeros(size)
    approx_wt = np.zeros(size)

    # Determine the number of rows for the first size-1 cores
    # without exceeding total
    for i in range(0,size-1):
        while approx_wt[i] < total:
            n_rows[i] += 1
            approx_wt[i] += weights[int(sum(n_rows))]
        n_rows[i] -= 1


    approx_wt[-1] = sum(weights[sum(n_rows):])

    # If the last core has more than 1.5 times all other cores,
    # give the other cores an additional row
    if 0.66*approx_wt[-1] > approx_wt[0:-2].all():
        n_rows += 1
        
    diff = var['M'] - sum(n_rows)
    n_rows[-1] += diff

    return n_rows
