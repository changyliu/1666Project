import numpy as np

def construction_heuristic(instance):
    # get the pickup indices
    pickup = [x-1 for x in instance['pickup'] if x != 0]
    pickup.sort()

    # sort pickup requests in the ascending order of est
    # tie-break by choosing earlier latest arrival time
    est = [instance['tw'][i][0] for i in pickup]
    lst = [instance['tw'][i][1] for i in pickup]
    idxs = [i for i in range(len(est))]

    sorted = []
    while len(est) > 0:
        cands = [i for i, x in enumerate(est) if x == min(est)]
        if len(cands) > 1: # tie-break
            ls = [lst[i] for i in cands]
            _idx = np.argmin(ls)
            idx = cands[_idx]
        else:
            idx = cands[0]

        sorted.append(pickup[idxs[idx]])
        est.pop(idx)
        lst.pop(idx)
        idxs.pop(idx)

    # insert the delivery node right after each pickup node
    tour = [0] # start from depot
    for p in sorted:
        tour.append(p)
        d = instance['delivery'][p]-1 # delivery request
        tour.append(d)
    tour = [x+1 for x in tour]

    return tour