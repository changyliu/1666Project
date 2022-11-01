# Check if a solution is feasible

'''
1
162
105
19
55
186
133
8
182
118
50
...
'''

from distance import Distance_EUC_2D

def checkPDPTW(soln, instance):
    precedence_check = True
    tw_chek = True

    # cost = 0
    # for i in range(len(soln)-1):
    #     cost += Distance_EUC_2D(instance['coordinates'][soln[i]-1], instance['coordinates'][soln[i+1]-1])

    # print(cost)

    # if cost == feasible_obj:

    # check precedence
    soln_indices = sorted(range(len(soln)), key=lambda k: soln[k])
    for loc in soln:
        if instance['pickup'][loc-1] > 0: # this location is delivery and has a associated pickup location
            if soln_indices[loc-1] <= soln_indices[instance['pickup'][loc-1]]: # if tour visits delivery before pickup
                tour_valid = False
        if instance['delivery'][loc-1] > 0: # this location is pickup and has a associated delivery location
            if soln_indices[loc-1] >= soln_indices[instance['delivery'][loc-1]]: # if tour visits delivery before pickup
                tour_valid = False

    # check tw

    return precedence_check, tw_check