# distance functions

import math

def Distance_1(locA, locB):
    return 1

def Distance_LARGE(locA, locB):
    return 10000000


# def Distance_Asymmetric(locA, locB)
#     n = DimensionSaved
#     if ((Na->Id <= n) == (Nb->Id <= n))
#         return M
#     if (abs(Na->Id - Nb->Id) == n)
#         return 0
#     return Na->Id <= n ? OldDistance(Na, Nb - n) : OldDistance(Nb, Na - n)


# def Distance_ATSP(Node * Na, Node * Nb)

#     def n = DimensionSaved
#     if ((Na->Id <= n) == (Nb->Id <= n))
#         return M
#     if (abs(Na->Id - Nb->Id) == n)
#         return 0
#     return Na->Id <= n ? Na->C[Nb->Id - n] : Nb->C[Na->Id - n]


# def Distance_ATT(Node * Na, Node * Nb)

#     double xd = Na->X - Nb->X, yd = Na->Y - Nb->Y
#     return (def) ceil(Scale * (sqrt((xd * xd + yd * yd) / 10.0)))


# def Distance_CEIL_2D(Node * Na, Node * Nb)

#     double xd = Na->X - Nb->X, yd = Na->Y - Nb->Y
#     return (def) ceil(Scale * sqrt(xd * xd + yd * yd))


# def Distance_CEIL_3D(Node * Na, Node * Nb)

#     double xd = Na->X - Nb->X, yd = Na->Y - Nb->Y, zd = Na->Z - Nb->Z
#     return (def) ceil(Scale * sqrt(xd * xd + yd * yd + zd * zd))


def Distance_EUC_2D(locA, locB):
    xd = locA[0] - locB[0]
    yd = locA[1] - locB[1]
    return int(math.sqrt(xd * xd + yd * yd))
