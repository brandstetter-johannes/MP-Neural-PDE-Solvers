import numpy as np

###########################################################################
# beta, gamma, and stencils needed for WENO5 scheme
betaA_3 = np.sqrt(13 / 12) * np.array([[[1, -2, 1, 0, 0]],
                                             [[0, 1, -2, 1, 0]],
                                             [[0, 0, 1, -2, 1]]])

betaB_3 = (1 / 2) * np.array([[[1, -4, 3, 0, 0]],
                                    [[0, 1, 0, -1, 0]],
                                    [[0, 0, 3, -4, 1]]])

gamma_3 = np.array([[[1],[6],[3]]]) / 10

stencils_3 = (1 / 6) * np.array([[[2,  -7,  11,  0,  0]],
                                    [[ 0,  -1,  5, 2,  0]],
                                    [[ 0,  0,  2, 5,  -1]]])


betaA_all = {
          3 : betaA_3
        }

betaB_all = {
          3 : betaB_3
        }

gamma_all = {
          3 : gamma_3
        }

stencils_all =  {
          3 : stencils_3
        }
###########################################################################
# Coefficients of central differences for 4th order accuracy for first and second derivative and
# 2nd order accuracy for third and fourth derivative (for simplicity)
# Coefficients taken from: https://en.wikipedia.org/wiki/Finite_difference_coefficient
derivative_1 = np.array([[[1/12, -2/3, 0, 2/3, -1/12]]])
derivative_2 = np.array([[[-1/12, 4/3, -5/2, 4/3, -1/12]]])
derivative_3 = np.array([[[-1/2, 1, 0, -1, 1/2]]])
#derivative_3 = np.array([[[-7/240, 3/10, -169/120, 61/30, 0, -61/30, 169/120, -3/10, 7/240]]])
derivative_4 = np.array([[[1., -4., 6., -4., 1.]]])

FDM_derivatives = {
        1 : derivative_1,
        2 : derivative_2,
        3 : derivative_3,
        4:  derivative_4
}