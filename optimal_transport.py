# -*- coding: utf-8 -*-
"""
***************** COPYRIGHT AND CONFIDENTIALITY INFORMATION *****************

Copyright © 2017 [Thomson Licensing] All Rights Reserved
This program contains proprietary information which is a trade secret/business 
secret of [Thomson Licensing] and is protected, even if unpublished, under 
applicable Copyright laws (including French droit d’auteur) and/or may be 
subject to one or more patent(s). 
Recipient is to retain this program in confidence and is not permitted to use 
or make copies thereof other than as permitted in a written agreement with 
[Thomson Licensing] unless otherwise expressly allowed by applicable laws or by
[Thomson Licensing] under express agreement.
Thomson Licensing is a company of the group TECHNICOLOR

*****************************************************************************
"""

import numpy as np


def build_moving_cost_matrix(palette1, palette2):
    """
    Cost matrix for optimal transport between palette1 and palette2
    """
    # Init.
    C = np.zeros((palette1.shape[0], palette2.shape[0]), dtype='float')
    # Compute distances
    for ind in range(palette1.shape[1]):
        x1 = palette1[:,ind]
        x2 = palette2[:,ind]
        x1.shape = (x1.size, 1)
        x2.shape = (1, x2.size)
        x1 = np.repeat(x1, x2.shape[1], 1)
        x2 = np.repeat(x2, x1.shape[0], 0)
        C += (x1 - x2)**2
    return np.sqrt(C)


def projection_sum_equal(X0, target_value):
    """
    Return the matrix X such that
    min_X || X - X0 || s.t. sum(X) = target_value
    """
    corr = (target_value - X0.sum())/X0.size
    Sol = np.array(X0) + corr
    return Sol


def projection_column_sum_in_range(X0, bounds):
    """
    Return the matrix X such that
    min_X || X - X0 ||  s.t. min(bounds) <= sum(X, 1) <= max(bounds)
    """
    # Get bounds
    alpha = np.min(bounds, 1)
    beta = np.max(bounds, 1)
    # Solution
    Sol = np.array(X0)
    # Compute the sum of the rows
    ref = X0.sum(1)
    # Find rows below alpha and adapt Sol
    loc = (ref < alpha)
    corr = (alpha[loc] - ref[loc])/Sol.shape[1]
    corr.shape = (corr.size, 1)
    corr = np.repeat(corr, Sol.shape[1], 1)
    Sol[loc,:] = Sol[loc,:] + corr
    # Find rows above beta and adapt Sol
    loc = (ref > beta)
    corr = (beta[loc] - ref[loc])/Sol.shape[1]
    corr.shape = (corr.size, 1)
    corr = np.repeat(corr, Sol.shape[1], 1)
    Sol[loc,:] = Sol[loc,:] + corr
    return Sol


def OT_ADMM(palette2Mod, paletteRef, eps=1e-4, miter=1e5, verbose=False):
    """
    Solve the optimal transport problem with ADMM
    """
    # Build cost matrix
    C = build_moving_cost_matrix(palette2Mod, paletteRef)
    C = C/C.max()

    # Constraints for transport
    bounds = np.zeros(2, dtype='object')
    size_pal = [palette2Mod.shape[0],paletteRef.shape[0]]
    for i in range(2):
        bounds[i] = np.array([[0, 1]]*size_pal[i])/float(size_pal[i])

    # Dual variables
    Lambda = np.array([np.zeros(C.shape)]*3)
    # Auxiliary variables
    Aux = np.array([np.zeros(C.shape)]*3)
    # Solution
    Sol = np.zeros(C.shape)
    Old = np.zeros(C.shape)

    # ADMM
    rho = 1e2
    iteration = 0
    while True:
        # Update all primal variables
        # Sol with projection onto positive set and spatial constraint
        Sol = (- C + rho*np.sum(Aux, 0) + np.sum(Lambda, 0))/(3*rho)
        Sol[Sol<0] = 0. # Positivity constraint
        for i in range(3):
            Aux[i] = Sol - Lambda[i]/rho
        # Projection onto column sum constraint
        Aux[0] = projection_column_sum_in_range(Aux[0], bounds[0])
        # Projection onto row sum constraint
        Aux[1] = projection_column_sum_in_range(Aux[1].T, bounds[1]).T
        # Projection onto total sum constraint
        Aux[2] = projection_sum_equal(Aux[2], 1.)
        # Update dual variables
        for i in range(3):
            Lambda[i] += rho*(Aux[i] - Sol)
        # Stopping criterion
        if verbose and iteration%100==0:
            obj = np.multiply(C, Sol).sum()
            print((iteration, obj,
                   np.linalg.norm(Sol-Old)/(np.linalg.norm(Sol)+1e-10),
                   np.linalg.norm(Sol-Aux[0])/(np.linalg.norm(Sol)+1e-10),
                   np.linalg.norm(Sol-Aux[1])/(np.linalg.norm(Sol)+1e-10),
                   np.linalg.norm(Sol-Aux[2])/(np.linalg.norm(Sol)+1e-10)))
        if iteration>miter:
            break
        elif np.linalg.norm(Sol-Old)<eps*np.linalg.norm(Sol) and \
                np.linalg.norm(Sol-Aux[0])<eps*np.linalg.norm(Sol) and \
                np.linalg.norm(Sol-Aux[1])<eps*np.linalg.norm(Sol) and \
                np.linalg.norm(Sol-Aux[2])<eps*np.linalg.norm(Sol):
            break
        else:
            Old[:,:] = Sol[:,:]
            iteration += 1

    return Sol


def transform_palette(palette_orig, palette_target, Transport):
    """
    Transform palette using the result of optimal transport
    """
    sum_gamma = Transport.sum(1)
    palette_new = np.dot(Transport, palette_target)
    for ind in range(palette_new.shape[1]):
        palette_new[:,ind] = np.divide(palette_new[:,ind], sum_gamma+1e-10)
    return palette_new


def compute_permutation(W1, W2):
    """
    Implemented for Quang's internship
    """

    # --- Compute optimal transport plan
    transport_plan = OT_ADMM(W1, W2)
    
    # --- Transform W2 to match W1
    W = transform_palette(W1, W2, transport_plan)
    
    return W


if __name__=='__main__':
    
    # --- Step 1: define matrix
    
    # Matrix W of, e.g., bass
    # CAREFUL: The dimensions are transposed compared to this morning explanations
    #   - 5 is the dimension in NMF
    #   - 128 is the number of features
    W_bass = np.random.rand(5, 128)
    
    # Matrix W of, e.g., flute
    #   - 10 is the dimension in NMF
    #   - 128 is the number of features
    W_flute = np.random.rand(10, 128)
    
    
    # --- Get resulting matrix transforming bass into flute
    # Result should be used in place of W_bass, e.g.
    #     You first obtain
    #           Phi_bass = W_bass^t  H_bass
    #           Phi_flute = W_flute^t  H_flute
    #
    #     Then you transform to bass in flute with
    #                   W_bass2flute^t H_bass
    W_bass2flute = compute_permutation(W_bass, W_flute)
    
    
    
    