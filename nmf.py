from numpy import *
from numpy.random import rand
from numpy.linalg import norm
from matplotlib import pyplot as plt
import scipy.io.wavfile
import librosa
from scipy.signal import resample, correlate, correlate2d
import sys


def nmf(V, W=None, K=20, b=2, update=True, plot=False, U=None, iter=50):
    '''Non negative Matrix Factorization
    Input:
        V : FxN matrix of the embedding
        W : basis matrix could be learned prior
        K : Number of basis
        b : 0 for IS; 1 for KL; 2 for EUC
        update: Boolean whether or not to update W each iteration
        U : Weighting matrix
        iter: number of iterations for learning
        plot: Boolean for plotting the cost function vs iter number
    Output:
        W : FxK basis matrix
        H : KxN coefficients matrix
    '''
    # V = power(absolute(X),2) + 1e-80  # F-by-N power spectrogram, X=stft(x)
    (F, N) = V.shape
    if W == None or W == []:
        W = 0.5 * (0.3 + 1.7 * rand(F, K)) * sqrt(V.mean())
    else:
        K = W.shape[1]

    if U == None:
        U = ones((F, N))

    const = 1e-30  # very small value

    # initialization of W, H
    H = 0.5 * (0.3 + 1.7 * rand(K, N)) * sqrt(V.mean())

    div = zeros((iter, 1))  # cost function
    for i in range(iter):
        print(' nmf iter : {}'.format(i))
        hat_V = maximum(const * ones((F, N)), dot(W, H))  # avoid 0^(-2)=Inf in W and H updates
        # update H
        # H = multiply(H,  (dot(W.T,(multiply(power(hat_V,(b-2)),V)))) / (dot(W.T,(power(hat_V,(b-1)))) ) )
        H = multiply(H, (dot(W.T, (multiply(power(multiply(U, hat_V), (b - 2)), V)))) / (
            dot(W.T, (power(multiply(U, hat_V), (b - 1))))))

        # update hat_V
        # hat_V = hat_V + dot(W - W_old, H);
        hat_V = maximum(const * ones((F, N)), dot(W, H));  # avoid 0^(-2)=Inf

        # update W
        if update:
            # W = W = multiply(W, (dot(multiply(power(hat_V,(b-2)) ,V),H.T) / (dot((power(hat_V,(b-1))),(H.T)) ) ))
            W = multiply(W, (dot(multiply(power(multiply(U, hat_V), (b - 2)), V), H.T)) / (
                dot((power(multiply(U, hat_V), (b - 1))), (H.T))))

            # update hat_V   
            # hat_V = hat_V + dot(W - W_old, H);
            hat_V = maximum(const * ones((F, N)), dot(W, H))  # avoid 0^(-2)=Inf

        # normalize columns of W to unit norm
        tmps = multiply(W, W)
        scale = sqrt(tmps.sum(axis=0))

        d = diag((1 / scale))
        W = dot(W, d)

        # adjust H accordingly since V=WH
        d = diag(scale)
        H = dot(d, H)

        # calculate cost function
        tmp = zeros(V.shape)
        if b == 0:  # IS divergence
            tmp = (V / hat_V) - log(V / hat_V) - 1
        elif b == 1:  # KL divergence
            tmp = multiply(V, log(V) - log(hat_V)) + (hat_V - V)
        else:  # b=2 for Euc
            tmp = (1.0 / (b * (b - 1))) * (
                        power(V, b) + (b - 1) * power(hat_V, b) - b * multiply(V, power(hat_V, b - 1)))

        div[i] = tmp.sum()

    if plot:
        plt.figure()
        plt.plot(range(iter), div)
        plt.xlabel('iteration')
        plt.ylabel('IS')
        plt.title('NMF ' + str(K) + ' components')
        plt.draw()
        plt.show()

    return W, H


def normalize(x):
    '''Normalize input signal to the range [-1,1]'''
    return scale(x, -1.0, 1.0)


def scale(x, a, b):
    '''Scale the input signal x to the range [a,b]'''
    x = float64(x)
    xa = x.min()
    xb = x.max()
    y = a + ((b - a) * 1.0 / (xb - xa)) * (x - xa)
    return y


def reconstruct(Vj, V, X):
    ''' Reconstruct time domain signal from given power spectrogram V '''
    X_rec = multiply((Vj * 1.0 / V), X)  # Wiener Filter
    x = istft(X_rec)
    return x
