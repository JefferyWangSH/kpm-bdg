'''
    Solve total density of states of 2D tight-binding model with KPM.
    Adapted from the kpmpy library (https://github.com/joselado/kpmpy) by joselado.
'''

import numpy as np
from scipy import sparse

def get_moments(v, m, n=100):
    '''
        (diagonal) moments

            mu_i = <v|T_i(M)|v>
        
        for 0 <= i < 2n. The input vector v should be a bra (row vector).
        And this function is specialized for diagonal moment and hermitian matrix m.
    '''
    mus = np.zeros(2*n, dtype=np.complex128)

    # all involved vectors are row vectors
    am = v.copy()
    a = m.dot(v)
    mu0 = np.vdot(v, v)
    mu1 = np.vdot(v, a)
    mus[0] = mu0
    mus[1] = mu1

    for i in range(1,n):
        ap = 2*m.dot(a) - am
        mus[2*i] = 2 * np.vdot(a, a) - mu0
        mus[2*i+1] = 2 * np.vdot(ap, a) - mu1
        am = a.copy()
        a = ap.copy()
    
    return mus

def random_trace(m, nvec, n):
    '''
        Stochastic estimation of moments

            mu_i = Tr[ T_i(M) ] ~ 1/N_vec sum_{v=1}^{N_vec} <v|T_i(M)|v>

        for 0 <= i < 2n. The relative error is of order O(1/sqrt(N_vec*D)) with D the dimension of the problem, e.g. the size of M.
    '''
    if m.shape[0] != m.shape[1]: raise
    
    mus = np.zeros(2*n, dtype=np.complex128)
    for _ in range(nvec):
        v = np.random.random(m.shape[0])-.5 + 1j*(np.random.random(m.shape[0])-.5)
        v /= np.sqrt(np.vdot(v, v))
        mus += get_moments(v, m, n)
    return mus/nvec

def jackson_kernel(mus):
    n = len(mus)    
    pn = np.pi/(n+1.)
    mo = mus * np.array([((n-i+1)*np.cos(pn*i)+np.sin(pn*i)/np.tan(pn))/(n+1) for i in range(n)])
    return mo

def lorentz_kernel(mus, lamb=.1):
    n = len(mus)
    mo = mus * np.array([np.sinh(lamb*(1.-i/n))/np.sinh(lamb) for i in range(n)])
    return mo

def generate_profile(mus, xs, kernel='jackson'):
    '''
        Generate profile of the target function
    '''
    if kernel == 'jackson': mus = jackson_kernel(mus)
    elif kernel == 'lorentz': mus = lorentz_kernel(mus)
    else: raise

    tm = np.ones(xs.shape)
    t = xs.copy()    
    ys = np.full(xs.shape, mus[0])

    for i in range(1,len(mus)):
        mu = mus[i]
        ys += 2.*mu*t
        tp = 2.*xs*t - tm # chebychev recursion relation
        tm = t.copy()
        t = tp.copy()
    ys /= np.pi * np.sqrt(1.-xs*xs)
    return ys


if __name__ == '__main__':
    
    L = 512
    N = L**2
    t = 1.

    row_indices = []
    col_indices = []
    data = []

    for i in range(N):
        x, y = i%L, i//L
        xp1, yp1 = ((x+1)%L) + y*L, x + ((y+1)%L)*L

        row_indices.append(i)
        col_indices.append(xp1)
        data.append(-t)
        row_indices.append(xp1)
        col_indices.append(i)
        data.append(-t)

        row_indices.append(i)
        col_indices.append(yp1)
        data.append(-t)
        row_indices.append(yp1)
        col_indices.append(i)
        data.append(-t)

    sH = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(N, N), dtype=np.complex128).tocsr()
    sH.eliminate_zeros()
   
    scale = 4*t * 1.05
    mus = random_trace(sH/scale, nvec=100, n=100)

    freqs = np.linspace(-4*t, 4*t, 1000, endpoint=True)
    xs = freqs/scale
    ys = np.real(generate_profile(mus, xs))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(3,3))
    ax.plot(freqs, ys/scale)

    # exact density of states
    import scipy as sp
    dos = sp.special.ellipk(1-(freqs/(4*t))**2) / (2*np.pi**2*t)
    ax.plot(freqs, dos, linestyle='dashed')

    print(np.sum(ys)*(freqs[1]-freqs[0])/scale) # sum rule
    print(np.sum(dos)*(freqs[1]-freqs[0]))

    plt.show()
