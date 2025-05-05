'''
    Solve the Green's functions of 2D BCS superconductor with KPM
    following https://link.aps.org/doi/10.1103/PhysRevLett.105.167006.
'''

import numpy as np
import scipy.sparse as sparse
from scipy.fft import fft

class kpmBdG:
    pass

def get_moments(v1, v2, m, n=100):
    '''
        general moments

            mu_i = <v2|T_i(M)|v1>
        
        for 0 <= i < n. The input vector v should be a bra (row vector).
    '''
    mus = np.zeros(n, dtype=np.complex128)

    # all involved vectors are row vectors
    am = v1.copy()
    a = m.dot(v1)
    mu0 = np.vdot(v2, v1)
    mu1 = np.vdot(v2, a)
    mus[0] = mu0
    mus[1] = mu1

    for i in range(2,n):
        ap = 2*m.dot(a) - am
        mus[i] = np.vdot(v2, ap)
        am = a.copy()
        a = ap.copy()
    
    return mus

def get_moments_gfbcs(m, i, j, n, type='11'):
    '''
        generate moments for Green's function <j|G^ab|i>
    '''
    if type not in ('11', '12', '21', '22'): raise
    if m.shape[0] != m.shape[1]: raise
    N = m.shape[0]//2

    vi = np.zeros(2*N)
    vj = np.zeros(2*N)
    if type == '11':
        vi[i] = 1
        vj[j] = 1
    elif type == '22':
        vi[i+N] = 1
        vj[j+N] = 1
    elif type == '12':
        vi[i+N] = 1
        vj[j] = 1
    elif type == '21':
        vi[i] = 1
        vj[j+N] = 1
    
    mus = get_moments(vi, vj, m, n)
    return mus

def jackson_kernel(mus):
    n = len(mus)    
    pn = np.pi/(n+1.)
    mo = mus * np.array([((n-i+1)*np.cos(pn*i)+np.sin(pn*i)/np.tan(pn))/(n+1) for i in range(n)])
    return mo

def lorentz_kernel(mus, lamb=.1):
    n = len(mus)
    mo = mus * np.array([np.sinh(lamb*(1.-i/n))/np.sinh(lamb) for i in range(n)])
    return mo

def generate_profile_gfbcs(mus, nx, kernel='jackson', lamb=.1):
    '''
        Construct profile of the Green's function with FFT.
        Usually we consider nx > n, e.g. nx = 2n, to ensure all moments are used.
    '''
    if kernel == 'jackson': mus = jackson_kernel(mus)
    elif kernel == 'lorentz': mus = lorentz_kernel(mus, lamb)
    else: raise

    phis = np.pi/nx * (np.array(range(0,nx)) + .5)
    xs = np.cos(phis)

    # FFT
    ntilde = max(len(mus),nx) # half of the dimension of fourier transform
    if ntilde != nx: xs = xs[:ntilde]

    target = [(2-int(i==0)) * mus[i] * np.exp(-1j*np.pi*i/(2*ntilde)) if i < len(mus) else 0. for i in range(2*ntilde)]
    ys = fft(target, n=2*ntilde)[:ntilde]
    ys *= -1j / np.sqrt(1.-xs**2)
    return xs, ys


if __name__ == '__main__':

    L = 512
    N = L**2
    t = 1.
    delta = 1.

    row_indices = []
    col_indices = []
    data = []

    for i in range(N):
        x, y = i%L, i//L
        xp1, yp1 = ((x+1)%L) + y*L, x + ((y+1)%L)*L

        # particle and hole sector
        for s in (0,1):
            row_indices.append(i+s*N)
            col_indices.append(xp1+s*N)
            data.append(-(1-2*s)*t)
            row_indices.append(xp1+s*N)
            col_indices.append(i+s*N)
            data.append(-(1-2*s)*t)

            row_indices.append(i+s*N)
            col_indices.append(yp1+s*N)
            data.append(-(1-2*s)*t)
            row_indices.append(yp1+s*N)
            col_indices.append(i+s*N)
            data.append(-(1-2*s)*t)

        row_indices.append(i)
        col_indices.append(i+N)
        data.append(delta)
        row_indices.append(i+N)
        col_indices.append(i)
        data.append(np.conj(delta))

    sH = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(2*N, 2*N), dtype=np.complex128).tocsr()
    sH.eliminate_zeros()
   
    scale = 4*t * 1.05
    mus = get_moments_gfbcs(sH/scale, i=0, j=0, n=500, type='11')
    xs, ys = generate_profile_gfbcs(mus, nx=1000, kernel='jackson')

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(3,3))
    ax.plot(xs*scale, -2*np.imag(ys)/scale)

    # exact local density of states of s-wave BCS
    import vegas

    class cal_ldos_bcs:
        def __init__(self, freq, t, delta, eta):
            self.freq = freq
            self.t = t
            self.delta = delta
            self.eta = eta
            self.dim = 2

        def integrand(self, vars):
            x1, x2 = vars
            kx = np.pi * x1
            ky = np.pi * x2
            jacobian_kx = np.pi
            jacobian_ky = np.pi

            dispersion = - 2*self.t * (np.cos(kx) + np.cos(ky))
            gf = 1 / (self.freq+1j*self.eta - dispersion - delta*np.conj(delta)/(self.freq+1j*self.eta + dispersion))
            return -2*np.imag(gf) * jacobian_kx * jacobian_ky * 4 / (2*np.pi)**2

    freqs = np.linspace(-4*t, 4*t, 100, endpoint=True)
    freqs *= 1.05
    ldos = []
    for freq in freqs:
        f = cal_ldos_bcs(freq, t, delta, eta=1e-2)
        integ = vegas.Integrator([[0,1]]*f.dim, nproc=1, mpi=True)
        f_call = lambda vars: f.integrand(vars)
        _ = integ(f_call, nitn=20, neval=1000) # training
        result = integ(f_call, nitn=20, neval=1000)
        ldos.append((result.mean, result.sdev))
    ax.errorbar(freqs, *zip(*ldos), linestyle='dashed')

    # # diagonalization
    # xs = np.linspace(-4*t, 4*t, 50, endpoint=True)
    # H = sH.todense()
    # gfs = [np.linalg.inv((x+1j*1e-1)*np.eye(2*N) - H)[0,0] for x in xs]
    # ys = -2*np.imag(gfs)
    # ax.plot(xs, ys)

    plt.show()