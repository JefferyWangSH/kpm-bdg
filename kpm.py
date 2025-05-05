'''
    Solve the Green's functions of 2D BCS superconductor with KPM
    following https://link.aps.org/doi/10.1103/PhysRevLett.105.167006.
'''

import numpy as np
from scipy.fft import fft

class kpm:
    def __init__(self):
        pass
    
    @staticmethod
    def _get_moments(v1, v2, m, n=100):
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
    
    @staticmethod
    def _get_moments_diagonal(v, m, n=100):
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

    @staticmethod
    def _jackson_kernel(mus):
        n = len(mus)    
        pn = np.pi/(n+1.)
        mo = mus * np.array([((n-i+1)*np.cos(pn*i)+np.sin(pn*i)/np.tan(pn))/(n+1) for i in range(n)])
        return mo

    @staticmethod
    def _lorentz_kernel(mus, lamb=.1):
        n = len(mus)
        mo = mus * np.array([np.sinh(lamb*(1.-i/n))/np.sinh(lamb) for i in range(n)])
        return mo

    @staticmethod
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
            mus += kpm._get_moments_diagonal(v, m, n)
        return mus/nvec

    @staticmethod
    def correlator(m, i, j, n):
        '''
            generate moments for Green's function <j|G^ab|i>
        '''
        if m.shape[0] != m.shape[1]: raise
        N = m.shape[0]
        if i < 0 or i >= N: raise
        if j < 0 or j >= N: raise

        vi = np.zeros(N)
        vj = np.zeros(N)
        vi[i] = 1
        vj[j] = 1

        mus = kpm._get_moments(vi, vj, m, n)
        return mus

    @staticmethod
    def generate_profile(mus, nx, kernel='jackson', lamb=.1):
        '''
            Construct profile of the Green's function with FFT.
            Usually we consider nx > n, e.g. nx = 2n, to ensure all moments are used.
        '''
        if kernel == 'jackson': mus = kpm._jackson_kernel(mus)
        elif kernel == 'lorentz': mus = kpm._lorentz_kernel(mus, lamb)
        else: raise

        phis = np.pi/nx * (np.array(range(0,nx)) + .5)
        xs = np.cos(phis)

        # FFT
        ntilde = max(len(mus),nx) # half of the dimension of fourier transformation
        if ntilde != nx: xs = xs[:ntilde]

        target = [(2-int(i==0)) * mus[i] * np.exp(-1j*np.pi*i/(2*ntilde)) if i < len(mus) else 0. for i in range(2*ntilde)]
        ys = fft(target, n=2*ntilde)[:ntilde]
        ys *= -1j / np.sqrt(1.-xs**2)
        return xs, ys

class kpmBdG(kpm):
    @staticmethod
    def correlator(m, i, j, n, type='11'):
        '''
            generate moments for Green's function <j|G^ab|i>
        '''
        if m.shape[0] != m.shape[1]: raise
        if m.shape[0] % 2 != 0: raise
        N = m.shape[0]//2
        if i < 0 or i >= N: raise
        if j < 0 or j >= N: raise

        if type == '22':
            i += N
            j += N
        elif type == '12':
            i += N
        elif type == '21':
            j += N
        return super(kpmBdG, kpmBdG).correlator(m, i, j, n)


if __name__ == '__main__':
    pass