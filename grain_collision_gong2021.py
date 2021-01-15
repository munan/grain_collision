import numpy as np
from scipy.integrate import quad

#TODO: I think one just need to specify k_eta/k_L, not seperately

class TurbulenceModel:
    """General turbulence model."""
    def __init__(self, Re, p, m):
        """Initialization of the class. 
        input:
            Re: Reynolds number (must be larger than 1) 
            p: kinetic energy power spectrum index E(k)~k^-p
               note: for convergence, we must have p > 1
            m: eddy autocorrelation time power spectrum index tau(k)~k^-m
        """
        self.Re = float(Re)
        self.k_L = 1.
        if Re > 1.:
            k_eta2k_L = Re**(2./(1.+p))
            self.k_eta = k_eta2k_L
        else:
            raise RuntimeError("TurbulenceModel: Re must be largen than k_L.")
        if p > 1.:
            self.p = float(p)
        else:
            raise RuntimeError("TurbulenceModel: p must be largen than 1.")
        if m > 0.:
            self.m = float(m)
        else:
            raise RuntimeError("TurbulenceModel: m must be largen than 0.")
        return

class KolmogorovTurbulence(TurbulenceModel):
    """Kolmogorov, inheritance from TurbulenceModel."""
    def __init__(self, Re):
        """Initialization of the class. 
        input:
            Re: Reynolds number (must be larger than 1) 
        """
        p = 5./3.
        m = 2./3.
        super().__init__(Re, p, m)
        return


class IKTurbulence(TurbulenceModel):
    """Iroshnikov-Kraichnan (IK), inheritance from TurbulenceModel."""
    def __init__(self, Re):
        """Initialization of the class. 
        input:
            Re: Reynolds number (must be larger than 1) 
        """
        p = 3./2.
        m = 3./4.
        super().__init__(Re, p, m)
        return

class MRITurbulence(TurbulenceModel):
    """MRI turbulence from Gong2020, inheritance from TurbulenceModel."""
    def __init__(self, Re):
        """Initialization of the class. 
        input:
            Re: Reynolds number (must be larger than 1) 
        """
        p = 4./3.
        m = 5./6.
        super().__init__(Re, p, m)
        return

class GrainCollision:
    """Grain collisional velocities from turbulence perturbation."""
    def __init__(self, turbulence_model):
        """Initialization of the class.
        input:
            turbulence_model: an instance of the class TurbulenceModel, which
            specifies the turbulence perturbation on the grain.
        """
        self.turbulence_model = turbulence_model
        self.k_L = turbulence_model.k_L
        self.k_eta = turbulence_model.k_eta
        self.p = turbulence_model.p
        self.m = turbulence_model.m
        self.vtot = np.sqrt(self.k_L / (self.p - 1.))
        return
    def get_Ek(self, k):
        """Assuming E0 = 1."""
        return (k/self.k_L)**(-self.p)
    def get_tauk2tau0(self, k):
        """tau(k)/tau_0 in the tubulence model"""
        return (k/self.k_L)**(-self.m)
    def get_vrel(self, st, k):
        vsq, _ = quad(lambda x: self.get_Ek(x)*(1./(1. +
            self.get_tauk2tau0(x)/st))**2, self.k_L, k, points=np.logspace(0, 10, 10))
        vrel = np.sqrt(vsq)
        return vrel
    def get_tau0(self):
        return 1./(self.k_L * np.sqrt(self.k_L))
    def get_kstar(self, st):
        """k* from apprixmiation of tau(k*)=tau_f."""
        kstar = self.k_L * st**(-1./self.m)
        return kstar
    def get_K(self, st, k):
        K = st / (st + self.get_tauk2tau0(k))
        return K
    def get_T1(self, st, kstar):
        """the class 1 term in v^2."""
        if kstar <= self.k_L:
            T1 = 0.
        else:
            k_p = min(self.k_eta, kstar)
            T1, _ = quad(lambda x: self.get_Ek(x)*(1.-self.get_K(st, x)**2),
                           self.k_L, k_p, points=np.logspace(0, 10, 10))
        return T1
    def get_T1_analytic(self, st, kstar):
        if kstar <= self.k_L:
            T1 = 0.
        else:
            k_p = min(self.k_eta, kstar)
            T1 = self.k_L/(self.p-1.) * (1. - (self.k_L/k_p)**(self.p-1)
                    ) - st**2 * self.k_L/(1. + 2.*self.m - self.p) * (
                            (k_p/self.k_L)**(1. + 2.*self.m - self.p) - 1.)
        return T1
    def get_T3(self, st, kstar):
        def fun3(k):
            tauk = self.get_tauk2tau0(k)*self.get_tau0()
            K = self.get_K(st, k)
            X = K*tauk*k*self.get_vrel(st, k)
            g = 1./X * np.arctan(X)
            h = 1./(1. + X**2)
            return self.get_Ek(k)*(1.-K)*(g+K*h)
        if kstar >= self.k_eta:
            T3 = 0;
        else:
            if kstar < self.k_L:
                kstar = self.k_L
            T3, _ = quad(fun3, kstar, self.k_eta, points=np.logspace(0, 10, 10))
        return T3;
    def get_T3_analytic(self, st, kstar):
        if kstar >= self.k_eta:
            T3 = 0;
        else:
            if kstar < self.k_L:
                kstar = self.k_L
            T3 = 2./(self.p+self.m-1.)/st * self.k_L * (
                    (self.k_L/kstar)**(self.p+self.m-1.) 
                    - (self.k_L/self.k_eta)**(self.p+self.m-1.))
        return T3;
    def get_vsq(self, st):
        """term <v^2>.
        normalized by total turbulent velocity."""
        kstar = self.get_kstar(st)
        T1 = self.get_T1(st, kstar)
        T3 = self.get_T3(st, kstar)
        vsq = T1 + T3
        return vsq/self.vtot**2
    def get_vsq_analytic(self, st):
        """term <v^2>, analytic approximation.
        normalized by total turbulent velocity."""
        kstar = self.get_kstar(st)
        T1 = self.get_T1_analytic(st, kstar)
        T3 = self.get_T3_analytic(st, kstar)
        vsq = T1 + T3
        return vsq/self.vtot**2
    def get_v1v2(self, st1, st2):
        """term <v1v2>
        normalized by total turbulent velocity."""
        st1 = float(st1)
        st2 = float(st2)
        if st1 < st2:
            tmp = st1
            st1 = st2
            st2 = tmp
        kstar = self.get_kstar(st1)
        v1v2 = st1/(st1+st2) * self.get_T1(st1, kstar
                ) + st2/(st1+st2) * self.get_T1(st2, kstar)
        return v1v2/self.vtot**2
    def get_v1v2_analytic(self, st1, st2):
        """term <v1v2>, analytic approximation.
        normalized by total turbulent velocity."""
        st1 = float(st1)
        st2 = float(st2)
        if st1 < st2:
            tmp = st1
            st1 = st2
            st2 = tmp
        kstar = self.get_kstar(st1)
        v1v2 = st1/(st1+st2) * self.get_T1_analytic(st1, kstar
                ) + st2/(st1+st2) * self.get_T1_analytic(st2, kstar)
        return v1v2/self.vtot**2
    def get_vcoll(self, st1, st2):
        """collisional velocity between two grains, numerical integration.
        input:
            st1: stokes number of grain 1
            st2: stokes number of grain 2
        return:
            vcoll: collisional velocity (normalized by total turbulent
            velocity)."""
        v1sq = self.get_vsq(st1)
        v2sq = self.get_vsq(st2)
        v1v2 = self.get_v1v2(st1, st2)
        vcoll = np.sqrt(v1sq + v2sq - 2.*v1v2)
        return vcoll
    def get_vcoll_analytic(self, st1, st2):
        """collisional velocity between two grains, analytic approximation.
        input:
            st1: stokes number of grain 1
            st2: stokes number of grain 2
        return:
            vcoll: collisional velocity (normalized by total turbulent
            velocity)."""
        v1sq = self.get_vsq_analytic(st1)
        v2sq = self.get_vsq_analytic(st2)
        v1v2 = self.get_v1v2_analytic(st1, st2)
        vcoll = np.sqrt(v1sq + v2sq - 2.*v1v2)
        return vcoll





