"""Functions for exclusive $B_s\to V\ell^+\ell^-$ decays, taking into account
the finite life-time difference between the $B_s$ mass eigenstates,
see arXiv:1502.05509."""

import flavio
from . import observables
from flavio.classes import Observable, Prediction
import cmath
import warnings
from .. import angular
from math import sqrt

### Definite time integrals of the different parts composing the time dependence
def definite_int_cosh(y, Gamma, tmin, tmax):
    den =  (y**2 - 1) # Gamma *
    num_tmax = cmath.exp(-tmax*Gamma) * (y* cmath.sinh(tmax*Gamma*y) + cmath.cosh(tmax*Gamma*y) )
    num_tmin = cmath.exp(-tmin*Gamma) * (y* cmath.sinh(tmin*Gamma*y) + cmath.cosh(tmin*Gamma*y) )
    return (num_tmax - num_tmin)/den

def definite_int_sinh(y, Gamma, tmin, tmax):
    den =  (y**2 - 1) # Gamma *
    num_tmax = cmath.exp(-tmax*Gamma) * (cmath.sinh(tmax*Gamma*y) + y * cmath.cosh(tmax*Gamma*y) )
    num_tmin = cmath.exp(-tmin*Gamma) * (cmath.sinh(tmin*Gamma*y) + y * cmath.cosh(tmin*Gamma*y) )
    return (num_tmax - num_tmin)/den

def definite_int_cos(x, Gamma, tmin, tmax):
    den =  (x**2 + 1) # Gamma *
    num_tmin = cmath.exp(-tmin*Gamma) * (cmath.cos(tmin*Gamma*x) - x * cmath.sin(tmin*Gamma*x) )
    num_tmax = cmath.exp(-tmax*Gamma) * (cmath.cos(tmax*Gamma*x) - x * cmath.sin(tmax*Gamma*x) )
    return (num_tmin - num_tmax)/den

def definite_int_sin(x, Gamma, tmin, tmax):
    den =  (x**2 + 1) # Gamma *
    num_tmin = cmath.exp(-tmin*Gamma) * (cmath.sin(tmin*Gamma*x) + x * cmath.cos(tmin*Gamma*x) )
    num_tmax = cmath.exp(-tmax*Gamma) * (cmath.sin(tmax*Gamma*x) + x * cmath.cos(tmax*Gamma*x) )
    return (num_tmin - num_tmax)/den
    
def bsvll_obs(function, q2, wc_obj, par, B, V, lep):
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mV = par['m_'+V]
    y = par['DeltaGamma/Gamma_'+B]/2.
    x = par['DeltaM/Gamma_'+B]
    gamma = 0.6597 # only for Bs TODO: remove
    if q2 < 4*ml**2 or q2 > (mB-mV)**2:
        return 0
    
    scale = flavio.config['renormalization scale']['bvll']
    mb = flavio.physics.running.running.get_mb(par, scale)
    ff = flavio.physics.bdecays.bvll.amplitudes.get_ff(q2, par, B, V)
    h = flavio.physics.bdecays.bvll.amplitudes.helicity_amps(q2, ff, wc_obj, par, B, V, lep)
    h_bar = flavio.physics.bdecays.bvll.amplitudes.helicity_amps_bar(q2, ff, wc_obj, par, B, V, lep)
    J = angular.angularcoeffs_general_v(h, q2, mB, mV, mb, 0, ml, ml)
    J_bar = angular.angularcoeffs_general_v(h_bar, q2, mB, mV, mb, 0, ml, ml)
    h_tilde = h_bar.copy()
    h_tilde[('pl', 'V')] = h_bar[('mi', 'V')]
    h_tilde[('pl', 'A')] = h_bar[('mi', 'A')]
    h_tilde[('mi', 'V')] = h_bar[('pl', 'V')]
    h_tilde[('mi', 'A')] = h_bar[('pl', 'A')]
    h_tilde['S'] = -h_bar['S']
    q_over_p = flavio.physics.mesonmixing.observables.q_over_p(wc_obj, par, B)
    phi = cmath.phase(-q_over_p) # the phase of -q/p
    J_h = angular.angularcoeffs_h_v(phi, h, h_tilde, q2, mB, mV, mb, 0, ml, ml)
    J_s = angular.angularcoeffs_s_v(phi, h, h_tilde, q2, mB, mV, mb, 0, ml, ml)
    return function(y, x, gamma, J, J_bar, J_h, J_s)

def bsvll_obs_trans(function, q2, wc_obj, par, B_meson, V_meson, lepton): 
    ml = par['m_'+lepton]
    mB = par['m_'+B_meson]
    mV = par['m_'+V_meson]
    y = par['DeltaGamma/Gamma_'+B_meson]/2.
    x = par['DeltaM/Gamma_'+B_meson]
    gamma = 0.6597 # only for Bs TODO: remove
    if q2 < 4*ml**2 or q2 > (mB-mV)**2:
        return 0
    
    scale = flavio.config['renormalization scale']['bvll']
    mb = flavio.physics.running.running.get_mb(par, scale)
    ff = flavio.physics.bdecays.bvll.amplitudes.get_ff(q2, par, B_meson, V_meson)
    A = flavio.physics.bdecays.bvll.amplitudes.transversity_amps(q2, ff, wc_obj, par, B_meson, V_meson, lepton)
    A_bar = flavio.physics.bdecays.bvll.amplitudes.transversity_amps_bar(q2, ff, wc_obj, par, B_meson, V_meson, lepton)
    # h = flavio.physics.bdecays.bvll.amplitudes.helicity_amps(q2, ff, wc_obj, par, B_meson, V_meson, lepton)
    # h_bar = flavio.physics.bdecays.bvll.amplitudes.helicity_amps_bar(q2, ff, wc_obj, par, B_meson, V_meson, lepton)
    # A = angular.helicity_to_transversity(h)
    # A_bar = angular.helicity_to_transversity(h_bar)
    J = angular.angularcoeffs_general_transversity(A, q2, ml)
    J_bar = angular.angularcoeffs_general_transversity(A_bar, q2, ml)
    A_tilde = A_bar.copy()
    A_tilde['perp_L'] = -1 * A_bar['perp_L']
    A_tilde['perp_R'] = -1 * A_bar['perp_R']
    A_tilde['S'] = -1 * A_bar['S']  # Table 3 of https://arxiv.org/pdf/1502.05509
    q_over_p = flavio.physics.mesonmixing.observables.q_over_p(wc_obj, par, B_meson)
    J_h: dict[str | int, float] = angular.angularcoeffs_h_transversity(A, A_tilde, q2, ml, q_over_p)
    J_s: dict[str | int, float] = angular.angularcoeffs_s_transversity(A, A_tilde, q2, ml, q_over_p)
    return function(y, x, gamma, J, J_bar, J_h, J_s)

def bsvll_obs_int_t(function, tmin, tmax, q2, wc_obj, par, B, V, lep):
    """
    As above but allows to choose a range for tmin-tmax
    """
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mV = par['m_'+V]
    y = par['DeltaGamma/Gamma_'+B]/2.
    x = par['DeltaM/Gamma_'+B]
    gamma = 0.6597 # only for Bs TODO: remove
    if q2 < 4*ml**2 or q2 > (mB-mV)**2:
        return 0
    
    scale = flavio.config['renormalization scale']['bvll']
    mb = flavio.physics.running.running.get_mb(par, scale)
    ff = flavio.physics.bdecays.bvll.amplitudes.get_ff(q2, par, B, V)
    h = flavio.physics.bdecays.bvll.amplitudes.helicity_amps(q2, ff, wc_obj, par, B, V, lep)
    h_bar = flavio.physics.bdecays.bvll.amplitudes.helicity_amps_bar(q2, ff, wc_obj, par, B, V, lep)
    J = angular.angularcoeffs_general_v(h, q2, mB, mV, mb, 0, ml, ml)
    J_bar = angular.angularcoeffs_general_v(h_bar, q2, mB, mV, mb, 0, ml, ml)
    h_tilde = h_bar.copy()
    h_tilde[('pl', 'V')] = h_bar[('mi', 'V')]
    h_tilde[('pl', 'A')] = h_bar[('mi', 'A')]
    h_tilde[('mi', 'V')] = h_bar[('pl', 'V')]
    h_tilde[('mi', 'A')] = h_bar[('pl', 'A')]
    h_tilde['S'] = -h_bar['S']
    q_over_p = flavio.physics.mesonmixing.observables.q_over_p(wc_obj, par, B)
    phi = cmath.phase(-q_over_p) # the phase of -q/p
    J_h = angular.angularcoeffs_h_v(phi, h, h_tilde, q2, mB, mV, mb, 0, ml, ml)
    J_s = angular.angularcoeffs_s_v(phi, h, h_tilde, q2, mB, mV, mb, 0, ml, ml)
    return function(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s)

def S_original_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    # (42) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    return 1/(1-y**2) * (J[i] + J_bar[i]) - y/(1-y**2) * J_h[i]

def S_original_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -S_original_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)
    return S_original_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)


def S_original_experiment_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    r"""CP-averaged angular observable $S_i$ in the LHCb convention.

    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return S_original_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)/SA_den_Bs(y, x, gamma, J, J_bar, J_h, J_s)

def S_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    # (42) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    if i in [5, '6s', '6c', 8, 9]:
        return 1/(1+x**2) * ((J[i] + J_bar[i]) - x * J_s[i])       
    else:
        return 1/(1-y**2) * ((J[i] + J_bar[i]) - y * J_h[i])

def A_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    # (42) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    if i in [5, '6s', '6c', 8, 9]:
        return 1/(1-y**2) * ((J[i] - J_bar[i]) - y * J_h[i])
    else:
        return 1/(1+x**2) * ((J[i] - J_bar[i]) - x * J_s[i])
    
def S_theory_num_int_t_Bs(tmin, tmax, y, x, Gamma, J, J_bar, J_h, J_s, i):
    # (42) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    if i in [5, '6s', '6c', 8, 9]:
        return ((J[i] + J_bar[i]) * definite_int_cos(x, Gamma, tmin, tmax) 
                - J_s[i] * definite_int_sin(x, Gamma, tmin, tmax))       
    else:
        return ((J[i] + J_bar[i]) * definite_int_cosh(y, Gamma, tmin, tmax) 
                - J_h[i] * definite_int_sinh(y, Gamma, tmin, tmax))

def A_theory_num_int_t_Bs(tmin, tmax, y, x, Gamma, J, J_bar, J_h, J_s, i):
    # (42) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    if i in [5, '6s', '6c', 8, 9]:
        return ((J[i] - J_bar[i]) * definite_int_cosh(y, Gamma, tmin, tmax) 
                - J_h[i] * definite_int_sinh(y, Gamma, tmin, tmax)
                )
    else:
        return ((J[i] - J_bar[i]) * definite_int_cos(x, Gamma, tmin, tmax) 
                - J_s[i] * definite_int_sin(x, Gamma, tmin, tmax)
                )

def K_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    """ Time independent CP averaged angular observable in the theory convention. 
    Only measurable via a time-dependent angular analysis. 
    The coefficients $5$, $6s$, $6c$, $8$, and $9$ require flavour-tagging to be measured. 
    Explanation of the factor $1 / (1 - y^2)$: 
    In the flavio SA_den_Bs function, which acts as the normalisation for the observables here, 
    the integrated branching fraction in a given q^2 region is given by: 
    SA_den_Bs = (1/(1-y**2) * (observables.dGdq2(J) + observables.dGdq2(J_bar))
                - y/(1-y**2) * observables.dGdq2(J_h))/2
    whereas the normalisation in the LHCb convention is given by: 
    I = (observables.dGdq2(J) + observables.dGdq2(J_bar)
         - y * observables.dGdq2(J_h))
    As the factor $1 / (1 - y^2)$ is constant, we can cancel it out here in this function. 
    """
    return 1/(1 - y*y) * (J[i] + J_bar[i])

def W_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    """ 
    Time independent CP asymmetric angular observable in the theory convention,
    only measurable via a time-dependent angular analysis. 
    The coefficients $1s$, $1c$, $2s$, $2c$, $3$, $4$, and $7$ require flavour-tagging to be measured. 
    see K_theory_num_Bs for an explanation of the factor $1 / (1 - y^2)$ 
    """
    return 1/(1 - y*y) * (J[i] - J_bar[i])

def H_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    """ 
    angular observable in the theory convention, 
    only measurable via a time-dependent angular analysis. 
    see K_theory_num_Bs for an explanation of the factor $1 / (1 - y^2)$ 
    """
    return 1/(1 - y*y) * J_h[i]

def Z_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    """ 
    angular observable in the theory convention, 
    only measurable via a flavour-tagged and time-dependent angular analysis.
    see K_theory_num_Bs for an explanation of the factor $1 / (1 - y^2)$ 
    """
    return 1/(1 - y*y) * J_s[i]

# def Q8mi_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s):
#     r"""CP-averaged angular observable $Q_8^{\mathrm{mi}}$ in the LHCb convention.
#     See eq. (56) of arXiv:1502.05509
#     """
#     flavio.citations.register("Descotes-Genon:2015hea")
#     return J_s[8] / ( -2 * ( J['2c'] + J_bar['2c'] ) * ( 2 * (J['2s'] + J_bar['2s']) - (J[3] + J_bar[3]) ) )**0.5

# def Q9_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s):
#     r"""CP-averaged angular observable $Q_8^{\mathrm{mi}}$ in the LHCb convention.
#     See eq. (57) of arXiv:1502.05509
#     """
#     flavio.citations.register("Descotes-Genon:2015hea")
#     return J_s[9] / ( 2 * ( J['2s'] + J_bar['2s'] ) )

def M_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    """ proposed optimised angular observables M_i """
    assert i in ['1s', '1c', '2s', '2c', 3, 4, 5, '6s', 7, 8, 9], f"{i} not implemented!"
    return J_h[i]

def M_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    """ proposed optimised angular observables M_i """
    assert i in ['1s', '1c', '2s', '2c', 3, 4, 5, '6s', 7, 8, 9], f"{i} not implemented!"
    norm = {
        '1s': (J['2s'] + J_bar['2s']),
        '1c': -(J['2c'] + J_bar['2c']),
        '2s': (J['2s'] + J_bar['2s']),
        '2c': -(J['2c'] + J_bar['2c']),
        3: 2 * (J['2s'] + J_bar['2s']),
        4: -(- (J['2c'] + J_bar['2c']) * (2 * (J['2s'] + J_bar['2s']) - (J[3] + J_bar[3])) )**0.5,
        5: ((J['2c'] + J_bar['2c']) * ((J[3] + J_bar[3]) - 2 * (J['2s'] + J_bar['2s'])) )**0.5,
        # '6s': (-( 4 * (J['2s'] + J_bar['2s']) * (J[3] + J_bar[3]) - (J[3] + J_bar[3])**2 - 4 * (J['2s'] + J_bar['2s'])**2 ))**0.25, 
        '6s': 2 * (J['2s'] + J_bar['2s']), 
        7: -( -(J['2c'] + J_bar['2c']) * (2 * (J['2s'] + J_bar['2s']) - (J[3] + J_bar[3])) )**0.5, 
        8: ( -2 * (J['2c'] + J_bar['2c']) * ( 2 * (J['2s'] + J_bar['2s']) - (J[3] + J_bar[3]) ) )**0.5,
        9: 2 * (J['2s'] + J_bar['2s']), 
    }
    return norm[i]

def Q_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    """ proposed optimised angular observables M_i """
    assert i in ['1s', '1c', '2s', '2c', 3, 4, 5, '6s', 7, 8, 9], f"{i} not implemented!"
    return J_s[i]

def Q_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    """ proposed optimised angular observables M_i """
    assert i in ['1s', '1c', '2s', '2c', 3, 4, 5, '6s', 7, 8, 9], f"{i} not implemented!"
    norm = {
        '1s': (J['2s'] + J_bar['2s']),
        '1c': -(J['2c'] + J_bar['2c']),
        '2s': (J['2s'] + J_bar['2s']),
        '2c': -(J['2c'] + J_bar['2c']),
        3: 2 * (J['2s'] + J_bar['2s']),
        4: -(- (J['2c'] + J_bar['2c']) * (2 * (J['2s'] + J_bar['2s']) - (J[3] + J_bar[3])) )**0.5,
        5: ((J['2c'] + J_bar['2c']) * ((J[3] + J_bar[3]) - 2 * (J['2s'] + J_bar['2s'])) )**0.5,
        # '6s': (-( 4 * (J['2s'] + J_bar['2s']) * (J[3] + J_bar[3]) - (J[3] + J_bar[3])**2 - 4 * (J['2s'] + J_bar['2s'])**2 ))**0.25, 
        '6s': 2 * (J['2s'] + J_bar['2s']), 
        7: -( -(J['2c'] + J_bar['2c']) * (2 * (J['2s'] + J_bar['2s']) - (J[3] + J_bar[3])) )**0.5, 
        8: ( -2 * (J['2c'] + J_bar['2c']) * ( 2 * (J['2s'] + J_bar['2s']) - (J[3] + J_bar[3]) ) )**0.5,
        9: 2 * (J['2s'] + J_bar['2s']), 
    }
    return norm[i]

def M_theory_den_Bs_prime(y, x, gamma, J, J_bar, J_h, J_s, i):
    """ proposed optimised angular observables M_i """
    assert i in [4, 5, 7, 8], f"{i} not implemented!"
    norm = {
        4: (-(J['2c'] + J_bar['2c']) * (J['2s'] + J_bar['2s']) )**0.5,
        5: 2 * (-(J['2c'] + J_bar['2c']) * (J['2s'] + J_bar['2s']) )**0.5,
        7: (-(J['2c'] + J_bar['2c']) * (J['2s'] + J_bar['2s']) )**0.5, 
        8: (-(J['2c'] + J_bar['2c']) * (J['2s'] + J_bar['2s']) )**0.5, 
    }
    return norm[i]

def Q_theory_den_Bs_prime(y, x, gamma, J, J_bar, J_h, J_s, i):
    """ proposed optimised angular observables M_i """
    assert i in [4, 5, 7, 8], f"{i} not implemented!"
    norm = {
        4: (-(J['2c'] + J_bar['2c']) * (J['2s'] + J_bar['2s']) )**0.5,
        5: 2 * (-(J['2c'] + J_bar['2c']) * (J['2s'] + J_bar['2s']) )**0.5,
        7: (-(J['2c'] + J_bar['2c']) * (J['2s'] + J_bar['2s']) )**0.5, 
        8: (-(J['2c'] + J_bar['2c']) * (J['2s'] + J_bar['2s']) )**0.5, 
    }
    return norm[i]

def S_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -S_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)
    return S_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)

def A_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -A_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)
    return A_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)

def S_experiment_num_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -S_theory_num_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s, i)
    return S_theory_num_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s, i)

def A_experiment_num_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -A_theory_num_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s, i)
    return A_theory_num_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s, i)

def K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -K_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)
    return K_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)

def W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -W_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)
    return W_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)

def H_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -H_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)
    return H_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)

def Z_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -Z_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)
    return Z_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)

def M_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -M_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)
    return M_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)

def Mprime_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -M_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)
    return M_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)

def Q_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -Q_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)
    return Q_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)

def Qprime_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -Q_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)
    return Q_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)

def S_experiment_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    r"""CP-averaged angular observable $S_i$ in the LHCb convention.
    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return S_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)/SA_den_Bs(y, x, gamma, J, J_bar, J_h, J_s)

def A_experiment_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    r"""Angular CP asymmetry $A_i$ in the LHCb convention.
    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return A_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)/SA_den_Bs(y, x, gamma, J, J_bar, J_h, J_s)

def S_experiment_int_t_Bs(tmix, tmax, y, x, gamma, J, J_bar, J_h, J_s, i):
    r"""CP-averaged angular observable $S_i$ in the LHCb convention.
    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return S_experiment_num_int_t_Bs(tmix, tmax, y, x, gamma, J, J_bar, J_h, J_s, i)/SA_den_int_t_Bs(tmix, tmax, y, x, gamma, J, J_bar, J_h, J_s)

def A_experiment_int_t_Bs(tmix, tmax, y, x, gamma, J, J_bar, J_h, J_s, i):
    r"""Angular CP asymmetry $A_i$ in the LHCb convention.
    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return A_experiment_num_int_t_Bs(tmix, tmax, y, x, gamma, J, J_bar, J_h, J_s, i)/SA_den_int_t_Bs(tmix, tmax, y, x, gamma, J, J_bar, J_h, J_s)

def K_experiment_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    r"""CP-averaged angular observable $K_i$ from the time-dependent differential decay rate.
    Observables related to the terms depending on \cosh(y\Gamma t)

    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)/SA_den_Bs(y, x, gamma, J, J_bar, J_h, J_s)

def W_experiment_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    r"""Angular CP asymmetry $W_i$ from the time-dependent differential decay rate.
    Observables related to the terms depending on \cosh(y\Gamma t)

    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)/SA_den_Bs(y, x, gamma, J, J_bar, J_h, J_s)

def H_experiment_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    r"""CP-averaged angular observable $H_i$ from the time-dependent differential decay rate.
    Observables related to the terms depending on \sinh(y\Gamma t)

    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return H_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)/SA_den_Bs(y, x, gamma, J, J_bar, J_h, J_s)

def Z_experiment_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    r"""CP-averaged angular observable $Z_i$ from the time-dependent differential decay rate.
    Observables related to the terms depending on \sinh(y\Gamma t)

    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return Z_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)/SA_den_Bs(y, x, gamma, J, J_bar, J_h, J_s)

def M_experiment_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    return M_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)/M_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)

def Mprime_experiment_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    return M_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)/M_theory_den_Bs_prime(y, x, gamma, J, J_bar, J_h, J_s, i)

def Q_experiment_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    return Q_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)/Q_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)

def Qprime_experiment_Bs(y, x, gamma, J, J_bar, J_h, J_s, i):
    return Q_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, i)/Q_theory_den_Bs_prime(y, x, gamma, J, J_bar, J_h, J_s, i)

def dGdq2_interference_Bs(y, x, gamma, J, J_bar, J_h, J_s):
    # TODO: add correct citation
    flavio.citations.register("Descotes-Genon:2015hea")
    return (- y/(1-y**2) * observables.dGdq2(J_h))/2.
    
def dGdq2_ave_Bs(y, x, gamma, J, J_bar, J_h, J_s):
    # (48) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    return (1/(1-y**2) * (observables.dGdq2(J) + observables.dGdq2(J_bar))
            - y/(1-y**2) * observables.dGdq2(J_h))/2.
    
def dGdq2_ave_Bs_part1(y, x, gamma, J, J_bar, J_h, J_s):
    # (48) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    return (1/(1-y**2) * (observables.dGdq2(J) + observables.dGdq2(J_bar)))/2.

def dGdq2_ave_Bs_part2(y, x, gamma, J, J_bar, J_h, J_s):
    # (48) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    return (- y/(1-y**2) * observables.dGdq2(J_h))/2.
    
def dGdq2_ave_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s):
    # (48) of 1502.05509
    flavio.citations.register("Descotes-Genon:2015hea")
    return ((observables.dGdq2(J) + observables.dGdq2(J_bar)) * definite_int_cosh(y, gamma, tmin, tmax)
            - observables.dGdq2(J_h) * definite_int_sinh(y, gamma, tmin, tmax))/2.

# denominator of S_i and A_i observables
def SA_den_Bs(y, x, gamma, J, J_bar, J_h, J_s):
    return 2*dGdq2_ave_Bs(y, x, gamma, J, J_bar, J_h, J_s)

def SA_den_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s):
    return 2*dGdq2_ave_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s)

def FL_Bs(y, x, gamma, J, J_bar, J_h, J_s):
    r"""Longitudinal polarization fraction $F_L$"""
    return FL_num_Bs(y, x, gamma, J, J_bar, J_h, J_s)/SA_den_Bs(y, x, gamma, J, J_bar, J_h, J_s)

def FL_num_Bs(y, x, gamma, J, J_bar, J_h, J_s):
    return -S_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '2c')

def FL_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s):
    r"""Longitudinal polarization fraction $F_L$"""
    return FL_num_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s)/SA_den_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s)

def AFB_num_Bs(y, x, gamma, J, J_bar, J_h, J_s): 
    """ Forward-backward asymmetry $A_{FB}$ """
    return 3/4 * A_theory_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '6s')

def AFB_Bs(y, x, gamma, J, J_bar, J_h, J_s):
    """ Forward-backward asymmetry $A_{FB}$ """
    return AFB_num_Bs(y, x, gamma, J, J_bar, J_h, J_s) / SA_den_Bs(y, x, gamma, J, J_bar, J_h, J_s)

def AFB_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s):
    """ Forward-backward asymmetry $A_{FB}$ time integrated from tmin to tmax 
    Takes: 
    - tmin, tmax: minimum and maximum time
    - y: DeltaGamma/Gamma
    - x: DeltaM/Gamma
    - gamma: decay width of the B_s meson (default 0.6597)
    - J, J_bar, J_h, J_s: angular coefficients at the given q2
    Returns:
    - AFB: the time-integrated forward-backward asymmetry, normalized to the time-averaged decay rate
    """
    num = 3/4 * A_theory_num_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s, '6s')  # AFB 
    den = SA_den_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s)                    # normalisation
    return num / den

def FL_num_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s):
    return -S_theory_num_int_t_Bs(tmin, tmax, y, x, gamma, J, J_bar, J_h, J_s, '2c')

def bsvll_dbrdq2(q2, wc_obj, par, B, V, lep):
    tauB = par['tau_'+B]
    return tauB * bsvll_obs(dGdq2_ave_Bs, q2, wc_obj, par, B, V, lep)

def bsvll_obs_int(function, q2min, q2max, wc_obj, par, B, V, lep, epsrel=0.005):
    def obs(q2):
        return bsvll_obs(function, q2, wc_obj, par, B, V, lep)
    return flavio.physics.bdecays.bvll.observables.nintegrate_pole(obs, q2min, q2max, epsrel=epsrel)

def bsvll_dbrdq2_int(q2min, q2max, wc_obj, par, B, V, lep, epsrel=0.005):
    def obs(q2):
        return bsvll_dbrdq2(q2, wc_obj, par, B, V, lep)
    return flavio.physics.bdecays.bvll.observables.nintegrate_pole(obs, q2min, q2max, epsrel=epsrel)/(q2max-q2min)

# Functions returning functions needed for Prediction instances

def bsvll_dbrdq2_int_func(B, V, lep):
    def fct(wc_obj, par, q2min, q2max):
        return bsvll_dbrdq2_int(q2min, q2max, wc_obj, par, B, V, lep)
    return fct

def bsvll_dbrdq2_func(B, V, lep):
    def fct(wc_obj, par, q2):
        return bsvll_dbrdq2(q2, wc_obj, par, B, V, lep)
    return fct

def bsvll_obs_int_ratio_func(func_num, func_den, B, V, lep):
    def fct(wc_obj, par, q2min, q2max):
        num = bsvll_obs_int(func_num, q2min, q2max, wc_obj, par, B, V, lep)
        if num == 0:
            return 0
        denom = bsvll_obs_int(func_den, q2min, q2max, wc_obj, par, B, V, lep)
        return num/denom
    return fct

def bsvll_obs_int_ratio_leptonflavour(func, B, V, l1, l2):
    def fct(wc_obj, par, q2min, q2max):
        num = bsvll_obs_int(func, q2min, q2max, wc_obj, par, B, V, l1, epsrel=0.0005)
        if num == 0:
            return 0
        denom = bsvll_obs_int(func, q2min, q2max, wc_obj, par, B, V, l2, epsrel=0.0005)
        return num/denom
    return fct

def bsvll_obs_ratio_func(func_num, func_den, B, V, lep):
    def fct(wc_obj, par, q2):
        num = bsvll_obs(func_num, q2, wc_obj, par, B, V, lep)
        if num == 0:
            return 0
        denom = bsvll_obs(func_den, q2, wc_obj, par, B, V, lep)
        return num/denom
    return fct

# function for the LHCb (1804.07167) Bs->K*0 integrated branching ratio
def bsvll_dbrdq2_19_func(B, P, lep):
    def fct(wc_obj, par):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The predictions in the region of narrow charmonium resonances are not meaningful.*")
            warnings.filterwarnings("ignore", message="The QCDF corrections should not be trusted*")

            q2max = 19
            q2min = 0.1
            return (1+par['delta_BsKstarmumu'])*bsvll_dbrdq2_int(q2min, q2max, wc_obj, par, B, P, lep)*(q2max-q2min)
    return fct

class BsVll_int_ratio(observables.BVllObservableBinned):
    """Binned ratio of functions if angular coefficients"""
    def __init__(self, func_num, func_den, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsrel = 0.005
        self.func_num = func_num
        self.func_den = func_den
        
    def j_s(self, q2):
        """Get angular coeffs. Cache and only recompute if necessary."""
        h = self.ha(q2)
        if q2 not in self._j:
            self._j[q2] = angular.angularcoeffs_s_v(h, q2, self.mB, self.mV, self.mb, 0, self.ml, self.ml)
        return self._j[q2]

    def j_h(self, q2):
        """Get CP conjugate angular coeffs. Cache and only recompute if necessary."""
        hbar = self.ha_bar(q2)
        if q2 not in self._j_bar:
            self._j_bar[q2] = angular.angularcoeffs_h_v(hbar, q2, self.mB, self.mV, self.mb, 0, self.ml, self.ml)
        return self._j_bar[q2]
    
    def jfunc(self, function, q2):
        """Return a function of J and Jbar at one value of q2."""
        return function(self.j(q2), self.jbar(q2), self.j_h(q2), self.j_s(q2))
    
    def obs_num(self, q2):
        return self.jfunc(self.func_num, q2)

    def obs_den(self, q2):
        return self.jfunc(self.func_den, q2)

    def __call__(self):
        if self.q2max_allowed <= self.q2min_allowed:
            return 0
        num = flavio.physics.bdecays.bvll.observables.nintegrate_pole(self.obs_num, self.q2min_allowed, self.q2max_allowed, epsrel=self.epsrel)
        if num == 0:
            return 0
        den = flavio.physics.bdecays.bvll.observables.nintegrate_pole(self.obs_den, self.q2min_allowed, self.q2max_allowed, epsrel=self.epsrel)
        return num / den

# Observable and Prediction instances

_tex = {'e': 'e', 'mu': '\mu', 'tau': r'\tau'}
_observables = {
'FL': {'func_num': FL_num_Bs, 'tex': r'\overline{F_L}', 'desc': 'Time-averaged longitudinal polarization fraction'},
'S1s': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: S_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '1s'), 'tex': r'\overline{S_{1s}}', 'desc': 'Time-averaged, CP-averaged angular observable'},
'S1c': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: S_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '1c'), 'tex': r'\overline{S_{1c}}', 'desc': 'Time-averaged, CP-averaged angular observable'},
'S2s': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: S_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '2s'), 'tex': r'\overline{S_{2s}}', 'desc': 'Time-averaged, CP-averaged angular observable'},
'S2c': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: S_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '2c'), 'tex': r'\overline{S_{2c}}', 'desc': 'Time-averaged, CP-averaged angular observable'},
'S3': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: S_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 3), 'tex': r'\overline{S_3}', 'desc': 'Time-averaged, CP-averaged angular observable'},
'S4': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: S_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 4), 'tex': r'\overline{S_4}', 'desc': 'Time-averaged, CP-averaged angular observable'},
'S7': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: S_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 7), 'tex': r'\overline{S_7}', 'desc': 'Time-averaged, CP-averaged angular observable'},
'A5': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: A_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 5), 'tex': r'\overline{A_5}', 'desc': 'Time-averaged, CP-asymmetric angular observable'},
'A6s': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: A_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '6s'), 'tex': r'\overline{A_6^s}', 'desc': 'Time-averaged, CP-asymmetric angular observable'},
'A8': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: A_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 8), 'tex': r'\overline{A_8}', 'desc': 'Time-averaged, CP-asymmetric angular observable'},
'A9': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: A_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 9), 'tex': r'\overline{A_9}', 'desc': 'Time-averaged, CP-asymmetric angular observable'},
}
_hadr = {
'Bs->phi': {'tex': r"B_s\to \phi ", 'B': 'Bs', 'V': 'phi', },
'Bs->K*0': {'tex': r"B_s\to K^* ", 'B': 'Bs', 'V': 'K*0', },
}
for l in ['e', 'mu', 'tau']:
    for M in _hadr.keys():

        _process_tex = _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+r"^-"
        _process_taxonomy = r'Process :: $b$ hadron decays :: FCNC decays :: $B\to V\ell^+\ell^-$ :: $' + _process_tex + r"$"

        for obs in sorted(_observables.keys()):

            # binned angular observables
            _obs_name = "<" + obs + ">("+M+l+l+")"
            _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
            _obs.set_description('Binned ' + _observables[obs]['desc'] + r" in $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$")
            _obs.tex = r"$\langle " + _observables[obs]['tex'] + r"\rangle(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
            _obs.add_taxonomy(_process_taxonomy)
            Prediction(_obs_name, bsvll_obs_int_ratio_func(_observables[obs]['func_num'], SA_den_Bs, _hadr[M]['B'], _hadr[M]['V'], l))

            # differential angular observables
            _obs_name = obs + "("+M+l+l+")"
            _obs = Observable(name=_obs_name, arguments=['q2'])
            _obs.set_description(_observables[obs]['desc'][0].capitalize() + _observables[obs]['desc'][1:] + r" in $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$")
            _obs.tex = r"$" + _observables[obs]['tex'] + r"(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
            _obs.add_taxonomy(_process_taxonomy)
            Prediction(_obs_name, bsvll_obs_ratio_func(_observables[obs]['func_num'], SA_den_Bs, _hadr[M]['B'], _hadr[M]['V'], l))

        # binned branching ratio
        _obs_name = "<dBR/dq2>("+M+l+l+")"
        _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
        _obs.set_description(r"Binned time-integrated differential branching ratio of $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$")
        _obs.tex = r"$\langle \frac{d\overline{\text{BR}}}{dq^2} \rangle(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
        _obs.add_taxonomy(_process_taxonomy)
        Prediction(_obs_name, bsvll_dbrdq2_int_func(_hadr[M]['B'], _hadr[M]['V'], l))

        # differential branching ratio
        _obs_name = "dBR/dq2("+M+l+l+")"
        _obs = Observable(name=_obs_name, arguments=['q2'])
        _obs.set_description(r"Differential time-integrated branching ratio of $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$")
        _obs.tex = r"$\frac{d\overline{\text{BR}}}{dq^2}(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
        _obs.add_taxonomy(_process_taxonomy)
        Prediction(_obs_name, bsvll_dbrdq2_func(_hadr[M]['B'], _hadr[M]['V'], l))

        # for Bs->K*0 mu mu we add a separate observable for the measurement in 1804.07167
        if M == 'Bs->K*0' and l == 'mu':
            _obs_name = "BR_LHCb("+M+l+l+")"
            _obs = Observable(name=_obs_name)
            _obs.set_description(r"Branching ratio of $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$ measured by LHCb in 2018")
            _obs.tex = r"$\overline{\text{BR}}(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
            _obs.add_taxonomy(_process_taxonomy)
            Prediction(_obs_name, bsvll_dbrdq2_19_func(_hadr[M]['B'], _hadr[M]['V'], l))

# Lepton flavour ratios
for l in [('mu','e'), ('tau','mu'),]:
    for M in _hadr.keys():

        # binned ratio of BRs
        _obs_name = "<R"+l[0]+l[1]+">("+M+"ll)"
        _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
        _obs.set_description(r"Ratio of partial branching ratios of $" + _hadr[M]['tex'] +_tex[l[0]]+r"^+ "+_tex[l[0]]+r"^-$" + " and " + r"$" + _hadr[M]['tex'] +_tex[l[1]]+r"^+ "+_tex[l[1]]+"^-$")
        _obs.tex = r"$\langle R_{" + _tex[l[0]] + ' ' + _tex[l[1]] + r"} \rangle(" + _hadr[M]['tex'] + r"\ell^+\ell^-)$"
        for li in l:
            # add taxonomy for both processes (e.g. Bs->Vee and Bs->Vmumu)
            _obs.add_taxonomy(r'Process :: $b$ hadron decays :: FCNC decays :: $B\to V\ell^+\ell^-$ :: $' + _hadr[M]['tex'] +_tex[li]+r"^+"+_tex[li]+r"^-$")
        Prediction(_obs_name, bsvll_obs_int_ratio_leptonflavour(dGdq2_ave_Bs, _hadr[M]['B'], _hadr[M]['V'], *l))

obs_td_bsphill = {
    'K1s': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '1s'), 'tex': r'\mathcal{K}_{1s}', 'desc': 'CP-averaged angular observable from the time-dependent decay rate. (proportional to \cosh(y\Gamma t))'},
    'K1c': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '1c'), 'tex': r'\mathcal{K}_{1c}', 'desc': 'CP-averaged angular observable from the time-dependent decay rate. (proportional to \cosh(y\Gamma t))'},
    'K2s': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '2s'), 'tex': r'\mathcal{K}_{2s}', 'desc': 'CP-averaged angular observable from the time-dependent decay rate. (proportional to \cosh(y\Gamma t))'},
    'K2c': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '2c'), 'tex': r'\mathcal{K}_{2c}', 'desc': 'CP-averaged angular observable from the time-dependent decay rate. (proportional to \cosh(y\Gamma t))'},
    'K3': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 3), 'tex': r'\mathcal{K}_{3}', 'desc': 'CP-averaged angular observable from the time-dependent decay rate. (proportional to \cosh(y\Gamma t))'},
    'K4': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 4), 'tex': r'\mathcal{K}_{4}', 'desc': 'CP-averaged angular observable from the time-dependent decay rate. (proportional to \cosh(y\Gamma t))'},
    'K5': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 5), 'tex': r'\mathcal{K}_{5}', 'desc': 'CP-asymmetric angular observable from the time-dependent decay rate. (proportional to \cos(y\Gamma t))'},
    'K6s': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '6s'), 'tex': r'\mathcal{K}_{6s}', 'desc': 'CP-asymmetric angular observable from the time-dependent decay rate. (proportional to \cos(y\Gamma t))'},
    'K6c': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '6c'), 'tex': r'\mathcal{K}_{6c}', 'desc': 'CP-asymmetric angular observable from the time-dependent decay rate. (proportional to \cos(y\Gamma t))'},
    'K7': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 7), 'tex': r'\mathcal{K}_{7}', 'desc': 'CP-averaged angular observable from the time-dependent decay rate. (proportional to \cosh(y\Gamma t))'},
    'K8': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 8), 'tex': r'\mathcal{K}_{8}', 'desc': 'CP-asymmetric angular observable from the time-dependent decay rate. (proportional to \cos(y\Gamma t))'},
    'K9': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 9), 'tex': r'\mathcal{K}_{9}', 'desc': 'CP-asymmetric angular observable from the time-dependent decay rate. (proportional to \cos(y\Gamma t))'},
    'W1s': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '1s'), 'tex': r'\mathcal{W}_{1s}', 'desc': 'CP-asymmetric angular observable from the time-dependent decay rate. (proportional to \cos(y\Gamma t))'},
    'W1c': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '1c'), 'tex': r'\mathcal{W}_{1c}', 'desc': 'CP-asymmetric angular observable from the time-dependent decay rate. (proportional to \cos(y\Gamma t))'},
    'W2s': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '2s'), 'tex': r'\mathcal{W}_{2s}', 'desc': 'CP-asymmetric angular observable from the time-dependent decay rate. (proportional to \cos(y\Gamma t))'},
    'W2c': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '2c'), 'tex': r'\mathcal{W}_{2c}', 'desc': 'CP-asymmetric angular observable from the time-dependent decay rate. (proportional to \cos(y\Gamma t))'},
    'W3': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 3), 'tex': r'\mathcal{W}_{3}', 'desc': 'CP-asymmetric angular observable from the time-dependent decay rate. (proportional to \cos(y\Gamma t))'},
    'W4': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 4), 'tex': r'\mathcal{W}_{4}', 'desc': 'CP-asymmetric angular observable from the time-dependent decay rate. (proportional to \cos(y\Gamma t))'},
    'W5': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 5), 'tex': r'\mathcal{W}_{5}', 'desc': 'CP-averaged angular observable from the time-dependent decay rate. (proportional to \cosh(y\Gamma t))'},
    'W6s': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '6s'), 'tex': r'\mathcal{W}_{6s}', 'desc': 'CP-averaged angular observable from the time-dependent decay rate. (proportional to \cosh(y\Gamma t))'},
    'W6c': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '6c'), 'tex': r'\mathcal{W}_{6c}', 'desc': 'CP-averaged angular observable from the time-dependent decay rate. (proportional to \cosh(y\Gamma t))'},
    'W7': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 7), 'tex': r'\mathcal{W}_{7}', 'desc': 'CP-asymmetric angular observable from the time-dependent decay rate. (proportional to \cos(y\Gamma t))'},
    'W8': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 8), 'tex': r'\mathcal{W}_{8}', 'desc': 'CP-averaged angular observable from the time-dependent decay rate. (proportional to \cosh(y\Gamma t))'},
    'W9': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 9), 'tex': r'\mathcal{W}_{9}', 'desc': 'CP-averaged angular observable from the time-dependent decay rate. (proportional to \cosh(y\Gamma t))'},
    'H1s': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: H_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '1s'), 'tex': r'\mathcal{H}_{1s}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))'},
    'H1c': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: H_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '1c'), 'tex': r'\mathcal{H}_{1c}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))'},
    'H2s': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: H_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '2s'), 'tex': r'\mathcal{H}_{2s}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))'},
    'H2c': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: H_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '2c'), 'tex': r'\mathcal{H}_{2c}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))'},
    'H3': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: H_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 3), 'tex': r'\mathcal{H}_{3}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))'},
    'H4': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: H_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 4), 'tex': r'\mathcal{H}_{4}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))'},
    'H5': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: H_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 5), 'tex': r'\mathcal{H}_{5}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))'},
    'H6s': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: H_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '6s'), 'tex': r'\mathcal{H}_{6s}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))'},
    'H6c': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: H_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '6c'), 'tex': r'\mathcal{H}_{6c}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))'},
    'H7': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: H_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 7), 'tex': r'\mathcal{H}_{7}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))'},
    'H8': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: H_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 8), 'tex': r'\mathcal{H}_{8}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))'},
    'H9': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: H_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 9), 'tex': r'\mathcal{H}_{9}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))'},
    'Z1s': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Z_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '1s'), 'tex': r'\mathcal{Z}_{1s}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))'},
    'Z1c': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Z_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '1c'), 'tex': r'\mathcal{Z}_{1c}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))'},
    'Z2s': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Z_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '2s'), 'tex': r'\mathcal{Z}_{2s}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))'},
    'Z2c': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Z_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '2c'), 'tex': r'\mathcal{Z}_{2c}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))'},
    'Z3': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Z_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 3), 'tex': r'\mathcal{Z}_{3}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))'},
    'Z4': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Z_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 4), 'tex': r'\mathcal{Z}_{4}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))'},
    'Z5': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Z_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 5), 'tex': r'\mathcal{Z}_{5}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))'},
    'Z6s': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Z_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '6s'), 'tex': r'\mathcal{Z}_{6s}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))'},
    'Z6c': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Z_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '6c'), 'tex': r'\mathcal{Z}_{6c}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))'},
    'Z7': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Z_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 7), 'tex': r'\mathcal{Z}_{7}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))'},
    'Z8': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Z_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 8), 'tex': r'\mathcal{Z}_{8}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))'},
    'Z9': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Z_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 9), 'tex': r'\mathcal{Z}_{9}', 'desc': 'Angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))'},
    'Q1s': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '1s'), 'tex': r'Q_{1s}', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, '1s')},
    'Q1c': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '1c'), 'tex': r'Q_{1c}', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, '1c')},
    'Q2s': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '2s'), 'tex': r'Q_{2s}', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, '2s')},
    'Q2c': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '2c'), 'tex': r'Q_{2c}', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, '2c')},
    'Q3': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 3), 'tex': r'Q_{3}', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, 3)},
    'Q4': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 4), 'tex': r'Q_{4}', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, 4)},
    'Q5': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 5), 'tex': r'Q_{5}', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, 5)},
    'Q6s': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '6s'), 'tex': r'Q_{6s}', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, '6s')},
    'Q7': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 7), 'tex': r'Q_{7}', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, 7)},
    'Q8': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 8), 'tex': r'Q_{8}^-', 'desc': 'Optimised angular observable from the time-dependent decay rate (Following 1502.05509)', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, 8)},
    'Q9': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 9), 'tex': r'Q_{9}', 'desc': 'Optimised angular observable from the time-dependent decay rate (Following 1502.05509)', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, 9)},
    'M1s': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: M_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '1s'), 'tex': r'\mathcal{M}_{1s}', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: M_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, '1s')},
    'M1c': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: M_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '1c'), 'tex': r'\mathcal{M}_{1c}', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: M_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, '1c')},
    'M2s': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: M_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '2s'), 'tex': r'\mathcal{M}_{2s}', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: M_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, '2s')},
    'M2c': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: M_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '2c'), 'tex': r'\mathcal{M}_{2c}', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: M_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, '2c')},
    'M3': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: M_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 3), 'tex': r'\mathcal{M}_{3}', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: M_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, 3)},
    'M4': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: M_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 4), 'tex': r'\mathcal{M}_{4}', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: M_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, 4)},
    'M5': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: M_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 5), 'tex': r'\mathcal{M}_{5}', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: M_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, 5)},
    'M6s': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: M_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '6s'), 'tex': r'\mathcal{M}_{6s}', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: M_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, '6s')},
    'M7': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: M_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 7), 'tex': r'\mathcal{M}_{7}', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: M_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, 7)},
    'M8': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: M_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 8), 'tex': r'\mathcal{M}_{8}', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: M_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, 8)},
    'M9': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: M_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 9), 'tex': r'\mathcal{M}_{9}', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: M_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, 9)},
    'M4p': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Mprime_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 4), 'tex': r'\mathcal{M}_{4}^\prime', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: M_theory_den_Bs_prime(y, x, gamma, J, J_bar, J_h, J_s, 4)},
    'M5p': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Mprime_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 5), 'tex': r'\mathcal{M}_{5}^\prime', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: M_theory_den_Bs_prime(y, x, gamma, J, J_bar, J_h, J_s, 5)},
    'M7p': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Mprime_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 7), 'tex': r'\mathcal{M}_{7}^\prime', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: M_theory_den_Bs_prime(y, x, gamma, J, J_bar, J_h, J_s, 7)},
    'M8p': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Mprime_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 8), 'tex': r'\mathcal{M}_{8}^\prime', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sinh(y\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: M_theory_den_Bs_prime(y, x, gamma, J, J_bar, J_h, J_s, 8)},
    'Q4p': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Qprime_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 4), 'tex': r'Q_{4}^\prime', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_theory_den_Bs_prime(y, x, gamma, J, J_bar, J_h, J_s, 4)},
    'Q5p': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Qprime_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 5), 'tex': r'Q_{5}^\prime', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_theory_den_Bs_prime(y, x, gamma, J, J_bar, J_h, J_s, 5)},
    'Q7p': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Qprime_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 7), 'tex': r'Q_{7}^\prime', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_theory_den_Bs_prime(y, x, gamma, J, J_bar, J_h, J_s, 7)},
    'Q8p': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: Qprime_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 8), 'tex': r'Q_{8}^\prime', 'desc': 'Optimised angular observable from the time-dependent decay rate. (proportional to \sin(x\Gamma t))', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: Q_theory_den_Bs_prime(y, x, gamma, J, J_bar, J_h, J_s, 8)},
    'M4_den': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: M_theory_den_Bs(y, x, gamma, J, J_bar, J_h, J_s, 4), 'tex': r'blab', 'desc': 'blubb', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: 1 },
    'M4_num': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: M_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 4), 'tex': r'blab', 'desc': 'blubb', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: 1 },
    'Iden': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: SA_den_Bs(y, x, gamma, J, J_bar, J_h, J_s), 'tex': r'blab', 'desc': 'blubb', 'den': lambda y, x, gamma, J, J_bar, J_h, J_s: 1/(1 - y*y) },
}

_hadr = {
'Bs->phi': {'tex': r"B_s\to \phi ", 'B': 'Bs', 'V': 'phi', },
}
for lep in ['e', 'mu', 'tau']:
    for M in _hadr.keys():

        _process_tex = _hadr[M]['tex'] +_tex[lep] + r"^+" + _tex[lep] + r"^-"
        _process_taxonomy = r'Process :: $b$ hadron decays :: FCNC decays :: $B\to V\ell^+\ell^-$ :: $' + _process_tex + r"$"

        for obs in sorted(obs_td_bsphill.keys()):

            # binned angular observables
            _obs_name = "<" + obs + ">(" + M + lep + lep + ")"
            _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
            _obs.set_description('Binned ' + obs_td_bsphill[obs]['desc'] + r" in $" + _hadr[M]['tex'] + _tex[lep] + r"^+" + _tex[lep] + "^-$")
            _obs.tex = r"$\langle " + obs_td_bsphill[obs]['tex'] + r"\rangle(" + _hadr[M]['tex'] + _tex[lep] + r"^+" + _tex[lep] + "^-)$"
            _obs.add_taxonomy(_process_taxonomy)
            norm_fun = SA_den_Bs if 'den' not in obs_td_bsphill[obs] else obs_td_bsphill[obs]['den']
            Prediction(_obs_name, bsvll_obs_int_ratio_func(obs_td_bsphill[obs]['func_num'], norm_fun, _hadr[M]['B'], _hadr[M]['V'], lep))

            # differential angular observables
            _obs_name = obs + "(" + M + lep + lep + ")"
            _obs = Observable(name=_obs_name, arguments=['q2'])
            _obs.set_description(obs_td_bsphill[obs]['desc'][0].capitalize() + obs_td_bsphill[obs]['desc'][1:] + r" in $" + _hadr[M]['tex'] + _tex[lep] + r"^+" + _tex[lep] + "^-$")
            _obs.tex = r"$" + obs_td_bsphill[obs]['tex'] + r"(" + _hadr[M]['tex'] + _tex[lep] + r"^+" + _tex[lep] + "^-)$"
            _obs.add_taxonomy(_process_taxonomy)
            Prediction(_obs_name, bsvll_obs_ratio_func(obs_td_bsphill[obs]['func_num'], norm_fun, _hadr[M]['B'], _hadr[M]['V'], lep))

def denominator_optimised_p(y, x, gamma, J, J_bar, J_h, J_s):
    return (J['2s'] + J_bar['2s'])

def denominator_optimised_pprime(y, x, gamma, J, J_bar, J_h, J_s):
    return sqrt(-1 * (J['2c'] + J_bar['2c']) * (J['2s'] + J_bar['2s']))

observables_p = {
    'SP1': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 3) / 2, 'tex': r'P_{1}', 'desc': 'CP-averaged optimised angular observable. '},
    'SP2': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '6s') / 8, 'tex': r'P_{2}', 'desc': 'CP-averaged optimised angular observable. '},
    'SP3': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: -K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 9) / 4, 'tex': r'P_{3}', 'desc': 'CP-averaged optimised angular observable. '},
    'SS': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '6c'), 'tex': r'S', 'desc': 'CP-averaged optimised angular observable. '},
    'AP1': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 3) / 2, 'tex': r'P_{1}', 'desc': 'CP-asymmetric optimised angular observable. '},
    'AP2': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '6s') / 8, 'tex': r'P_{2}', 'desc': 'CP-asymmetric optimised angular observable. '},
    'AP3': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: -W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 9) / 4, 'tex': r'P_{3}', 'desc': 'CP-asymmetric optimised angular observable. '},
    'AS': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, '6c'), 'tex': r'S', 'desc': 'CP-asymmetric optimised angular observable. '},
}
observables_pprime = {
    'SP4p': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 4), 'tex': r'P_{4}^\prime', 'desc': 'CP-averaged optimised angular observable. '},
    'SP5p': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: 0.5 * K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 5), 'tex': r'P_{5}^\prime', 'desc': 'CP-averaged optimised angular observable. '},
    'SP6p': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: -0.5 * K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 7), 'tex': r'P_{6}^\prime', 'desc': 'CP-averaged optimised angular observable. '},
    'SP8p': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: K_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 8), 'tex': r'P_{8}^\prime', 'desc': 'CP-averaged optimised angular observable. '},
    'AP4p': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 4), 'tex': r'P_{4}^\prime', 'desc': 'CP-asymmetric optimised angular observable. '},
    'AP5p': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: 0.5 * W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 5), 'tex': r'P_{5}^\prime', 'desc': 'CP-asymmetric optimised angular observable. '},
    'AP6p': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: -0.5 * W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 7), 'tex': r'P_{6}^\prime', 'desc': 'CP-asymmetric optimised angular observable. '},
    'AP8p': {'func_num': lambda y, x, gamma, J, J_bar, J_h, J_s: W_experiment_num_Bs(y, x, gamma, J, J_bar, J_h, J_s, 8), 'tex': r'P_{8}^\prime', 'desc': 'CP-asymmetric optimised angular observable. '},
}

for lep in ['e', 'mu', 'tau']:
    for M in _hadr.keys():

        _process_tex = _hadr[M]['tex'] +_tex[lep] + r"^+" + _tex[lep] + r"^-"
        _process_taxonomy = r'Process :: $b$ hadron decays :: FCNC decays :: $B\to V\ell^+\ell^-$ :: $' + _process_tex + r"$"

        for obs, element in sorted(observables_p.items()):

            # binned angular observables
            _obs_name = "<" + obs + ">(" + M + lep + lep + ")"
            _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
            _obs.set_description('Binned ' + element['desc'] + r" in $" + _hadr[M]['tex'] + _tex[lep] + r"^+" + _tex[lep] + "^-$")
            _obs.tex = r"$\langle " + element['tex'] + r"\rangle(" + _hadr[M]['tex'] + _tex[lep] + r"^+" + _tex[lep] + "^-)$"
            _obs.add_taxonomy(_process_taxonomy)
            Prediction(_obs_name, bsvll_obs_int_ratio_func(element['func_num'], denominator_optimised_p, _hadr[M]['B'], _hadr[M]['V'], lep))

            # differential angular observables
            _obs_name = obs + "(" + M + lep + lep + ")"
            _obs = Observable(name=_obs_name, arguments=['q2'])
            _obs.set_description(element['desc'][0].capitalize() + element['desc'][1:] + r" in $" + _hadr[M]['tex'] + _tex[lep] + r"^+" + _tex[lep] + "^-$")
            _obs.tex = r"$" + element['tex'] + r"(" + _hadr[M]['tex'] + _tex[lep] + r"^+" + _tex[lep] + "^-)$"
            _obs.add_taxonomy(_process_taxonomy)
            Prediction(_obs_name, bsvll_obs_ratio_func(element['func_num'], denominator_optimised_p, _hadr[M]['B'], _hadr[M]['V'], lep))

for lep in ['e', 'mu', 'tau']:
    for M in _hadr.keys():

        _process_tex = _hadr[M]['tex'] +_tex[lep] + r"^+" + _tex[lep] + r"^-"
        _process_taxonomy = r'Process :: $b$ hadron decays :: FCNC decays :: $B\to V\ell^+\ell^-$ :: $' + _process_tex + r"$"

        for obs, element in sorted(observables_pprime.items()):

            # binned angular observables
            _obs_name = "<" + obs + ">(" + M + lep + lep + ")"
            _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
            _obs.set_description('Binned ' + element['desc'] + r" in $" + _hadr[M]['tex'] + _tex[lep] + r"^+" + _tex[lep] + "^-$")
            _obs.tex = r"$\langle " + element['tex'] + r"\rangle(" + _hadr[M]['tex'] + _tex[lep] + r"^+" + _tex[lep] + "^-)$"
            _obs.add_taxonomy(_process_taxonomy)
            Prediction(_obs_name, bsvll_obs_int_ratio_func(element['func_num'], denominator_optimised_pprime, _hadr[M]['B'], _hadr[M]['V'], lep))

            # differential angular observables
            _obs_name = obs + "(" + M + lep + lep + ")"
            _obs = Observable(name=_obs_name, arguments=['q2'])
            _obs.set_description(element['desc'][0].capitalize() + element['desc'][1:] + r" in $" + _hadr[M]['tex'] + _tex[lep] + r"^+" + _tex[lep] + "^-$")
            _obs.tex = r"$" + element['tex'] + r"(" + _hadr[M]['tex'] + _tex[lep] + r"^+" + _tex[lep] + "^-)$"
            _obs.add_taxonomy(_process_taxonomy)
            Prediction(_obs_name, bsvll_obs_ratio_func(element['func_num'], denominator_optimised_pprime, _hadr[M]['B'], _hadr[M]['V'], lep))
