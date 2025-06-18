r"""Generic $B\to V \ell_1 \bar \ell_2$ helicity amplitudes and angular
distribution. Can be used for $B\to V\ell^+\ell^-$, $B\to V\ell\nu$, and
lepton flavour violating decays."""


from flavio.physics.bdecays.common import lambda_K
from math import sqrt, pi
from argparse import Namespace
import cmath


def transversity_to_helicity(ta):
    H={}
    H['0' ,'V'] = -1j * (ta['0_R'] + ta['0_L'])
    H['0' ,'A'] = -1j * (ta['0_R'] - ta['0_L'])
    H['pl' ,'V'] = 1j * ((ta['para_R'] + ta['para_L']) + (ta['perp_R'] + ta['perp_L']))/sqrt(2)
    H['pl' ,'A'] = 1j * ((ta['para_R'] - ta['para_L']) + (ta['perp_R'] - ta['perp_L']))/sqrt(2)
    H['mi' ,'V'] = 1j * ((ta['para_R'] + ta['para_L']) - (ta['perp_R'] + ta['perp_L']))/sqrt(2)
    H['mi' ,'A'] = 1j * ((ta['para_R'] - ta['para_L']) - (ta['perp_R'] - ta['perp_L']))/sqrt(2)
    return H

def helicity_to_transversity(h: dict[str, float]) -> dict[str, float]:
    """ translate helicity amplitudes to transversity amplitudes """
    ta = {
        '0_R': -0.5j * (h['0', 'V'] + h['0', 'A']),
        '0_L': -0.5j * (h['0', 'V'] - h['0', 'A']), 
        'perp_R': 1j/2/sqrt(2) * (h['mi', 'A'] - h['pl', 'A'] + h['mi', 'V'] - h['pl', 'V']),
        'perp_L': 1j/2/sqrt(2) * (h['pl', 'A'] - h['mi', 'A'] + h['pl', 'V'] - h['mi', 'V']),
        'para_R': -1j/2/sqrt(2) * (h['mi', 'A'] + h['pl', 'A'] + h['mi', 'V'] + h['pl', 'V']),
        'para_L': 1j/2/sqrt(2) * (h['mi', 'A'] + h['pl', 'A'] - h['pl', 'V'] - h['mi', 'V']),
    }
    return ta

def transversity_amps(q2, mB, mV, mqh, mql, ml1, ml2, ff, wc, prefactor):
    """
    transversity amplitudes for B -> V ll decays. 
    Taken from https://arxiv.org/pdf/0811.1214 EQ (3.28) and following. 
    prefactor refers to the N from the paper above. 
    N = Vtb Vts* sqrt{GF^2 alpha_em^2 q^2 beta_mu sqrt(lambda_K(mB^2, mV^2, q^2)) / [3 * 2^10 pi^5 mB^3]}

    NB: - light quark mass not used here, but kept for consistency in the function signature with helicity_amps_v. 
        - ml1 and ml2 are always the same lepton mass. It therefore does not matter which one is used for the A_t definition. 
        - transversity amplitudes are also given in https://arxiv.org/pdf/hep-ph/0502060
    """
    def lambda_qsq(q2, mB, mV):
        return ((mB + mV)**2 - q2) * ((mB - mV)**2 - q2)  # (D.3) in https://arxiv.org/pdf/1503.05534
    lambda_b = lambda_K(mB**2, mV**2, q2)
    wc_9 = wc['v']
    wc_9p = wc['vp']
    wc_10 = wc['a']
    wc_10p = wc['ap']
    wc_7, wc_7p = wc['7'], wc['7p']
    wc_p, wc_pp = wc['p'], wc['pp']

    # (D.5) in https://arxiv.org/pdf/1503.05534
    ff['A2'] = -(mB + mV)*(-ff['A1']*mB**3 - ff['A1']*mB**2*mV + ff['A1']*mB*mV**2 + ff['A1']*mB*q2 + ff['A1']*mV**3 + ff['A1']*mV*q2 + 16*ff['A12']*mB*mV**2)/lambda_qsq(q2, mB, mV)
    ff['T3'] = -(mB - mV)*(-ff['T2']*mB**3 - ff['T2']*mB**2*mV - 3*ff['T2']*mB*mV**2 + ff['T2']*mB*q2 - 3*ff['T2']*mV**3 + ff['T2']*mV*q2 + 8*ff['T23']*mB*mV**2)/lambda_qsq(q2, mB, mV)

    # from https://arxiv.org/pdf/0811.1214
    a0_l_term_one = ( (wc_9 - wc_9p) - (wc_10 - wc_10p) ) * ( (mB**2 - mV**2 - q2) * (mB + mV) * ff['A1'] - lambda_b * ff['A2'] / (mB + mV) )  # first summand of (3.30)
    a0_r_term_one = ( (wc_9 - wc_9p) + (wc_10 - wc_10p) ) * ( (mB**2 - mV**2 - q2) * (mB + mV) * ff['A1'] - lambda_b * ff['A2'] / (mB + mV) )  
    a0_term_two = 2 * mqh * (wc_7 - wc_7p) * ( ( mB**2 + 3 * mV**2 - q2 ) * ff['T2'] - lambda_b / (mB**2 - mV**2) * ff['T3'] )  # second summand of (3.30) 

    transversity_amps = {
        'perp_L': sqrt(2 * lambda_b) * ( ( (wc_9 + wc_9p) - (wc_10 + wc_10p)) * ff['V']/(mB + mV) + 2 * mqh / q2 * (wc_7 + wc_7p) * ff['T1'] ),  # (3.28)
        'perp_R': sqrt(2 * lambda_b) * ( ( (wc_9 + wc_9p) + (wc_10 + wc_10p)) * ff['V']/(mB + mV) + 2 * mqh / q2 * (wc_7 + wc_7p) * ff['T1'] ),  # (3.28)
        'para_L': -sqrt(2) * (mB**2 - mV**2) * ( ((wc_9 - wc_9p) - (wc_10 - wc_10p)) * ff['A1']/(mB - mV) + 2 * mqh / q2 * (wc_7 - wc_7p) * ff['T2'] ),  # (3.29)
        'para_R': -sqrt(2) * (mB**2 - mV**2) * ( ((wc_9 - wc_9p) + (wc_10 - wc_10p)) * ff['A1']/(mB - mV) + 2 * mqh / q2 * (wc_7 - wc_7p) * ff['T2'] ),  # (3.29)
        '0_L': - 1 / (2 * mV * sqrt(q2)) * ( a0_l_term_one + a0_term_two ),  # (3.30)
        '0_R': - 1 / (2 * mV * sqrt(q2)) * ( a0_r_term_one + a0_term_two ),  # (3.30)
        't': sqrt(lambda_b / q2) * ff['A0'] * ( 2 * (wc_10 - wc_10p) + q2 / ml1 * (wc_p - wc_pp) ),  # (3.31)  
        'S': - 2 * sqrt(lambda_b) * (wc['s'] - wc['sp']) * ff['A0'],  # (3.32)
    }

    return {k: prefactor * v for k, v in transversity_amps.items()}

def helicity_amps_v(q2, mB, mV, mqh, mql, ml1, ml2, ff, wc, prefactor):
    laB = lambda_K(mB**2, mV**2, q2)
    H = {}
    H['0','V'] = (4 * 1j * mB * mV)/(sqrt(q2) * (mB+mV)) * ((wc['v']-wc['vp']) * (mB+mV) * ff['A12']+mqh * (wc['7']-wc['7p']) * ff['T23'])
    H['0','A'] = 4 * 1j * mB * mV/sqrt(q2) * (wc['a']-wc['ap']) * ff['A12']
    H['pl','V'] = 1j/(2 * (mB+mV)) * (+(wc['v']+wc['vp']) * sqrt(laB) * ff['V']-(mB+mV)**2 * (wc['v']-wc['vp']) * ff['A1'])+1j * mqh/q2 * (+(wc['7']+wc['7p']) * sqrt(laB) * ff['T1']-(wc['7']-wc['7p']) * (mB**2-mV**2) * ff['T2'])
    H['mi','V'] = 1j/(2 * (mB+mV)) * (-(wc['v']+wc['vp']) * sqrt(laB) * ff['V']-(mB+mV)**2 * (wc['v']-wc['vp']) * ff['A1'])+1j * mqh/q2 * (-(wc['7']+wc['7p']) * sqrt(laB) * ff['T1']-(wc['7']-wc['7p']) * (mB**2-mV**2) * ff['T2'])
    H['pl','A'] = 1j/(2 * (mB+mV)) * (+(wc['a']+wc['ap']) * sqrt(laB) * ff['V']-(mB+mV)**2 * (wc['a']-wc['ap']) * ff['A1'])
    H['mi','A'] = 1j/(2 * (mB+mV)) * (-(wc['a']+wc['ap']) * sqrt(laB) * ff['V']-(mB+mV)**2 * (wc['a']-wc['ap']) * ff['A1'])
    H['P'] = 1j * sqrt(laB)/2 * ((wc['p']-wc['pp'])/(mqh+mql)+(ml1+ml2)/q2 * (wc['a']-wc['ap'])) * ff['A0']
    H['S'] = 1j * sqrt(laB)/2 * ((wc['s']-wc['sp'])/(mqh+mql)+(ml1-ml2)/q2 * (wc['v']-wc['vp'])) * ff['A0']
    H['0','T'] = 2 * sqrt(2) * mB * mV/(mB+mV) * (wc['t']+wc['tp']) * ff['T23']
    H['0','Tt'] = 2 * mB * mV/(mB+mV) * (wc['t']-wc['tp']) * ff['T23']
    H['pl','T'] = 1/(sqrt(2) * sqrt(q2)) * (+(wc['t']-wc['tp']) * sqrt(laB) * ff['T1']-(wc['t']+wc['tp']) * (mB**2-mV**2) * ff['T2'])
    H['mi','T'] = 1/(sqrt(2) * sqrt(q2)) * (-(wc['t']-wc['tp']) * sqrt(laB) * ff['T1']-(wc['t']+wc['tp']) * (mB**2-mV**2) * ff['T2'])
    H['pl','Tt'] = 1/(2 * sqrt(q2)) * (+(wc['t']+wc['tp']) * sqrt(laB) * ff['T1']-(wc['t']-wc['tp']) * (mB**2-mV**2) * ff['T2'])
    H['mi','Tt'] = 1/(2 * sqrt(q2)) * (-(wc['t']+wc['tp']) * sqrt(laB) * ff['T1']-(wc['t']-wc['tp']) * (mB**2-mV**2) * ff['T2'])
    return {k: prefactor*v for k, v in H.items()}

def _Re(z):
    return z.real
def _Im(z):
    return z.imag
def _Co(z):
    return complex(z).conjugate()

def angularcoeffs_general_Gbasis_v(H, q2, mB, mV, mqh, mql, ml1, ml2):
    laB = lambda_K(mB**2, mV**2, q2)
    laGa = lambda_K(q2, ml1**2, ml2**2)
    E1 = sqrt(ml1**2+laGa/(4 * q2))
    E2 = sqrt(ml2**2+laGa/(4 * q2))
    CH = {k: complex(v).conjugate() for k, v in H.items()}
    G = {}
    G[0,0,0] = (
         4/9 * (3 * E1 * E2+laGa/(4 * q2)) * (abs(H['pl','V'])**2+abs(H['mi','V'])**2+abs(H['0','V'])**2+abs(H['pl','A'])**2+abs(H['mi','A'])**2+abs(H['0','A'])**2)
         +4 * ml1 * ml2/3 * (abs(H['pl','V'])**2+abs(H['mi','V'])**2+abs(H['0','V'])**2-abs(H['pl','A'])**2-abs(H['mi','A'])**2-abs(H['0','A'])**2)
         +4/3 * (E1 * E2-ml1 * ml2+laGa/(4 * q2)) * abs(H['S'])**2+4/3 * (E1 * E2+ml1 * ml2+laGa/(4 * q2)) * abs(H['P'])**2
         +16/9 * (3 * (E1 * E2+ml1 * ml2)-laGa/(4 * q2)) * (abs(H['pl','Tt'])**2+abs(H['mi','Tt'])**2+abs(H['0','Tt'])**2)
         +8/9 * (3 * (E1 * E2-ml1 * ml2)-laGa/(4 * q2)) * (abs(H['pl','T'])**2+abs(H['mi','T'])**2+abs(H['0','T'])**2)
         +16/3 * (ml1 * E2+ml2 * E1) * _Im(H['pl','V'] * CH['pl','Tt']+H['mi','V'] * CH['mi','Tt']+H['0','V'] * CH['0','Tt'])
         +8 * sqrt(2)/3 * (ml1 * E2-ml2 * E1) * _Im(H['pl','A'] * CH['pl','T']+H['mi','A'] * CH['mi','T']+H['0','A'] * CH['0','T']))
    G[0,1,0] = (4 * sqrt(laGa)/3 * (
        _Re(H['pl','V'] * CH['pl','A']-H['mi','V'] * CH['mi','A'])
        +2 * sqrt(2)/q2 * (ml1**2-ml2**2) * _Re(H['pl','T'] * CH['pl','Tt']-H['mi','T'] * CH['mi','Tt'])
        +2 * (ml1+ml2)/sqrt(q2) * _Im(H['pl','A'] * CH['pl','Tt']-H['mi','A'] * CH['mi','Tt'])
        +sqrt(2)*(ml1-ml2)/sqrt(q2) * _Im(H['pl','V'] * CH['pl','T']-H['mi','V'] * CH['mi','T'])
        -(ml1-ml2)/sqrt(q2) * _Re(H['0','A'] * CH['P'])-(ml1+ml2)/sqrt(q2) * _Re(H['0','V'] * CH['S'])
        +_Im(sqrt(2) * H['0','T'] * CH['P']+2 * H['0','Tt'] * CH['S'])
        ))
    G[0,2,0] = -2/9 * laGa/q2 * (
    -abs(H['pl','V'])**2-abs(H['mi','V'])**2+2 * abs(H['0','V'])**2-abs(H['pl','A'])**2-abs(H['mi','A'])**2+2 * abs(H['0','A'])**2
    -2 * (-abs(H['pl','T'])**2-abs(H['mi','T'])**2+2 * abs(H['0','T'])**2)-4 * (-abs(H['pl','Tt'])**2-abs(H['mi','Tt'])**2+2 * abs(H['0','Tt'])**2))
    G[2,0,0] = (-4/9 * (3 * E1 * E2+laGa/(4 * q2)) * (abs(H['pl','V'])**2+abs(H['mi','V'])**2-2 * abs(H['0','V'])**2+abs(H['pl','A'])**2+abs(H['mi','A'])**2
    -2 * abs(H['0','A'])**2)-4 * ml1 * ml2/3 * (abs(H['pl','V'])**2+abs(H['mi','V'])**2-2 * abs(H['0','V'])**2-abs(H['pl','A'])**2
    -abs(H['mi','A'])**2+2 * abs(H['0','A'])**2)+8/3 * (E1 * E2-ml1 * ml2+laGa/(4 * q2)) * abs(H['S'])**2
    +8/3 * (E1 * E2+ml1 * ml2+laGa/(4 * q2)) * abs(H['P'])**2
    -16/9 * (3 * (E1 * E2+ml1 * ml2)-laGa/(4 * q2)) * (abs(H['pl','Tt'])**2+abs(H['mi','Tt'])**2-2 * abs(H['0','Tt'])**2)
    -8/9 * (3 * (E1 * E2-ml1 * ml2)-laGa/(4 * q2)) * (abs(H['pl','T'])**2+abs(H['mi','T'])**2-2 * abs(H['0','T'])**2)
    -16/3 * (ml1 * E2+ml2 * E1) * _Im(H['pl','V'] * CH['pl','Tt']+H['mi','V'] * CH['mi','Tt']-2 * H['0','V'] * CH['0','Tt'])
    -8 * sqrt(2)/3 * (ml1 * E2-ml2 * E1) * _Im(H['pl','A'] * CH['pl','T']+H['mi','A'] * CH['mi','T']-2 * H['0','A'] * CH['0','T']))
    G[2,1,0] = (-4 * sqrt(laGa)/3 * (_Re(H['pl','V'] * CH['pl','A']-H['mi','V'] * CH['mi','A'])
    +2 * sqrt(2) * (ml1**2-ml2**2)/q2 * _Re(H['pl','T'] * CH['pl','Tt']-H['mi','T'] * CH['mi','Tt'])
    +2 * (ml1+ml2)/sqrt(q2) * _Im(H['pl','A'] * CH['pl','Tt']-H['mi','A'] * CH['mi','Tt'])
    +sqrt(2) * (ml1-ml2)/sqrt(q2) * _Im(H['pl','V'] * CH['pl','T']-H['mi','V'] * CH['mi','T'])
    +2 * (ml1-ml2)/sqrt(q2) * _Re(H['0','A'] * CH['P'])+2 * (ml1+ml2)/sqrt(q2) * _Re(H['0','V'] * CH['S'])
    -2 * _Im(sqrt(2) * H['0','T'] * CH['P']+2 * H['0','Tt'] * CH['S'])))
    G[2,2,0] = (-2/9 * laGa/q2 * (abs(H['pl','V'])**2+abs(H['mi','V'])**2+4 * abs(H['0','V'])**2+abs(H['pl','A'])**2+abs(H['mi','A'])**2
    +4 * abs(H['0','A'])**2-2 * (abs(H['pl','T'])**2+abs(H['mi','T'])**2+4 * abs(H['0','T'])**2)-4 * (abs(H['pl','Tt'])**2+abs(H['mi','Tt'])**2+4 * abs(H['0','Tt'])**2)))
    G[2,1,1] = (4/sqrt(3) * sqrt(laGa) * (H['pl','V'] * CH['0','A']+H['pl','A'] * CH['0','V']-H['0','V'] * CH['mi','A']-H['0','A'] * CH['mi','V']
    +(ml1+ml2)/sqrt(q2) * (H['pl','V'] * CH['S']+H['S'] * CH['mi','V'])-sqrt(2) * 1j * (H['P'] * CH['mi','T']-H['pl','T'] * CH['P']
    +sqrt(2)*(H['S'] * CH['mi','Tt']-H['pl','Tt'] * CH['S']))
    +(ml1-ml2)/sqrt(q2) * (H['pl','A'] * CH['P']+H['P'] * CH['mi','A'])
    -2 * 1j * (ml1+ml2)/sqrt(q2) * (H['pl','A'] * CH['0','Tt']+H['0','Tt'] * CH['mi','A']-H['pl','Tt'] * CH['0','A']-H['0','A'] * CH['mi','Tt'])
    -sqrt(2) * 1j * (ml1-ml2)/sqrt(q2) * (H['pl','V'] * CH['0','T']+H['0','T'] * CH['mi','V']-H['pl','T'] * CH['0','V']-H['0','V'] * CH['mi','T'])
    +2 * sqrt(2) * (ml1**2-ml2**2)/q2 * (H['pl','T'] * CH['0','Tt']+H['pl','Tt'] * CH['0','T']-H['0','T'] * CH['mi','Tt']-H['0','Tt'] * CH['mi','T'])))
    G[2,2,1] = (4/3 * laGa/q2 * (H['pl','V'] * CH['0','V']+H['0','V'] * CH['mi','V']+H['pl','A'] * CH['0','A']+H['0','A'] * CH['mi','A']
    -2 * (H['pl','T'] * CH['0','T']+H['0','T'] * CH['mi','T']+2 * (H['pl','Tt'] * CH['0','Tt']+H['0','Tt'] * CH['mi','Tt']))))
    G[2,2,2] = -8/3 * laGa/q2 * (H['pl','V'] * CH['mi','V']+H['pl','A'] * CH['mi','A']-2 * (H['pl','T'] * CH['mi','T']+2 * H['pl','Tt'] * CH['mi','Tt']))
    prefactor = sqrt(laB)*sqrt(laGa)/(2**9 * pi**3 * mB**3 * q2)
    return {k: prefactor*v for k, v in G.items()}

def angularcoeffs_h_Gbasis_v(phi, H, Htilde, q2, mB, mV, mqh, mql, ml1, ml2):
    qp = -cmath.exp(1j * phi) # here it is assumed that q/p is a pure phase, as appropriate for B and Bs mixing
    laB = lambda_K(mB**2, mV**2, q2)
    laGa = lambda_K(q2, ml1**2, ml2**2)
    E1 = sqrt(ml1**2+laGa/(4 * q2))
    E2 = sqrt(ml2**2+laGa/(4 * q2))
    CH = {k: complex(v).conjugate() for k, v in H.items()}
    CHtilde = {k: complex(v).conjugate() for k, v in Htilde.items()}
    G = {}
    G[0,0,0] = (
         4/9 * (3 * E1 * E2+laGa/(4 * q2)) * (2 * _Re(-qp * Htilde['pl','V'] * CH['pl','V'])+2 * _Re(-qp * Htilde['mi','V'] * CH['mi','V'])+2 * _Re(-qp * Htilde['0','V'] * CH['0','V'])+2 * _Re(-qp * Htilde['pl','A'] * CH['pl','A'])+2 * _Re(-qp * Htilde['mi','A'] * CH['mi','A'])+2 * _Re(-qp * Htilde['0','A'] * CH['0','A']))
         +4 * ml1 * ml2/3 * (2 * _Re(-qp * Htilde['pl','V'] * CH['pl','V'])+2 * _Re(-qp * Htilde['mi','V'] * CH['mi','V'])+2 * _Re(-qp * Htilde['0','V'] * CH['0','V'])-2 * _Re(-qp * Htilde['pl','A'] * CH['pl','A'])-2 * _Re(-qp * Htilde['mi','A'] * CH['mi','A'])-2 * _Re(-qp * Htilde['0','A'] * CH['0','A']))
         +4/3 * (E1 * E2-ml1 * ml2+laGa/(4 * q2)) * 2 * _Re(-qp * Htilde['S'] * CH['S'])+4/3 * (E1 * E2+ml1 * ml2+laGa/(4 * q2)) * 2 * _Re(-qp * Htilde['P'] * CH['P'])
         +16/9 * (3 * (E1 * E2+ml1 * ml2)-laGa/(4 * q2)) * (2 * _Re(-qp * Htilde['pl','Tt'] * CH['pl','Tt'])+2 * _Re(-qp * Htilde['mi','Tt'] * CH['mi','Tt'])+2 * _Re(-qp * Htilde['0','Tt'] * CH['0','Tt']))
         +8/9 * (3 * (E1 * E2-ml1 * ml2)-laGa/(4 * q2)) * (2 * _Re(-qp * Htilde['pl','T'] * CH['pl','T'])+2 * _Re(-qp * Htilde['mi','T'] * CH['mi','T'])+2 * _Re(-qp * Htilde['0','T'] * CH['0','T']))
         +16/3 * (ml1 * E2+ml2 * E1) * _Im((-qp * Htilde['pl','V']  * CH['pl','Tt'] + _Co(-qp) * H['pl','V']  * CHtilde['pl','Tt'])+(-qp * Htilde['mi','V']  * CH['mi','Tt'] + _Co(-qp) * H['mi','V']  * CHtilde['mi','Tt'])+(-qp * Htilde['0','V']  * CH['0','Tt'] + _Co(-qp) * H['0','V']  * CHtilde['0','Tt']))
         +8 * sqrt(2)/3 * (ml1 * E2-ml2 * E1) * _Im((-qp * Htilde['pl','A']  * CH['pl','T'] + _Co(-qp) * H['pl','A']  * CHtilde['pl','T'])+(-qp * Htilde['mi','A']  * CH['mi','T'] + _Co(-qp) * H['mi','A']  * CHtilde['mi','T'])+(-qp * Htilde['0','A']  * CH['0','T'] + _Co(-qp) * H['0','A']  * CHtilde['0','T'])))
    G[0,1,0] = (4 * sqrt(laGa)/3 * (
        _Re((-qp * Htilde['pl','V']  * CH['pl','A'] + _Co(-qp) * H['pl','V']  * CHtilde['pl','A'])-(-qp * Htilde['mi','V']  * CH['mi','A'] + _Co(-qp) * H['mi','V']  * CHtilde['mi','A']))
        +2 * sqrt(2)/q2 * (ml1**2-ml2**2) * _Re((-qp * Htilde['pl','T']  * CH['pl','Tt'] + _Co(-qp) * H['pl','T']  * CHtilde['pl','Tt'])-(-qp * Htilde['mi','T']  * CH['mi','Tt'] + _Co(-qp) * H['mi','T']  * CHtilde['mi','Tt']))
        +2 * (ml1+ml2)/sqrt(q2) * _Im((-qp * Htilde['pl','A']  * CH['pl','Tt'] + _Co(-qp) * H['pl','A']  * CHtilde['pl','Tt'])-(-qp * Htilde['mi','A']  * CH['mi','Tt'] + _Co(-qp) * H['mi','A']  * CHtilde['mi','Tt']))
        +sqrt(2)*(ml1-ml2)/sqrt(q2) * _Im((-qp * Htilde['pl','V']  * CH['pl','T'] + _Co(-qp) * H['pl','V']  * CHtilde['pl','T'])-(-qp * Htilde['mi','V']  * CH['mi','T'] + _Co(-qp) * H['mi','V']  * CHtilde['mi','T']))
        -(ml1-ml2)/sqrt(q2) * _Re((-qp * Htilde['0','A']  * CH['P'] + _Co(-qp) * H['0','A']  * CHtilde['P']))-(ml1+ml2)/sqrt(q2) * _Re((-qp * Htilde['0','V']  * CH['S'] + _Co(-qp) * H['0','V']  * CHtilde['S']))
        +_Im(sqrt(2) * (-qp * Htilde['0','T']  * CH['P'] + _Co(-qp) * H['0','T']  * CHtilde['P'])+2 * (-qp * Htilde['0','Tt']  * CH['S'] + _Co(-qp) * H['0','Tt']  * CHtilde['S']))
        ))
    G[0,2,0] = -2/9 * laGa/q2 * (
    -2 * _Re(-qp * Htilde['pl','V'] * CH['pl','V'])-2 * _Re(-qp * Htilde['mi','V'] * CH['mi','V'])+2 * 2 * _Re(-qp * Htilde['0','V'] * CH['0','V'])-2 * _Re(-qp * Htilde['pl','A'] * CH['pl','A'])-2 * _Re(-qp * Htilde['mi','A'] * CH['mi','A'])+2 * 2 * _Re(-qp * Htilde['0','A'] * CH['0','A'])
    -2 * (-2 * _Re(-qp * Htilde['pl','T'] * CH['pl','T'])-2 * _Re(-qp * Htilde['mi','T'] * CH['mi','T'])+2 * 2 * _Re(-qp * Htilde['0','T'] * CH['0','T']))-4 * (-2 * _Re(-qp * Htilde['pl','Tt'] * CH['pl','Tt'])-2 * _Re(-qp * Htilde['mi','Tt'] * CH['mi','Tt'])+2 * 2 * _Re(-qp * Htilde['0','Tt'] * CH['0','Tt'])))
    G[2,0,0] = (-4/9 * (3 * E1 * E2+laGa/(4 * q2)) * (2 * _Re(-qp * Htilde['pl','V'] * CH['pl','V'])+2 * _Re(-qp * Htilde['mi','V'] * CH['mi','V'])-2 * 2 * _Re(-qp * Htilde['0','V'] * CH['0','V'])+2 * _Re(-qp * Htilde['pl','A'] * CH['pl','A'])+2 * _Re(-qp * Htilde['mi','A'] * CH['mi','A'])
    -2 * 2 * _Re(-qp * Htilde['0','A'] * CH['0','A']))-4 * ml1 * ml2/3 * (2 * _Re(-qp * Htilde['pl','V'] * CH['pl','V'])+2 * _Re(-qp * Htilde['mi','V'] * CH['mi','V'])-2 * 2 * _Re(-qp * Htilde['0','V'] * CH['0','V'])-2 * _Re(-qp * Htilde['pl','A'] * CH['pl','A'])
    -2 * _Re(-qp * Htilde['mi','A'] * CH['mi','A'])+2 * 2 * _Re(-qp * Htilde['0','A'] * CH['0','A']))+8/3 * (E1 * E2-ml1 * ml2+laGa/(4 * q2)) * 2 * _Re(-qp * Htilde['S'] * CH['S'])
    +8/3 * (E1 * E2+ml1 * ml2+laGa/(4 * q2)) * 2 * _Re(-qp * Htilde['P'] * CH['P'])
    -16/9 * (3 * (E1 * E2+ml1 * ml2)-laGa/(4 * q2)) * (2 * _Re(-qp * Htilde['pl','Tt'] * CH['pl','Tt'])+2 * _Re(-qp * Htilde['mi','Tt'] * CH['mi','Tt'])-2 * 2 * _Re(-qp * Htilde['0','Tt'] * CH['0','Tt']))
    -8/9 * (3 * (E1 * E2-ml1 * ml2)-laGa/(4 * q2)) * (2 * _Re(-qp * Htilde['pl','T'] * CH['pl','T'])+2 * _Re(-qp * Htilde['mi','T'] * CH['mi','T'])-2 * 2 * _Re(-qp * Htilde['0','T'] * CH['0','T']))
    -16/3 * (ml1 * E2+ml2 * E1) * _Im((-qp * Htilde['pl','V']  * CH['pl','Tt'] + _Co(-qp) * H['pl','V']  * CHtilde['pl','Tt'])+(-qp * Htilde['mi','V']  * CH['mi','Tt'] + _Co(-qp) * H['mi','V']  * CHtilde['mi','Tt'])-2 * (-qp * Htilde['0','V']  * CH['0','Tt'] + _Co(-qp) * H['0','V']  * CHtilde['0','Tt']))
    -8 * sqrt(2)/3 * (ml1 * E2-ml2 * E1) * _Im((-qp * Htilde['pl','A']  * CH['pl','T'] + _Co(-qp) * H['pl','A']  * CHtilde['pl','T'])+(-qp * Htilde['mi','A']  * CH['mi','T'] + _Co(-qp) * H['mi','A']  * CHtilde['mi','T'])-2 * (-qp * Htilde['0','A']  * CH['0','T'] + _Co(-qp) * H['0','A']  * CHtilde['0','T'])))

    G[2,1,0] = (-4 * sqrt(laGa)/3 * (_Re((-qp * Htilde['pl','V'] * CH['pl','A'] + _Co(-qp) * H['pl','V'] * CHtilde['pl','A']) - (-qp * Htilde['mi','V']  * CH['mi','A'] + _Co(-qp) * H['mi','V']  * CHtilde['mi','A']))
    +2 * sqrt(2) * (ml1**2-ml2**2)/q2 * _Re((-qp * Htilde['pl','T']  * CH['pl','Tt'] + _Co(-qp) * H['pl','T']  * CHtilde['pl','Tt'])-(-qp * Htilde['mi','T']  * CH['mi','Tt'] + _Co(-qp) * H['mi','T']  * CHtilde['mi','Tt']))
    +2 * (ml1+ml2)/sqrt(q2) * _Im((-qp * Htilde['pl','A']  * CH['pl','Tt'] + _Co(-qp) * H['pl','A']  * CHtilde['pl','Tt'])-(-qp * Htilde['mi','A']  * CH['mi','Tt'] + _Co(-qp) * H['mi','A']  * CHtilde['mi','Tt']))
    +sqrt(2) * (ml1-ml2)/sqrt(q2) * _Im((-qp * Htilde['pl','V']  * CH['pl','T'] + _Co(-qp) * H['pl','V']  * CHtilde['pl','T'])-(-qp * Htilde['mi','V']  * CH['mi','T'] + _Co(-qp) * H['mi','V']  * CHtilde['mi','T']))
    +2 * (ml1-ml2)/sqrt(q2) * _Re((-qp * Htilde['0','A']  * CH['P'] + _Co(-qp) * H['0','A']  * CHtilde['P']))+2 * (ml1+ml2)/sqrt(q2) * _Re((-qp * Htilde['0','V']  * CH['S'] + _Co(-qp) * H['0','V']  * CHtilde['S']))
    -2 * _Im(sqrt(2) * (-qp * Htilde['0','T']  * CH['P'] + _Co(-qp) * H['0','T']  * CHtilde['P'])+2 * (-qp * Htilde['0','Tt']  * CH['S'] + _Co(-qp) * H['0','Tt']  * CHtilde['S']))))
    # G[2, 1, 0] = -4 * sqrt(laGa) / 3 * ( 
    #     _Re( H['pl', 'V'] * CHtilde['pl', 'A'] - H['mi', 'V'] * CHtilde['mi', 'A'] )
    #     + 2 * sqrt(2) * (ml1**2 - ml2**2) / q2 * _Im( H['pl', 'T'] * CHtilde['pl', 'Tt'] - H['mi', 'T'] * CHtilde['mi', 'Tt'] )  # == 0
    #     + 2 * (ml1 + ml2) / sqrt(q2) * _Im( H['pl', 'A'] * CHtilde['pl', 'Tt'] - H['mi', 'A'] * CHtilde['mi', 'Tt'] )  
    #     + sqrt(2) * (ml1 - ml2) / sqrt(q2) * _Im( H['pl', 'V'] * CHtilde['pl', 'T'] - H['mi', 'V'] * CHtilde['mi', 'T'] )  # == 0
    #     + 2 * (ml1 - ml2) / sqrt(q2) * _Re( H['0', 'A'] * CHtilde['P'] )  # == 0
    #     + 2 * (ml1 + ml2) / sqrt(q2) * _Re( H['0', 'V'] * CHtilde['S'] )
    #     - 2 * _Im( sqrt(2) * H['0', 'T'] * CHtilde['P'] + 2 * H['0', 'Tt'] * CHtilde['S'] )
    # )

    G[2,2,0] = (-2/9 * laGa/q2 * (2 * _Re(-qp * Htilde['pl','V'] * CH['pl','V'])+2 * _Re(-qp * Htilde['mi','V'] * CH['mi','V'])+4 * 2 * _Re(-qp * Htilde['0','V'] * CH['0','V'])+2 * _Re(-qp * Htilde['pl','A'] * CH['pl','A'])+2 * _Re(-qp * Htilde['mi','A'] * CH['mi','A'])
    +4 * 2 * _Re(-qp * Htilde['0','A'] * CH['0','A'])-2 * (2 * _Re(-qp * Htilde['pl','T'] * CH['pl','T'])+2 * _Re(-qp * Htilde['mi','T'] * CH['mi','T'])+4 * 2 * _Re(-qp * Htilde['0','T'] * CH['0','T']))-4 * (2 * _Re(-qp * Htilde['pl','Tt'] * CH['pl','Tt'])+2 * _Re(-qp * Htilde['mi','Tt'] * CH['mi','Tt'])+4 * 2 * _Re(-qp * Htilde['0','Tt'] * CH['0','Tt']))))
    
    G[2,1,1] = (4/sqrt(3) * sqrt(laGa) * ((-qp * Htilde['pl','V']  * CH['0','A'] + _Co(-qp) * H['pl','V']  * CHtilde['0','A'])+(-qp * Htilde['pl','A']  * CH['0','V'] + _Co(-qp) * H['pl','A']  * CHtilde['0','V'])-(-qp * Htilde['0','V']  * CH['mi','A'] + _Co(-qp) * H['0','V']  * CHtilde['mi','A'])-(-qp * Htilde['0','A']  * CH['mi','V'] + _Co(-qp) * H['0','A']  * CHtilde['mi','V'])
    +(ml1+ml2)/sqrt(q2) * ((-qp * Htilde['pl','V']  * CH['S'] + _Co(-qp) * H['pl','V']  * CHtilde['S'])+(-qp * Htilde['S']  * CH['mi','V'] + _Co(-qp) * H['S']  * CHtilde['mi','V']))-sqrt(2) * 1j * ((-qp * Htilde['P']  * CH['mi','T'] + _Co(-qp) * H['P']  * CHtilde['mi','T'])-(-qp * Htilde['pl','T']  * CH['P'] + _Co(-qp) * H['pl','T']  * CHtilde['P'])
    +sqrt(2)*((-qp * Htilde['S']  * CH['mi','Tt'] + _Co(-qp) * H['S']  * CHtilde['mi','Tt'])-(-qp * Htilde['pl','Tt']  * CH['S'] + _Co(-qp) * H['pl','Tt']  * CHtilde['S'])))
    +(ml1-ml2)/sqrt(q2) * ((-qp * Htilde['pl','A']  * CH['P'] + _Co(-qp) * H['pl','A']  * CHtilde['P'])+(-qp * Htilde['P']  * CH['mi','A'] + _Co(-qp) * H['P']  * CHtilde['mi','A']))
    -2 * 1j * (ml1+ml2)/sqrt(q2) * ((-qp * Htilde['pl','A']  * CH['0','Tt'] + _Co(-qp) * H['pl','A']  * CHtilde['0','Tt'])+(-qp * Htilde['0','Tt']  * CH['mi','A'] + _Co(-qp) * H['0','Tt']  * CHtilde['mi','A'])-(-qp * Htilde['pl','Tt']  * CH['0','A'] + _Co(-qp) * H['pl','Tt']  * CHtilde['0','A'])-(-qp * Htilde['0','A']  * CH['mi','Tt'] + _Co(-qp) * H['0','A']  * CHtilde['mi','Tt']))
    -sqrt(2) * 1j * (ml1-ml2)/sqrt(q2) * ((-qp * Htilde['pl','V']  * CH['0','T'] + _Co(-qp) * H['pl','V']  * CHtilde['0','T'])+(-qp * Htilde['0','T']  * CH['mi','V'] + _Co(-qp) * H['0','T']  * CHtilde['mi','V'])-(-qp * Htilde['pl','T']  * CH['0','V'] + _Co(-qp) * H['pl','T']  * CHtilde['0','V'])-(-qp * Htilde['0','V']  * CH['mi','T'] + _Co(-qp) * H['0','V']  * CHtilde['mi','T']))
    +2 * sqrt(2) * (ml1**2-ml2**2)/q2 * ((-qp * Htilde['pl','T']  * CH['0','Tt'] + _Co(-qp) * H['pl','T']  * CHtilde['0','Tt'])+(-qp * Htilde['pl','Tt']  * CH['0','T'] + _Co(-qp) * H['pl','Tt']  * CHtilde['0','T'])-(-qp * Htilde['0','T']  * CH['mi','Tt'] + _Co(-qp) * H['0','T']  * CHtilde['mi','Tt'])-(-qp * Htilde['0','Tt']  * CH['mi','T'] + _Co(-qp) * H['0','Tt']  * CHtilde['mi','T']))))

    G[2,2,1] = (4/3 * laGa/q2 * ((-qp * Htilde['pl','V']  * CH['0','V'] + _Co(-qp) * H['pl','V']  * CHtilde['0','V'])+(-qp * Htilde['0','V']  * CH['mi','V'] + _Co(-qp) * H['0','V']  * CHtilde['mi','V'])+(-qp * Htilde['pl','A']  * CH['0','A'] + _Co(-qp) * H['pl','A']  * CHtilde['0','A'])+(-qp * Htilde['0','A']  * CH['mi','A'] + _Co(-qp) * H['0','A']  * CHtilde['mi','A'])
    -2 * ((-qp * Htilde['pl','T']  * CH['0','T'] + _Co(-qp) * H['pl','T']  * CHtilde['0','T'])+(-qp * Htilde['0','T']  * CH['mi','T'] + _Co(-qp) * H['0','T']  * CHtilde['mi','T'])+2 * ((-qp * Htilde['pl','Tt']  * CH['0','Tt'] + _Co(-qp) * H['pl','Tt']  * CHtilde['0','Tt'])+(-qp * Htilde['0','Tt']  * CH['mi','Tt'] + _Co(-qp) * H['0','Tt']  * CHtilde['mi','Tt'])))))
    G[2,2,2] = -8/3 * laGa/q2 * ((-qp * Htilde['pl','V']  * CH['mi','V'] + _Co(-qp) * H['pl','V']  * CHtilde['mi','V'])+(-qp * Htilde['pl','A']  * CH['mi','A'] + _Co(-qp) * H['pl','A']  * CHtilde['mi','A'])-2 * ((-qp * Htilde['pl','T']  * CH['mi','T'] + _Co(-qp) * H['pl','T']  * CHtilde['mi','T'])+2 * (-qp * Htilde['pl','Tt']  * CH['mi','Tt'] + _Co(-qp) * H['pl','Tt']  * CHtilde['mi','Tt'])))
    prefactor = sqrt(laB)*sqrt(laGa)/(2**9 * pi**3 * mB**3 * q2)
    return {k: prefactor*v for k, v in G.items()}


def angularcoeffs_s_Gbasis_v(phi, H, Htilde, q2, mB, mV, mqh, mql, ml1, ml2):
    qp = -cmath.exp(1j * phi) # here it is assumed that q/p is a pure phase, as appropriate for B and Bs mixing
    laB = lambda_K(mB**2, mV**2, q2)
    laGa = lambda_K(q2, ml1**2, ml2**2)
    E1 = sqrt(ml1**2+laGa/(4 * q2))
    E2 = sqrt(ml2**2+laGa/(4 * q2))
    CH = {k: complex(v).conjugate() for k, v in H.items()}
    CHtilde = {k: complex(v).conjugate() for k, v in Htilde.items()}
    G = {}
    G[0,0,0] = (
        # for parts coming from moduli, exchange _Re with _Im
         4/9 * (3 * E1 * E2+laGa/(4 * q2)) * (2 * _Im(-qp * Htilde['pl','V'] * CH['pl','V'])
                                              +2 * _Im(-qp * Htilde['mi','V'] * CH['mi','V'])
                                              +2 * _Im(-qp * Htilde['0','V'] * CH['0','V'])
                                              +2 * _Im(-qp * Htilde['pl','A'] * CH['pl','A'])
                                              +2 * _Im(-qp * Htilde['mi','A'] * CH['mi','A'])
                                              +2 * _Im(-qp * Htilde['0','A'] * CH['0','A']))
         +4 * ml1 * ml2/3 * (2 * _Im(-qp * Htilde['pl','V'] * CH['pl','V'])
                             +2 * _Im(-qp * Htilde['mi','V'] * CH['mi','V'])
                             +2 * _Im(-qp * Htilde['0','V'] * CH['0','V'])
                             -2 * _Im(-qp * Htilde['pl','A'] * CH['pl','A'])
                             -2 * _Im(-qp * Htilde['mi','A'] * CH['mi','A'])
                             -2 * _Im(-qp * Htilde['0','A'] * CH['0','A']))
         +4/3 * (E1 * E2-ml1 * ml2+laGa/(4 * q2)) * 2 * _Im(-qp * Htilde['S'] * CH['S'])
         +4/3 * (E1 * E2+ml1 * ml2+laGa/(4 * q2)) * 2 * _Im(-qp * Htilde['P'] * CH['P'])
         +16/9 * (3 * (E1 * E2+ml1 * ml2)-laGa/(4 * q2)) * (2 * _Im(-qp * Htilde['pl','Tt'] * CH['pl','Tt'])
                                                            +2 * _Im(-qp * Htilde['mi','Tt'] * CH['mi','Tt'])
                                                            +2 * _Im(-qp * Htilde['0','Tt'] * CH['0','Tt']))
         +8/9 * (3 * (E1 * E2-ml1 * ml2)-laGa/(4 * q2)) * (2 * _Im(-qp * Htilde['pl','T'] * CH['pl','T'])
                                                           +2 * _Im(-qp * Htilde['mi','T'] * CH['mi','T'])
                                                           +2 * _Im(-qp * Htilde['0','T'] * CH['0','T']))
         # the parts coming from real and imaginary parts instead
         # first + _Co -> - _Co, then _Im -> (-1) * _Re
         +16/3 * (ml1 * E2+ml2 * E1) * (-1) * _Re((-qp * Htilde['pl','V']  * CH['pl','Tt'] - _Co(-qp) * H['pl','V']  * CHtilde['pl','Tt'])
                                           +(-qp * Htilde['mi','V']  * CH['mi','Tt'] - _Co(-qp) * H['mi','V']  * CHtilde['mi','Tt'])
                                           +(-qp * Htilde['0','V']  * CH['0','Tt'] - _Co(-qp) * H['0','V']  * CHtilde['0','Tt']))
         +8 * sqrt(2)/3 * (ml1 * E2-ml2 * E1) * (-1) * _Re((-qp * Htilde['pl','A']  * CH['pl','T'] - _Co(-qp) * H['pl','A']  * CHtilde['pl','T'])
                                                    +(-qp * Htilde['mi','A']  * CH['mi','T'] - _Co(-qp) * H['mi','A']  * CHtilde['mi','T'])
                                                    +(-qp * Htilde['0','A']  * CH['0','T'] - _Co(-qp) * H['0','A']  * CHtilde['0','T'])))
    # change sign in front of every _Co, then swap -Re -> _Im and _Im with -1* _Re
    G[0,1,0] = (4 * sqrt(laGa)/3 * (
        _Im((-qp * Htilde['pl','V']  * CH['pl','A'] - _Co(-qp) * H['pl','V']  * CHtilde['pl','A'])
            -(-qp * Htilde['mi','V']  * CH['mi','A'] - _Co(-qp) * H['mi','V']  * CHtilde['mi','A']))
        +2 * sqrt(2)/q2 * (ml1**2-ml2**2) * _Im((-qp * Htilde['pl','T']  * CH['pl','Tt'] - _Co(-qp) * H['pl','T']  * CHtilde['pl','Tt'])
                                                -(-qp * Htilde['mi','T']  * CH['mi','Tt'] - _Co(-qp) * H['mi','T']  * CHtilde['mi','Tt']))
        +2 * (ml1+ml2)/sqrt(q2) * (-1) * _Re((-qp * Htilde['pl','A']  * CH['pl','Tt'] - _Co(-qp) * H['pl','A']  * CHtilde['pl','Tt'])
                                      -(-qp * Htilde['mi','A']  * CH['mi','Tt'] - _Co(-qp) * H['mi','A']  * CHtilde['mi','Tt']))
        +sqrt(2)*(ml1-ml2)/sqrt(q2) * (-1) * _Re((-qp * Htilde['pl','V']  * CH['pl','T'] - _Co(-qp) * H['pl','V']  * CHtilde['pl','T'])
                                          -(-qp * Htilde['mi','V']  * CH['mi','T'] - _Co(-qp) * H['mi','V']  * CHtilde['mi','T']))
        -(ml1-ml2)/sqrt(q2) * _Im((-qp * Htilde['0','A']  * CH['P'] - _Co(-qp) * H['0','A']  * CHtilde['P']))
        -(ml1+ml2)/sqrt(q2) * _Im((-qp * Htilde['0','V']  * CH['S'] - _Co(-qp) * H['0','V']  * CHtilde['S']))
        + (-1) * _Re(sqrt(2) * (-qp * Htilde['0','T']  * CH['P'] - _Co(-qp) * H['0','T']  * CHtilde['P'])
             +2 * (-qp * Htilde['0','Tt']  * CH['S'] - _Co(-qp) * H['0','Tt']  * CHtilde['S']))
        ))
    # from _Re to _Im since they all come from modules sqaured
    G[0,2,0] = -2/9 * laGa/q2 * (
                                -2 * _Im(-qp * Htilde['pl','V'] * CH['pl','V'])
                                -2 * _Im(-qp * Htilde['mi','V'] * CH['mi','V'])
                                +2 * 2 * _Im(-qp * Htilde['0','V'] * CH['0','V'])
                                -2 * _Im(-qp * Htilde['pl','A'] * CH['pl','A'])
                                -2 * _Im(-qp * Htilde['mi','A'] * CH['mi','A'])
                                +2 * 2 * _Im(-qp * Htilde['0','A'] * CH['0','A'])
    -2 * (-2 * _Im(-qp * Htilde['pl','T'] * CH['pl','T'])
          -2 * _Im(-qp * Htilde['mi','T'] * CH['mi','T'])
          +2 * 2 * _Im(-qp * Htilde['0','T'] * CH['0','T']))
    -4 * (-2 * _Im(-qp * Htilde['pl','Tt'] * CH['pl','Tt'])
          -2 * _Im(-qp * Htilde['mi','Tt'] * CH['mi','Tt'])
          +2 * 2 * _Im(-qp * Htilde['0','Tt'] * CH['0','Tt'])))
    # for bits linked to moduli squared, _Re->_Im
    G[2,0,0] = (-4/9 * (3 * E1 * E2+laGa/(4 * q2)) * (2 * _Im(-qp * Htilde['pl','V'] * CH['pl','V'])
                                                      +2 * _Im(-qp * Htilde['mi','V'] * CH['mi','V'])
                                                      -2 * 2 * _Im(-qp * Htilde['0','V'] * CH['0','V'])
                                                      +2 * _Im(-qp * Htilde['pl','A'] * CH['pl','A'])
                                                      +2 * _Im(-qp * Htilde['mi','A'] * CH['mi','A'])
                                                      -2 * 2 * _Im(-qp * Htilde['0','A'] * CH['0','A']))
                -4 * ml1 * ml2/3 * (2 * _Im(-qp * Htilde['pl','V'] * CH['pl','V'])
                                    +2 * _Im(-qp * Htilde['mi','V'] * CH['mi','V'])
                                    -2 * 2 * _Im(-qp * Htilde['0','V'] * CH['0','V'])
                                    -2 * _Im(-qp * Htilde['pl','A'] * CH['pl','A'])
                                    -2 * _Im(-qp * Htilde['mi','A'] * CH['mi','A'])
                                    +2 * 2 * _Im(-qp * Htilde['0','A'] * CH['0','A']))
                +8/3 * (E1 * E2-ml1 * ml2+laGa/(4 * q2)) * 2 * _Im(-qp * Htilde['S'] * CH['S'])
                +8/3 * (E1 * E2+ml1 * ml2+laGa/(4 * q2)) * 2 * _Im(-qp * Htilde['P'] * CH['P'])
                -16/9 * (3 * (E1 * E2+ml1 * ml2)-laGa/(4 * q2)) * (2 * _Im(-qp * Htilde['pl','Tt'] * CH['pl','Tt'])
                                                                   +2 * _Im(-qp * Htilde['mi','Tt'] * CH['mi','Tt'])
                                                                   -2 * 2 * _Im(-qp * Htilde['0','Tt'] * CH['0','Tt']))
                -8/9 * (3 * (E1 * E2-ml1 * ml2)-laGa/(4 * q2)) * (2 * _Im(-qp * Htilde['pl','T'] * CH['pl','T'])
                                                                  +2 * _Im(-qp * Htilde['mi','T'] * CH['mi','T'])
                                                                  -2 * 2 * _Im(-qp * Htilde['0','T'] * CH['0','T']))
                # Then change from _Co to - _Co, then _Im to (-1) * _Re
                -16/3 * (ml1 * E2+ml2 * E1) * (-1) * _Re((-qp * Htilde['pl','V']  * CH['pl','Tt'] - _Co(-qp) * H['pl','V']  * CHtilde['pl','Tt'])
                                                  +(-qp * Htilde['mi','V']  * CH['mi','Tt'] - _Co(-qp) * H['mi','V']  * CHtilde['mi','Tt'])
                                                  -2 * (-qp * Htilde['0','V']  * CH['0','Tt'] - _Co(-qp) * H['0','V']  * CHtilde['0','Tt']))
                -8 * sqrt(2)/3 * (ml1 * E2-ml2 * E1) * (-1) * _Re((-qp * Htilde['pl','A']  * CH['pl','T'] - _Co(-qp) * H['pl','A']  * CHtilde['pl','T'])
                                                           +(-qp * Htilde['mi','A']  * CH['mi','T'] - _Co(-qp) * H['mi','A']  * CHtilde['mi','T'])
                                                           -2 * (-qp * Htilde['0','A']  * CH['0','T'] - _Co(-qp) * H['0','A']  * CHtilde['0','T'])))
    ## change sign before _Co for _Re and _Im, then _Re->_Im, _Im -> (-1) * _Re
    G[2,1,0] = (-4 * sqrt(laGa)/3 * (_Im((-qp * Htilde['pl','V']  * CH['pl','A'] - _Co(-qp) * H['pl','V']  * CHtilde['pl','A'])
                                         -(-qp * Htilde['mi','V']  * CH['mi','A'] - _Co(-qp) * H['mi','V']  * CHtilde['mi','A']))
    +2 * sqrt(2) * (ml1**2-ml2**2)/q2 * _Im((-qp * Htilde['pl','T']  * CH['pl','Tt'] - _Co(-qp) * H['pl','T']  * CHtilde['pl','Tt'])
                                            -(-qp * Htilde['mi','T']  * CH['mi','Tt'] - _Co(-qp) * H['mi','T']  * CHtilde['mi','Tt']))
    +2 * (ml1+ml2)/sqrt(q2) * (-1) * _Re((-qp * Htilde['pl','A']  * CH['pl','Tt'] - _Co(-qp) * H['pl','A']  * CHtilde['pl','Tt'])
                                  -(-qp * Htilde['mi','A']  * CH['mi','Tt'] - _Co(-qp) * H['mi','A']  * CHtilde['mi','Tt']))
    +sqrt(2) * (ml1-ml2)/sqrt(q2) * (-1) * _Re((-qp * Htilde['pl','V']  * CH['pl','T'] - _Co(-qp) * H['pl','V']  * CHtilde['pl','T'])
                                        -(-qp * Htilde['mi','V']  * CH['mi','T'] - _Co(-qp) * H['mi','V']  * CHtilde['mi','T']))
    +2 * (ml1-ml2)/sqrt(q2) * _Im((-qp * Htilde['0','A']  * CH['P'] - _Co(-qp) * H['0','A']  * CHtilde['P']))
    +2 * (ml1+ml2)/sqrt(q2) * _Im((-qp * Htilde['0','V']  * CH['S'] - _Co(-qp) * H['0','V']  * CHtilde['S']))
    -2 * (-1) * _Re(sqrt(2) * (-qp * Htilde['0','T']  * CH['P'] - _Co(-qp) * H['0','T']  * CHtilde['P'])
                  +2 * (-qp * Htilde['0','Tt']  * CH['S'] - _Co(-qp) * H['0','Tt']  * CHtilde['S']))))
    #Replaced _Re with _Im
    G[2,2,0] = (-2/9 * laGa/q2 * (2 * _Im(-qp * Htilde['pl','V'] * CH['pl','V'])
                                  + 2 * _Im(-qp * Htilde['mi','V'] * CH['mi','V'])
                                  + 4 * 2 * _Im(-qp * Htilde['0','V'] * CH['0','V'])
                                  + 2 * _Im(-qp * Htilde['pl','A'] * CH['pl','A'])
                                  + 2 * _Im(-qp * Htilde['mi','A'] * CH['mi','A']) 
                                  + 4 * 2 * _Im(-qp * Htilde['0','A'] * CH['0','A'])
                                  - 2 * (2 * _Im(-qp * Htilde['pl','T'] * CH['pl','T'])
                                       +2 * _Im(-qp * Htilde['mi','T'] * CH['mi','T'])
                                       +4 * 2 * _Im(-qp * Htilde['0','T'] * CH['0','T']))
                                  - 4 * (2 * _Im(-qp * Htilde['pl','Tt'] * CH['pl','Tt'])
                                       +2 * _Im(-qp * Htilde['mi','Tt'] * CH['mi','Tt'])
                                       +4 * 2 * _Im(-qp * Htilde['0','Tt'] * CH['0','Tt']))))
    ############## The terms below have Im and Re parts applied in the G_s_to_g_s function, changes of overall signs
    ############## and swaps of _Im and _Re are carried out there.
    #_Co to - _Co 
    G[2,1,1] = (4/sqrt(3) * sqrt(laGa) * ((-qp * Htilde['pl','V']  * CH['0','A'] - _Co(-qp) * H['pl','V']  * CHtilde['0','A'])
                                          +(-qp * Htilde['pl','A']  * CH['0','V'] - _Co(-qp) * H['pl','A']  * CHtilde['0','V'])
                                          -(-qp * Htilde['0','V']  * CH['mi','A'] - _Co(-qp) * H['0','V']  * CHtilde['mi','A'])
                                          -(-qp * Htilde['0','A']  * CH['mi','V'] - _Co(-qp) * H['0','A']  * CHtilde['mi','V'])
    +(ml1+ml2)/sqrt(q2) * ((-qp * Htilde['pl','V']  * CH['S'] - _Co(-qp) * H['pl','V']  * CHtilde['S'])
                           +(-qp * Htilde['S']  * CH['mi','V'] - _Co(-qp) * H['S']  * CHtilde['mi','V']))
    -sqrt(2) * 1j * ((-qp * Htilde['P']  * CH['mi','T'] - _Co(-qp) * H['P']  * CHtilde['mi','T'])
                     -(-qp * Htilde['pl','T']  * CH['P'] - _Co(-qp) * H['pl','T']  * CHtilde['P'])
    +sqrt(2)*((-qp * Htilde['S']  * CH['mi','Tt'] - _Co(-qp) * H['S']  * CHtilde['mi','Tt'])
              -(-qp * Htilde['pl','Tt']  * CH['S'] - _Co(-qp) * H['pl','Tt']  * CHtilde['S'])))
    +(ml1-ml2)/sqrt(q2) * ((-qp * Htilde['pl','A']  * CH['P'] - _Co(-qp) * H['pl','A']  * CHtilde['P'])
                           +(-qp * Htilde['P']  * CH['mi','A'] - _Co(-qp) * H['P']  * CHtilde['mi','A']))
    -2 * 1j * (ml1+ml2)/sqrt(q2) * ((-qp * Htilde['pl','A']  * CH['0','Tt'] - _Co(-qp) * H['pl','A']  * CHtilde['0','Tt'])
                                    +(-qp * Htilde['0','Tt']  * CH['mi','A'] - _Co(-qp) * H['0','Tt']  * CHtilde['mi','A'])
                                    -(-qp * Htilde['pl','Tt']  * CH['0','A'] - _Co(-qp) * H['pl','Tt']  * CHtilde['0','A'])
                                    -(-qp * Htilde['0','A']  * CH['mi','Tt'] - _Co(-qp) * H['0','A']  * CHtilde['mi','Tt']))
    -sqrt(2) * 1j * (ml1-ml2)/sqrt(q2) * ((-qp * Htilde['pl','V']  * CH['0','T'] - _Co(-qp) * H['pl','V']  * CHtilde['0','T'])
                                          +(-qp * Htilde['0','T']  * CH['mi','V'] - _Co(-qp) * H['0','T']  * CHtilde['mi','V'])
                                          -(-qp * Htilde['pl','T']  * CH['0','V'] - _Co(-qp) * H['pl','T']  * CHtilde['0','V'])
                                          -(-qp * Htilde['0','V']  * CH['mi','T'] - _Co(-qp) * H['0','V']  * CHtilde['mi','T']))
    +2 * sqrt(2) * (ml1**2-ml2**2)/q2 * ((-qp * Htilde['pl','T']  * CH['0','Tt'] - _Co(-qp) * H['pl','T']  * CHtilde['0','Tt'])
                                         +(-qp * Htilde['pl','Tt']  * CH['0','T'] - _Co(-qp) * H['pl','Tt']  * CHtilde['0','T'])
                                         -(-qp * Htilde['0','T']  * CH['mi','Tt'] - _Co(-qp) * H['0','T']  * CHtilde['mi','Tt'])
                                         -(-qp * Htilde['0','Tt']  * CH['mi','T'] - _Co(-qp) * H['0','Tt']  * CHtilde['mi','T']))))
    #_Co -> - _Co
    G[2,2,1] = (4/3 * laGa/q2 * ((-qp * Htilde['pl','V']  * CH['0','V'] - _Co(-qp) * H['pl','V']  * CHtilde['0','V'])
                                 +(-qp * Htilde['0','V']  * CH['mi','V'] - _Co(-qp) * H['0','V']  * CHtilde['mi','V'])
                                 +(-qp * Htilde['pl','A']  * CH['0','A'] - _Co(-qp) * H['pl','A']  * CHtilde['0','A'])
                                 +(-qp * Htilde['0','A']  * CH['mi','A'] - _Co(-qp) * H['0','A']  * CHtilde['mi','A'])
    -2 * ((-qp * Htilde['pl','T']  * CH['0','T'] - _Co(-qp) * H['pl','T']  * CHtilde['0','T'])
          +(-qp * Htilde['0','T']  * CH['mi','T'] - _Co(-qp) * H['0','T']  * CHtilde['mi','T'])
          +2 * ((-qp * Htilde['pl','Tt']  * CH['0','Tt'] - _Co(-qp) * H['pl','Tt']  * CHtilde['0','Tt'])
                +(-qp * Htilde['0','Tt']  * CH['mi','Tt'] - _Co(-qp) * H['0','Tt']  * CHtilde['mi','Tt'])))))
    #_Co -> - _Co
    G[2,2,2] = -8/3 * laGa/q2 * ((-qp * Htilde['pl','V']  * CH['mi','V'] - _Co(-qp) * H['pl','V']  * CHtilde['mi','V'])+
                                 (-qp * Htilde['pl','A']  * CH['mi','A'] - _Co(-qp) * H['pl','A']  * CHtilde['mi','A'])
                                 -2 * ((-qp * Htilde['pl','T']  * CH['mi','T'] - _Co(-qp) * H['pl','T']  * CHtilde['mi','T'])
                                       +2 * (-qp * Htilde['pl','Tt']  * CH['mi','Tt'] - _Co(-qp) * H['pl','Tt']  * CHtilde['mi','Tt']))
                                 )
    prefactor = sqrt(laB)*sqrt(laGa)/(2**9 * pi**3 * mB**3 * q2)
    return {k: prefactor*v for k, v in G.items()}

def G_to_g(G):
    g = {}
    g['1s'] = 1/32 * (8 * G[0,0,0] + 2 * G[0,2,0] - 4 * G[2,0,0] - G[2,2,0] )
    g['1c'] = 1/16 * (4 * G[0,0,0] +  G[0,2,0] + 4 * G[2,0,0] + G[2,2,0] )
    g['2s'] = 3/32 * ( 2 * G[0,2,0] - G[2,2,0] )
    g['2c'] = 3/16 * (G[0,2,0] + G[2,2,0] )
    g['6s'] = 1/8 * ( 2 * G[0,1,0] - G[2,1,0] )
    g['6c'] = 1/4 * ( G[0,1,0] + G[2,1,0] )
    g[3] = 3/32 * _Re(G[2,2,2])
    g[4] = 3/32 * _Re(G[2,2,1])
    g[5] = sqrt(3)/16 * _Re(G[2,1,1])
    g[7] = sqrt(3)/16 * _Im(G[2,1,1])
    g[8] = 3/32 * _Im(G[2,2,1])
    g[9] = 3/32 * _Im(G[2,2,2])
    return g

def G_s_to_g_s(G):
    # _Im -> -1 * _Re and _Re -> _Im
    g = {}
    g['1s'] = 1/32 * (8 * G[0,0,0] + 2 * G[0,2,0] - 4 * G[2,0,0] - G[2,2,0] )
    g['1c'] = 1/16 * (4 * G[0,0,0] +  G[0,2,0] + 4 * G[2,0,0] + G[2,2,0] )
    g['2s'] = 3/32 * ( 2 * G[0,2,0] - G[2,2,0] )
    g['2c'] = 3/16 * (G[0,2,0] + G[2,2,0] )
    g['6s'] = 1/8 * ( 2 * G[0,1,0] - G[2,1,0] )
    g['6c'] = 1/4 * ( G[0,1,0] + G[2,1,0] )
    g[3] = 3/32 * _Im(G[2,2,2])
    g[4] = 3/32 * _Im(G[2,2,1])
    g[5] = sqrt(3)/16 * _Im(G[2,1,1])
    g[7] = sqrt(3)/16 * -1 * _Re(G[2,1,1])
    g[8] = 3/32 * -1 * _Re(G[2,2,1])
    g[9] = 3/32 * -1 * _Re(G[2,2,2])
    return g

def angularcoeffs_general_v(*args, **kwargs):
    G = angularcoeffs_general_Gbasis_v(*args, **kwargs)
    g = G_to_g(G)
    signflip = [4, '6s', '6c', 7, 9]
    J = {k: -8*4/3.*g[k] if k in signflip else 8*4/3.*g[k] for k in g}
    return J

def angularcoeffs_h_v(*args, **kwargs):
    h = angularcoeffs_h_Gbasis_v(*args, **kwargs)
    g_h = G_to_g(h)
    signflip = [4, '6s', '6c', 7, 9]
    J_h = {k: -8*4/3.*g_h[k] if k in signflip else 8*4/3.*g_h[k] for k in g_h}
    return J_h

def angularcoeffs_s_v(*args, **kwargs):
    s = angularcoeffs_s_Gbasis_v(*args, **kwargs)
    g_s = G_s_to_g_s(s)
    signflip = [4, '6s', '6c', 7, 9]
    J_s = {k: -8*4/3.*g_s[k] if k in signflip else 8*4/3.*g_s[k] for k in g_s}
    return J_s

def helicity_amps_p(q2, mB, mP, mqh, mql, ml1, ml2, ff, wc, prefactor):
    laB = lambda_K(mB**2, mP**2, q2)
    h = {}
    h['V'] = sqrt(laB)/(2*sqrt(q2)) * (
        2*mqh/(mB+mP)*(wc['7']+wc['7p'])*ff['fT']+(wc['v']+wc['vp'])*ff['f+'] )
    h['A'] = sqrt(laB)/(2*sqrt(q2)) * (wc['a']+wc['ap'])*ff['f+']
    h['S'] = (mB**2-mP**2)/2. * ff['f0'] * (
            (wc['s']+wc['sp'])/(mqh-mql) + (ml1-ml2)/q2*(wc['v']+wc['vp']) )
    h['P'] = (mB**2-mP**2)/2. * ff['f0'] * (
            (wc['p']+wc['pp'])/(mqh-mql) + (ml1+ml2)/q2*(wc['a']+wc['ap']) )
    h['T']  = -1j*sqrt(laB)/(2*(mB+mP)) * (wc['t']-wc['tp']) * ff['fT']
    h['Tt'] = -1j*sqrt(laB)/(2*(mB+mP)) * (wc['t']+wc['tp']) * ff['fT']
    return {k: prefactor*v for k, v in h.items()}

def angularcoeffs_general_Gbasis_p(h, q2, mB, mP, mqh, mql, ml1, ml2):
        laB = lambda_K(mB**2, mP**2, q2)
        laGa = lambda_K(q2, ml1**2, ml2**2)
        E1 = sqrt(ml1**2+laGa/(4 * q2))
        E2 = sqrt(ml2**2+laGa/(4 * q2))
        G = {}
        G[0] = (
              ( 4*(E1*E2 + ml1*ml2) + laGa/(3*q2) ) * abs(h['V'])**2
            + ( 4*(E1*E2 - ml1*ml2) + laGa/(3*q2) ) * abs(h['A'])**2
            + ( 4*(E1*E2 - ml1*ml2) + laGa/(  q2) ) * abs(h['S'])**2
            + ( 4*(E1*E2 + ml1*ml2) + laGa/(  q2) ) * abs(h['P'])**2
            +  16*(E1*E2 + ml1*ml2  - laGa/(12*q2)) * abs(h['Tt'])**2
            +   8*(E1*E2 - ml1*ml2  - laGa/(12*q2)) * abs(h['T'])**2
            +      16 * (ml1*E2 + ml2*E1) * _Im( h['V'] * _Co(h['Tt']) )
            + 8*sqrt(2)*(ml1*E2 - ml2*E1) * _Im( h['A'] * _Co(h['T']) ) )
        G[1] = -4*sqrt(laGa) * (
              _Re(   (ml1+ml2)/sqrt(q2) * h['V'] * _Co(h['S'])
                   + (ml1-ml2)/sqrt(q2) * h['A'] * _Co(h['P']) )
            - _Im( 2 * h['Tt'] * _Co(h['S']) + sqrt(2) * h['T'] * _Co(h['P'])) )
        G[2] = -4*laGa/(3*q2) * (
            abs(h['V'])**2 + abs(h['A'])**2 - 2*abs(h['T'])**2 - 4*abs(h['Tt'])**2 )
        prefactor = sqrt(laB)*sqrt(laGa)/(2**9 * pi**3 * mB**3 * q2)
        return {k: prefactor*v for k, v in G.items()}

def angularcoeffs_general_p(*args, **kwargs):
    G = angularcoeffs_general_Gbasis_p(*args, **kwargs)
    J = {}
    J['a'] = G[0] - G[2]/2.
    J['b'] = G[1]
    J['c'] = 3*G[2]/2.
    return J


def angularcoeffs_general_transversity(A, q2, ml):
    """
    Returns the angular coefficients from the transversity amplitudes, 
    compare e.g. https://arxiv.org/pdf/1502.05509 EQ 9.
    The transversity amplitudes are stored in the dictionary A with the keys:
     - para_L,R, 
     - perp_L,R, 
     - 0_L,R, 
     - t, 
     - S

    NB: phi is not used in this function, but is included for consistency with the other functions.
    """
    def _CAS(x):  # complex absolute square
        return x * _Co(x)

    beta_l = sqrt(1 - 4 * ml**2 / q2)

    J = {
        '1s': (2 + beta_l**2) / 4 * ( _CAS(A['para_L']) + _CAS(A['para_R']) + _CAS(A['perp_L']) + _CAS(A['perp_R']) ) 
              + 4 * ml**2 / q2 * _Re( A['perp_L'] * _Co(A['perp_R']) + A['para_L'] * _Co(A['para_R']) ),
        '1c': (_CAS(A['0_L']) + _CAS(A['0_R'])) + 4 * ml**2 / q2 * ( _CAS(A['t']) + 2 * _Re( A['0_L'] * _Co(A['0_R']) ) ) + beta_l**2 * _CAS(A['S']),
        '2s': beta_l**2 / 4 * ( _CAS(A['para_L']) + _CAS(A['para_R']) + _CAS(A['perp_L']) + _CAS(A['perp_R']) ),
        '2c': - 1 * beta_l**2 * (_CAS(A['0_L']) + _CAS(A['0_R'])),
        3: beta_l**2 / 2 * ( _CAS(A['perp_L']) - _CAS(A['para_L']) + _CAS(A['perp_R']) - _CAS(A['para_R']) ),
        4: beta_l**2 / sqrt(2) * ( _Re( A['0_L'] * _Co(A['para_L']) + A['0_R'] * _Co(A['para_R']) ) ),
        5: sqrt(2) * beta_l * ( _Re( A['0_L'] * _Co(A['perp_L']) - A['0_R'] * _Co(A['perp_R']) ) - ml / sqrt(q2) * _Re( A['para_L'] * _Co(A['S']) + A['para_R'] * _Co(A['S']) ) ),  # differs between https://arxiv.org/pdf/0811.1214 (this version) and https://arxiv.org/pdf/1502.05509
        '6s': 2 * beta_l * ( _Re( A['para_L'] * _Co(A['perp_L']) - A['para_R'] * _Co(A['perp_R']) ) ),
        '6c': 4 * beta_l * ml / sqrt(q2) * ( _Re( A['0_L'] * _Co(A['S']) + A['0_R'] * _Co(A['S']) ) ),  # differs between https://arxiv.org/pdf/0811.1214 (this version) and https://arxiv.org/pdf/1502.05509
        7: sqrt(2) * beta_l * ( _Im( A['0_L'] * _Co(A['para_L']) - A['0_R'] * _Co(A['para_R']) ) + ml / sqrt(q2) * _Im( A['perp_L'] * _Co(A['S']) - A['perp_R'] * _Co(A['S']) ) ),  # differs between https://arxiv.org/pdf/0811.1214 (this version) and https://arxiv.org/pdf/1502.05509
        8: beta_l**2 / sqrt(2) * ( _Im( A['0_L'] * _Co(A['perp_L']) + A['0_R'] * _Co(A['perp_R']) ) ),
        9: beta_l**2  * ( _Im( _Co(A['para_L']) * A['perp_L'] + _Co(A['para_R']) * A['perp_R'] ) ),
        # 7, 8, 9 differ between https://arxiv.org/pdf/1502.05509 (this version) https://arxiv.org/pdf/0811.1214 (suspect parenthesis typo with L->R writing)
    }
    return J


def angularcoeffs_h_transversity(A, Atilde, q2, ml, qp) -> dict[str | int, float]: 
    """ 
    Returns the angular coefficients h_i from the transversity amplitudes. 
    Compare e.g. https://arxiv.org/pdf/1502.05509 Appendix C, EQ 117 and following. 
    """
    qp = -qp
    beta_l = sqrt(1 - 4 * ml**2 / q2)
    beta_l2 = 1 - 4 * ml**2 / q2

    AtL_ALs_perp = Atilde['perp_L'] * _Co(A['perp_L'])
    AtR_ARs_perp = Atilde['perp_R'] * _Co(A['perp_R'])
    AtL_ALs_para = Atilde['para_L'] * _Co(A['para_L'])
    AtR_ARs_para = Atilde['para_R'] * _Co(A['para_R'])
    AtL_ALs_0 = Atilde['0_L'] * _Co(A['0_L'])
    AtR_ARs_0 = Atilde['0_R'] * _Co(A['0_R'])
    At_As_t = Atilde['t'] * _Co(A['t'])
    At_As_S = Atilde['S'] * _Co(A['S'])

    AtL_ARs_perp = Atilde['perp_L'] * _Co(A['perp_R'])
    AtR_ALs_perp = Atilde['perp_R'] * _Co(A['perp_L'])
    AtL_ARs_para = Atilde['para_L'] * _Co(A['para_R'])
    AtR_ALs_para = Atilde['para_R'] * _Co(A['para_L'])

    AtL_ARs_0 = Atilde['0_L'] * _Co(A['0_R'])
    AtR_ALs_0 = Atilde['0_R'] * _Co(A['0_L'])

    h = {
        '1s': (2 + beta_l2) / 2 * _Re( qp * ( AtL_ALs_perp + AtL_ALs_para + AtR_ARs_perp + AtR_ARs_para ) )
              + 4 * ml**2 / q2 * _Re( qp * ( AtL_ARs_perp + AtL_ARs_para ) + _Co(qp) * ( _Co( AtR_ALs_perp ) * _Co(AtR_ALs_para) ) ),  # (117)
        '1c': 2 * _Re( qp * ( AtL_ALs_0 + AtR_ARs_0 ) ) 
              + 8 * ml**2 / q2 * (_Re( qp * At_As_t ) + _Re( qp * AtL_ARs_0 + _Co(qp) * _Co(AtR_ALs_0) ))
              + 2 * beta_l2 * _Re( qp * At_As_S ),  # (118)
        '2s': beta_l2 / 2 * _Re( qp * ( AtL_ALs_perp + AtL_ALs_para + AtR_ARs_perp + AtR_ARs_para ) ),  # (119) (= 1s for massless leptons)
        '2c': -2 * beta_l2 * _Re( qp * ( AtL_ALs_0 + AtR_ARs_0 ) ),  # (120) (= -1c for massless leptons)
        3: beta_l2 * _Re( qp * ( AtL_ALs_perp - AtL_ALs_para + AtR_ARs_perp - AtR_ARs_para ) ),  # (121)
        4: beta_l2 / sqrt(2) * _Re( qp * (Atilde['0_L'] * _Co(A['para_L']) + Atilde['0_R'] * _Co(A['para_R'])) + _Co(qp) * ( A['0_L'] * _Co(Atilde['para_L']) + A['0_R'] * _Co(Atilde['para_R']) ) ),  # (122)
        5: sqrt(2) * beta_l * (_Re( qp * ( Atilde['0_L'] * _Co(A['perp_L']) - Atilde['0_R'] * _Co(A['perp_R']) ) + _Co(qp) * (A['0_L'] * _Co(Atilde['perp_L']) - A['0_R'] * _Co(Atilde['perp_R'])) ) 
                                 - ml / sqrt(q2) * _Re( qp * ( Atilde['para_L'] * _Co(A['S']) + Atilde['para_R'] * _Co(A['S']) ) + _Co(qp) * ( A['para_L'] * _Co(Atilde['S']) + A['para_R'] * _Co(Atilde['S']) ) ) ),  # (123)
        '6s': 2 * beta_l * _Re( qp * ( Atilde['para_L'] * _Co(A['perp_L']) - Atilde['para_R'] * _Co(A['perp_R']) ) + _Co(qp) * ( A['para_L'] * _Co(Atilde['perp_L'] - A['para_R'] * _Co(Atilde['perp_R'])) ) ),  # (124)
        '6c': 4 * beta_l * ml / sqrt(q2) * _Re( qp * ( Atilde['0_L'] * _Co(A['S'] + Atilde['0_R'] * _Co(A['S'])) ) + _Co(qp) * ( A['0_L'] * _Co(Atilde['S']) + A['0_R'] * _Co(Atilde['S']) ) ),  # (125)
        7: sqrt(2) * beta_l * ( _Im( qp * ( Atilde['0_L'] * _Co(A['para_L'] - Atilde['0_R'] * _Co(A['para_R'])) ) + _Co(qp) * ( A['0_L'] * _Co(Atilde['para_L']) - A['0_R'] * _Co(Atilde['para_R']) ) ) 
                                 + ml / sqrt(q2) * _Im( qp * ( Atilde['perp_L'] * _Co(A['S']) + Atilde['perp_R'] * _Co(A['S']) ) + _Co(qp) * ( A['perp_L'] * _Co(Atilde['S']) + A['perp_R'] * _Co(Atilde['S']) ) ) ),  # (126)
        8: beta_l2 / sqrt(2) * _Im( qp * ( Atilde['0_L'] * _Co(A['perp_L']) + Atilde['0_R'] * _Co(A['perp_R']) ) + _Co(qp) * ( A['0_L'] * _Co(Atilde['perp_L']) + A['0_R'] * _Co(Atilde['perp_R']) ) ),  # (127)
        9: -beta_l2 * _Im( qp * ( Atilde['para_L'] * _Co(A['perp_L']) + Atilde['para_R'] * _Co(A['perp_R']) ) + _Co(qp) * ( A['para_L'] * _Co(Atilde['perp_L']) + A['para_R'] * _Co(Atilde['perp_R']) ) ),  # (128)
    }
    return h


def angularcoeffs_s_transversity(A, Atilde, q2, ml, qp) -> dict[str | int, float]: 
    """ 
    Returns the angular coefficients h_i from the transversity amplitudes. 
    Compare e.g. https://arxiv.org/pdf/1502.05509 Appendix C, EQ 105 and following. 
    """
    qp = -qp
    beta_l = sqrt(1 - 4 * ml**2 / q2)
    beta_l2 = 1 - 4 * ml**2 / q2

    s = {
        '1s': (2 + beta_l2) / 2 * _Im( qp * ( Atilde['perp_L'] * _Co(A['perp_L']) + Atilde['para_L'] * _Co(A['para_L']) + Atilde['perp_R'] * _Co(A['perp_R']) + Atilde['para_R'] * _Co(A['para_R']) ) ) 
              + 4 * ml**2 / q2 * _Im( qp * ( Atilde['perp_L'] * _Co(A['perp_R']) + Atilde['para_L'] * _Co(A['para_R']) ) - _Co(qp) * ( A['perp_L'] * _Co(Atilde['perp_R']) + A['para_L'] * _Co(Atilde['para_R']) ) ),  # (105)
        '1c': 2 * _Im( qp * ( Atilde['0_L'] * _Co(A['0_L']) + Atilde['0_R'] * _Co(A['0_R']) ) ) 
              + 8 * ml**2 / q2 * ( _Im( qp * Atilde['t'] * _Co(A['t']) ) + _Im( qp * Atilde['0_L'] * _Co(A['0_R']) - _Co(qp) * A['0_L'] * _Co(Atilde['0_R']) ) )
              + 2 * beta_l2 * _Im( qp * Atilde['S'] * _Co(A['S']) ),  # (106)
        '2s': beta_l2 / 2 * _Im( qp * ( Atilde['perp_L'] * _Co(A['perp_L']) + Atilde['para_L'] * _Co(A['para_L']) + Atilde['perp_R'] * _Co(A['perp_R']) + Atilde['para_R'] * _Co(A['para_R']) ) ),  # (107)
        '2c': -2 * beta_l2 * _Im( qp * ( Atilde['0_L'] * _Co(A['0_L']) + Atilde['0_R'] * _Co(A['0_R']) ) ),  # (108)
        3: beta_l2 * _Im( qp * ( Atilde['perp_L'] * _Co(A['perp_L']) - Atilde['para_L'] * _Co(A['para_L']) + Atilde['perp_R'] * _Co(A['perp_R']) - Atilde['para_R'] * _Co(A['para_R']) ) ),  # (109)
        4: beta_l2 / sqrt(2) * _Im( qp * ( Atilde['0_L'] * _Co(A['para_L']) + Atilde['0_R'] * _Co(A['para_R']) ) - _Co(qp) * ( A['0_L'] * _Co(Atilde['para_L']) + A['0_R'] * _Co(Atilde['para_R']) ) ),  # (110)
        5: sqrt(2) * beta_l * ( _Im( qp * ( Atilde['0_L'] * _Co(A['perp_L']) - Atilde['0_R'] * _Co(A['perp_R']) ) - _Co(qp) * ( A['0_L'] * _Co(Atilde['perp_L']) - A['0_R'] * _Co(Atilde['perp_R']) ) )
                               - ml / sqrt(2) * _Im( qp * ( Atilde['para_L'] * _Co(A['S']) + Atilde['para_R'] * _Co(A['S']) ) - _Co(qp) * ( A['para_L'] * _Co(Atilde['S']) + A['para_R'] * _Co(Atilde['S']) ) ) ),  # (111)
        '6s': 2 * beta_l * _Im( qp * ( Atilde['para_L'] * _Co(A['perp_L']) - Atilde['para_R'] * _Co(A['perp_R']) ) - _Co(qp) * ( A['para_L'] * _Co(Atilde['perp_L']) - A['para_R'] * _Co(Atilde['perp_R']) ) ),  # (112)
        '6c': 4 * beta_l * ml / sqrt(q2) * _Im( qp * ( Atilde['0_L'] * _Co(A['S']) + Atilde['0_R'] * _Co(A['S']) ) - _Co(qp) * ( A['0_L'] * _Co(Atilde['S']) + A['0_R'] * _Co(Atilde['S']) ) ),  # (113)
        7: -sqrt(2) * beta_l * ( _Re( qp * ( Atilde['0_L'] * _Co(A['para_L']) - Atilde['0_R'] * _Co(A['para_R']) ) - _Co(qp) * ( A['0_L'] * _Co(Atilde['para_L']) - A['0_R'] * _Co(Atilde['para_R']) ) ) 
                                + ml / sqrt(q2) * _Re( qp * ( Atilde['perp_L'] * _Co(A['S']) + Atilde['perp_R'] * _Co(A['S']) ) - _Co(qp) * ( A['perp_L'] * _Co(Atilde['S']) + A['perp_R'] * _Co(Atilde['S']) ) ) ),  # (114)
        8: -beta_l2 / sqrt(2) * _Re( qp * ( Atilde['0_L'] * _Co(A['perp_L']) + Atilde['0_R'] * _Co(A['perp_R']) ) - _Co(qp) * ( A['0_L'] * _Co(Atilde['perp_L']) + A['0_R'] * _Co(Atilde['perp_R']) ) ),  # (115)
        9: beta_l2 * _Re( qp * ( Atilde['para_L'] * _Co(A['perp_L']) + Atilde['para_R'] * _Co(A['perp_R']) ) - _Co(qp) * ( A['para_L'] * _Co(Atilde['perp_L']) + A['para_R'] * _Co(Atilde['perp_R']) ) ),  # (116)
    }    
    return s

