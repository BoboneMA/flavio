from sympy import symbols, sqrt, re, im, pprint, latex

qp = symbols(r'e^{i\phi}', complex=True)
qp_co = symbols(r'e^{-i\phi}', complex=True)
ml1, ml2 = symbols(r'm_{\ell_1} m_{\ell_2}', real=True, positive=True)
laGa = symbols(r'\lambda', real=True, positive=True)
q2 = symbols(r'q^2', real=True, positive=True)

Htilde = {
    ('pl', 'V'): symbols(r'\tilde{H^{V}_{+}}', complex=True),
    ('pl', 'A'): symbols(r'\tilde{H^{A}_{+}}', complex=True),
    ('mi', 'V'): symbols(r'\tilde{H^{V}_{-}}', complex=True),
    ('mi', 'A'): symbols(r'\tilde{H^{A}_{-}}', complex=True),
    ('pl', 'T'): symbols(r'\tilde{H^{T}_{+}}', complex=True),
    ('mi', 'T'): symbols(r'\tilde{H^{T}_{-}}', complex=True),
    ('pl', 'Tt'): symbols(r'\tilde{H^{Tt}_{+}}', complex=True),
    ('mi', 'Tt'): symbols(r'\tilde{H^{Tt}_{-}}', complex=True),
    ('0', 'A'): symbols(r'\tilde{H^{A}_{0}}', complex=True),
    ('0', 'V'): symbols(r'\tilde{H^{V}_{0}}', complex=True),
    'P': symbols(r'\tilde{H^{P}}', complex=True),
    'S': symbols(r'\tilde{H^{S}}', complex=True),
    ('0', 'T'): symbols(r'\tilde{H^{T}_{0}}', complex=True),
    ('0', 'Tt'): symbols(r'\tilde{H^{Tt}_{0}}', complex=True),
}

H = {
    ('pl', 'V'): symbols(r'H^{V}_{+}', complex=True),
    ('pl', 'A'): symbols(r'H^{A}_{+}', complex=True),
    ('mi', 'V'): symbols(r'H^{V}_{-}', complex=True),
    ('mi', 'A'): symbols(r'H^{A}_{-}', complex=True), 
    ('pl', 'T'): symbols(r'H^{T}_{+}', complex=True),
    ('mi', 'T'): symbols(r'H^{T}_{-}', complex=True),
    ('pl', 'Tt'): symbols(r'H^{Tt}_{+}', complex=True),
    ('mi', 'Tt'): symbols(r'H^{Tt}_{-}', complex=True),
    ('0', 'A'): symbols(r'H^{A}_{0}', complex=True),
    ('0', 'V'): symbols(r'H^{V}_{0}', complex=True),
    'P': symbols(r'H^{P}', complex=True),
    'S': symbols(r'H^{S}', complex=True),
    ('0', 'T'): symbols(r'H^{T}_{0}', complex=True),
    ('0', 'Tt'): symbols(r'H^{Tt}_{0}', complex=True),
}

CH = {
    ('pl', 'V'): symbols(r'H_{+}^{V*}', complex=True),
    ('pl', 'A'): symbols(r'H_{+}^{A*}', complex=True),
    ('mi', 'V'): symbols(r'H_{-}^{V*}', complex=True),
    ('mi', 'A'): symbols(r'H_{-}^{A*}', complex=True),
    ('pl', 'T'): symbols(r'H^{T*}_{+}', complex=True),
    ('mi', 'T'): symbols(r'H^{T*}_{-}', complex=True),
    ('pl', 'Tt'): symbols(r'H^{Tt*}_{+}', complex=True),
    ('mi', 'Tt'): symbols(r'H^{Tt*}_{-}', complex=True),
    ('0', 'A'): symbols(r'H^{A*}_{0}', complex=True),
    ('0', 'V'): symbols(r'H^{V*}_{0}', complex=True),
    'P': symbols(r'H^{P*}', complex=True),
    'S': symbols(r'H^{S*}', complex=True),
    ('0', 'T'): symbols(r'H^{T*}_{0}', complex=True),
    ('0', 'Tt'): symbols(r'H^{Tt*}_{0}', complex=True),
}

CHtilde = {
    ('pl', 'V'): symbols(r'\tilde{H_{+}^{V*}}', complex=True),
    ('pl', 'A'): symbols(r'\tilde{H_{+}^{A*}}', complex=True),
    ('mi', 'V'): symbols(r'\tilde{H_{-}^{V*}}', complex=True),
    ('mi', 'A'): symbols(r'\tilde{H_{-}^{A*}}', complex=True), 
    ('pl', 'T'): symbols(r'\tilde{H^{T*}_{+}}', complex=True),
    ('mi', 'T'): symbols(r'\tilde{H^{T*}_{-}}', complex=True),
    ('pl', 'Tt'): symbols(r'\tilde{H^{Tt*}_{+}}', complex=True),
    ('mi', 'Tt'): symbols(r'\tilde{H^{Tt*}_{-}}', complex=True),
    ('0', 'A'): symbols(r'\tilde{H^{A*}_{0}}', complex=True),
    ('0', 'V'): symbols(r'\tilde{H^{V*}_{0}}', complex=True),
    'P': symbols(r'\tilde{H^{P*}}', complex=True),
    'S': symbols(r'\tilde{H^{S*}}', complex=True),
    ('0', 'T'): symbols(r'\tilde{H^{T*}_{0}}', complex=True),
    ('0', 'Tt'): symbols(r'\tilde{H^{Tt*}_{0}}', complex=True),
}

G = {}
G[2,1,0] = (-4 * sqrt(laGa)/3 * (re((-qp * Htilde['pl','V'] * CH['pl','A'] + -qp_co * H['pl','V'] * CHtilde['pl','A']) - (-qp * Htilde['mi','V']  * CH['mi','A'] + -qp_co * H['mi','V']  * CHtilde['mi','A']))
+2 * sqrt(2) * (ml1**2-ml2**2)/q2 * re((-qp * Htilde['pl','T']  * CH['pl','Tt'] + -qp_co * H['pl','T']  * CHtilde['pl','Tt'])-(-qp * Htilde['mi','T']  * CH['mi','Tt'] + -qp_co * H['mi','T']  * CHtilde['mi','Tt']))
+2 * (ml1+ml2)/sqrt(q2) * im((-qp * Htilde['pl','A']  * CH['pl','Tt'] + -qp_co * H['pl','A']  * CHtilde['pl','Tt'])-(-qp * Htilde['mi','A']  * CH['mi','Tt'] + -qp_co * H['mi','A']  * CHtilde['mi','Tt']))
+sqrt(2) * (ml1-ml2)/sqrt(q2) * im((-qp * Htilde['pl','V']  * CH['pl','T'] + -qp_co * H['pl','V']  * CHtilde['pl','T'])-(-qp * Htilde['mi','V']  * CH['mi','T'] + -qp_co * H['mi','V']  * CHtilde['mi','T']))
+2 * (ml1-ml2)/sqrt(q2) * re((-qp * Htilde['0','A']  * CH['P'] + -qp_co * H['0','A']  * CHtilde['P']))+2 * (ml1+ml2)/sqrt(q2) * re((-qp * Htilde['0','V']  * CH['S'] + -qp_co * H['0','V']  * CHtilde['S']))
-2 * im(sqrt(2) * (-qp * Htilde['0','T']  * CH['P'] + -qp_co * H['0','T']  * CHtilde['P'])+2 * (-qp * Htilde['0','Tt']  * CH['S'] + -qp_co * H['0','Tt']  * CHtilde['S']))))

print(r"G^{2,1}_{0} =" + f" {G[2,1,0]}")
pprint(G[2, 1, 0])
print(latex(G[2, 1, 0]))