from sympy import sqrt
import sympy as sp

H = {}
H['pl', 'V'] = sp.symbols('HVp', complex=True)
H['pl', 'A'] = sp.symbols('HAp', complex=True)
H['mi', 'V'] = sp.symbols('HVm', complex=True)
H['mi', 'A'] = sp.symbols('HAm', complex=True)

ta = {
    'para_R': sp.symbols('A_para^R', complex=True),
    'para_L': sp.symbols('A_para^L', complex=True),
    'perp_R': sp.symbols('A_perp^R', complex=True),
    'perp_L': sp.symbols('A_perp^L', complex=True)
}
eqns = [
    H['pl' ,'V'] - 1j * ((ta['para_R'] + ta['para_L']) + (ta['perp_R'] + ta['perp_L']))/sqrt(2),
    H['pl' ,'A'] - 1j * ((ta['para_R'] - ta['para_L']) + (ta['perp_R'] - ta['perp_L']))/sqrt(2),
    H['mi' ,'V'] - 1j * ((ta['para_R'] + ta['para_L']) - (ta['perp_R'] + ta['perp_L']))/sqrt(2),
    H['mi' ,'A'] - 1j * ((ta['para_R'] - ta['para_L']) - (ta['perp_R'] - ta['perp_L']))/sqrt(2),
]

res = sp.solve(eqns, (ta['para_R'], ta['para_L'], ta['perp_R'], ta['perp_L']), dict=True)
for kk, vv in res[0].items():
    print(f"{kk} = {vv.simplify()}")

T23 = sp.symbols("ff['T23']", complex=True)
A12 = sp.symbols("ff['A12']", complex=True)
T2, T3, T1 = sp.symbols("ff['T2'] ff['T3'] ff['T1']", complex=True)
q2 = sp.symbols('q2', real=True, positive=True)
A0, A1, A2, V = sp.symbols("ff['A0'] ff['A1'] ff['A2'] ff['V']", complex=True)
mB, mV = sp.symbols('mB mV', real=True, positive=True)
lamda = sp.symbols('lambda_qsq()', real=True, positive=True)

eqns = [ ( (mB + mV)**2 * (mB**2 - mV**2 - q2) * A1 - lamda * A2) / (16 * mB * mV**2 * (mB + mV)) - A12 ]
res = sp.solve(eqns, A2, dict=True)
print(f"A2 = {res[0][A2].factor().simplify().factor()}")

eqns = [ ((mB**2 - mV**2) * (mB**2 + 3*mV**2 - q2) * T2 - lamda * T3) / (8 * mB * mV**2 * (mB - mV)) - T23 ]
res = sp.solve(eqns, T3, dict=True)
print(f"T3 = {res[0][T3].factor().simplify().factor()}")