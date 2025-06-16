import flavio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from flavio.physics.bdecays.bvll import observables, observables_bs
mpl.rc_file('local-tests/schmitse-rc.rc')


def main() -> None:

    npoints = 100
    fl_pars = flavio.default_parameters.get_central_all()
    B_meson = 'Bs'
    V_meson = 'phi'

    reduced_q2_arr = np.linspace(1, 10, npoints)
    ff = [flavio.physics.bdecays.bvll.amplitudes.get_ff(qsq, fl_pars, B_meson, V_meson) for qsq in reduced_q2_arr]
    ff_keys = ['A0', 'A1', 'A12', 'V', 'T1', 'T2', 'T23']
    ffs = {key: np.asarray([d[key] for d in ff]) for key in ff_keys}

    norm = mpl.colors.Normalize(vmin=0, vmax=len(ff_keys))
    cmap = mpl.cm.get_cmap('viridis')

    dx = [2, 8]
    dy = [0.2, 0.8]
    from_paper = {
        'A0': [-0.16539440203562347, 0.2675438596491229,
                0.9643765903307886, 0.8333333333333334,
                0.07124681933842242, 0.35526315789473684,
                0.38167938931297696, 0.5000000000000002,
                0.643765903307888, 0.6403508771929823,
                1.1959287531806613, 0.986842105263158,
                1.330788804071247, 1.0833333333333337],
        'A1': [-0.16284987277353685, 0.22368421052631599,
                0.08396946564885492, 0.28070175438596495,
                0.37659033078880405, 0.3552631578947368,
                0.6717557251908397, 0.4473684210526314,
                0.9618320610687024, 0.557017543859649,
                1.1984732824427482, 0.6710526315789473,
                1.3282442748091603, 0.7412280701754382],
        'A2': [-0.15521628498727735, 0.06578947368421055,
                0.08905852417302798, 0.07894736842105288,
                0.38931297709923673, 0.10087719298245623,
                0.6513994910941475, 0.12280701754385967,
                0.9516539440203563, 0.1535087719298245,
                1.1933842239185748, 0.1798245614035088,
                1.3155216284987277, 0.19736842105263167],
        'V': [-0.16284987277353696, 7.80777561059455e-34, 
              0.10178117048346054, 0.030701754385965105, 
              0.366412213740458, 0.05701754385964915, 
              0.6386768447837151, 0.08333333333333345, 
              0.9414758269720103, 0.10964912280701776, 
              1.1679389312977098, 0.1315789473684211, 
              1.3206106870229006, 0.14473684210526308], 
        'T1': [-0.16539440203562333, 0.14473684210526316, 
                0.05089058524173019, 0.18859649122807043, 
                0.3206106870229008, 0.2500000000000001, 
                0.603053435114504, 0.3201754385964915, 
                0.847328244274809, 0.39473684210526316, 
                1.099236641221374, 0.49122807017543874, 
                1.3307888040712468, 0.5964912280701754], 
        'T2': [-0.16539440203562342, 0.12280701754385978, 
                0.08142493638676825, 0.14473684210526325, 
                0.3282442748091603, 0.16228070175438605, 
                0.633587786259542, 0.18859649122807026, 
                0.8702290076335876, 0.21052631578947348, 
                1.0814249363867685, 0.23245614035087747, 
                1.3206106870229002, 0.25877192982456115], 
        'T3': [-0.16030534351145054, -0.04824561403508775,
                0.06615776081424922, -0.03508771929824567,
                0.3358778625954196, -0.008771929824561275,
                0.6361323155216283, 0.017543859649122837,
                0.8651399491094146, 0.03947368421052649,
                1.0916030534351147, 0.06140350877193021,
                1.3206106870229006, 0.0877192982456139],
    }
    from_paper = {key: (dx[0] + np.asarray(val)[::2] * (dx[1] - dx[0]), dy[0] + np.asarray(val)[1::2] * (dy[1] - dy[0]) )
                  for key, val in from_paper.items()}

    mB = fl_pars[f'm_{B_meson}']
    mV = fl_pars[f'm_{V_meson}']
    def lambda_qsq(q2, mB, mV):
        return ((mB + mV)**2 - q2) * ((mB - mV)**2 - q2)
    # ff_T3 = (ffs['T23'] - ffs['T2']) * (mB - mV) / (mB + mV)
    # ff_A2 = (ffs['A1'] - ffs['A12']) * (mB - mV) / (mB + mV)
    ff_A2 = -(mB + mV)*(-ffs['A1']*mB**3 - ffs['A1']*mB**2*mV + ffs['A1']*mB*mV**2 + ffs['A1']*mB*reduced_q2_arr + ffs['A1']*mV**3 + ffs['A1']*mV*reduced_q2_arr + 16*ffs['A12']*mB*mV**2)/lambda_qsq(reduced_q2_arr, mB, mV)
    ff_T3 = -(mB - mV)*(-ffs['T2']*mB**3 - ffs['T2']*mB**2*mV - 3*ffs['T2']*mB*mV**2 + ffs['T2']*mB*reduced_q2_arr - 3*ffs['T2']*mV**3 + ffs['T2']*mV*reduced_q2_arr + 8*ffs['T23']*mB*mV**2)/lambda_qsq(reduced_q2_arr, mB, mV)

    fig, axs = plt.subplots(1, 2, figsize=(25, 10))
    axs[0].plot(reduced_q2_arr, ffs['A0'], label='A0 (flavio)', color=cmap(norm(0)))
    axs[0].plot(reduced_q2_arr, ffs['A1'], label='A1 (flavio)', color=cmap(norm(1)), linestyle='dashed')
    axs[0].plot(reduced_q2_arr, ffs['A12'], label='A12 (flavio)', color=cmap(norm(2)), linestyle='dotted')
    axs[0].plot(reduced_q2_arr, ffs['V'], label='V (flavio)', color=cmap(norm(3)), linestyle='dashdot')

    axs[0].plot(from_paper['A0'][0], from_paper['A0'][1], label='A0 (from [0811.1214])', color=cmap(norm(0)), marker='o', markersize=10, linestyle='None')
    axs[0].plot(from_paper['A1'][0], from_paper['A1'][1], label='A1 (from [0811.1214])', color=cmap(norm(1)), marker='^', markersize=10, linestyle='None')
    axs[0].plot(from_paper['A2'][0], from_paper['A2'][1], label='A2 (from [0811.1214])', color=cmap(norm(2)), marker='v', markersize=10, linestyle='None')
    axs[0].plot(from_paper['V'][0], from_paper['V'][1], label='V (from [0811.1214])', color=cmap(norm(3)), marker='s', markersize=10, linestyle='None')

    axs[0].plot(reduced_q2_arr, ff_A2, label='A2 ([1503.05534])', color='orange', linestyle='dashdot')

    axs[0].set_xlabel(r'$q^2$ [GeV$^2$]')
    axs[0].set_ylabel('Form Factors')
    # axs[0].set_ylim(0, 1)
    axs[0].legend(ncols=2)
    axs[1].plot(reduced_q2_arr, ffs['T1'], label='T1 (flavio)', color=cmap(norm(4)))
    axs[1].plot(reduced_q2_arr, ffs['T2'], label='T2 (flavio)', color=cmap(norm(5)), linestyle='dashed')
    axs[1].plot(reduced_q2_arr, ffs['T23'], label='T23 (flavio)', color=cmap(norm(6)), linestyle='dotted')
    # axs[1].plot(reduced_q2_arr, ffs['T23'] - ffs['T2'], label='T23 - T2', color=cmap(norm(6)), linestyle='dotted')

    axs[1].plot(from_paper['T1'][0], from_paper['T1'][1], label='T1 (from [0811.1214])', color=cmap(norm(4)), marker='o', markersize=10, linestyle='None')
    axs[1].plot(from_paper['T2'][0], from_paper['T2'][1], label='T2 (from [0811.1214])', color=cmap(norm(5)), marker='^', markersize=10, linestyle='None')
    axs[1].plot(from_paper['T3'][0], from_paper['T3'][1], label='T3 (from [0811.1214])', color=cmap(norm(6)), marker='v', markersize=10, linestyle='None')

    axs[1].plot(reduced_q2_arr, ff_T3, label='T3 ([1503.05534])', color='black', linestyle='dashdot')

    axs[1].set_xlabel(r'$q^2$ [GeV$^2$]')
    axs[1].set_ylabel('Form Factors')
    axs[1].set_ylim(0, 1)
    axs[1].legend(ncols=2)
    fig.tight_layout()
    plt.tight_layout()
    plt.savefig('plots/form_factors_Bs_phi_mu.pdf', transparent=True)
    print('Saved figure to plots/form_factors_Bs_phi_mu.pdf')
    plt.close()
    return None


if __name__ == "__main__":
    main()