import flavio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import logging
from flavio.physics.bdecays.bvll import observables, observables_bs
from scipy.interpolate import interp1d
from wilson import Wilson
mpl.rc_file('local-tests/schmitse-rc.rc')
logging.getLogger('matplotlib.texmanager').setLevel(logging.WARNING)


def get_amplitudes(y, x, gamma, J, J_bar, J_h, J_s) -> tuple[list]:
    return y, x, gamma, J, J_bar, J_h, J_s
def get_old_amplitudes(J, J_bar) -> tuple[list]:
    return J, J_bar


def main() -> None:

    q2_min, q2_max = 0.98, 18
    npoints = 100
    q2_arr = np.linspace(q2_min, q2_max, npoints)

    fl_pars = flavio.default_parameters.get_central_all()
    wc_obj = flavio.WilsonCoefficients()
    # wc_dict = {
    #     'C9_bsmumu': 4.0749,
    #     'C10_bsmumu': -4.3085, 
    # }
    # wc_obj.set_initial(wc_dict, scale=4.8)

    B_meson = 'Bs'
    V_meson = 'phi'
    lepton = 'mu'

    amplitudes_helicity = np.array([observables.BVll_obs(get_old_amplitudes, q2, B_meson, V_meson, lepton, wc_obj, fl_pars)() for q2 in q2_arr])
    amplitudes_transversity = np.array([observables_bs.bsvll_obs_trans(get_amplitudes, q2, wc_obj, fl_pars, B_meson, V_meson, lepton) for q2 in q2_arr])
    amplitudes_bs_helicity = np.array([observables_bs.bsvll_obs(get_amplitudes, q2, wc_obj, fl_pars, B_meson, V_meson, lepton) for q2 in q2_arr])

    obses = ['1s', '1c', '2s', '2c', 3, 4, 5, '6s', '6c', 7, 8, 9]
    amp_arr_hel = {key: np.asarray([d[0][key] for d in amplitudes_helicity]) for key in obses}
    amp_arr_trans = {key: np.asarray([d[3][key] for d in amplitudes_transversity]) for key in obses}
    amp_arr_bs_hel = {key: np.asarray([d[3][key] for d in amplitudes_bs_helicity]) for key in obses}

    amp_arr_bar_hel = {key: np.asarray([d[1][key] for d in amplitudes_helicity]) for key in obses}
    amp_arr_bar_trans = {key: np.asarray([d[4][key] for d in amplitudes_transversity]) for key in obses}
    amp_arr_bar_bs_hel = {key: np.asarray([d[4][key] for d in amplitudes_bs_helicity]) for key in obses}

    amp_arr_h_hel, amp_arr_s_hel = {key: [] for key in obses}, {key: [] for key in obses}
    amp_arr_h_trans, amp_arr_s_trans = {key: [] for key in obses}, {key: [] for key in obses}
    for ctr_one in range(npoints):
        for key in obses:
            amp_arr_h_hel[key].append(amplitudes_bs_helicity[ctr_one][5][key])
            amp_arr_h_trans[key].append(amplitudes_transversity[ctr_one][5][key])
            amp_arr_s_hel[key].append(amplitudes_bs_helicity[ctr_one][6][key])
            amp_arr_s_trans[key].append(amplitudes_transversity[ctr_one][6][key])
    amp_arr_h_hel = {key: np.asarray(amp_arr_h_hel[key]) for key in obses}
    amp_arr_h_trans = {key: np.asarray(amp_arr_h_trans[key]) for key in obses}
    amp_arr_s_hel = {key: np.asarray(amp_arr_s_hel[key]) for key in obses}
    amp_arr_s_trans = {key: np.asarray(amp_arr_s_trans[key]) for key in obses}

    fig, axs = plt.subplots(4, 3, figsize=(25, 25))

    for j, obs in enumerate(obses):
        ax = axs[j // 3, j % 3]
        ax.plot(q2_arr, amp_arr_bs_hel[obs], label='$B_s^0$ Helicity Amplitudes', color='red', linestyle='-')
        ax.plot(q2_arr, amp_arr_hel[obs], label='Helicity Amplitudes', color='blue', linestyle=':')
        ax.plot(q2_arr, amp_arr_trans[obs], label='Transversity Amplitudes', color='darkturquoise', linestyle='--')
        ax.set_xlabel(r'$q^2$ [GeV$^2$]')
        ax.set_ylabel(f'$J_{obs}$')
        ax.legend()

    fig.tight_layout()
    plt.savefig('plots/comparison_J_helicity_transversity.pdf', transparent=True)
    print('Saved figure to plots/comparison_J_helicity_transversity.pdf')

    fig, axs = plt.subplots(4, 3, figsize=(25, 25))

    for j, obs in enumerate(obses):
        ax = axs[j // 3, j % 3]
        ax.plot(q2_arr, amp_arr_bar_bs_hel[obs], label='$B_s^0$ Helicity Amplitudes', color='red', linestyle='-')
        ax.plot(q2_arr, amp_arr_bar_hel[obs], label='Helicity Amplitudes', color='blue', linestyle=':')
        ax.plot(q2_arr, amp_arr_bar_trans[obs], label='Transversity Amplitudes', color='darkturquoise', linestyle='--')
        ax.set_xlabel(r'$q^2$ [GeV$^2$]')
        ax.set_ylabel(r'$\bar{J}'+f'_{obs}$')
        ax.legend()

    fig.tight_layout()
    plt.savefig('plots/comparison_Jbar_helicity_transversity.pdf', transparent=True)
    print('Saved figure to plots/comparison_Jbar_helicity_transversity.pdf')

    fig, axs = plt.subplots(4, 3, figsize=(25, 25))
    for j, obs in enumerate(obses):
        ax = axs[j // 3, j % 3]
        ax.plot(q2_arr, amp_arr_bs_hel[obs] + amp_arr_bar_bs_hel[obs], label='$B_s^0$ Helicity Amplitudes', color='red', linestyle='-')
        ax.plot(q2_arr, amp_arr_hel[obs] + amp_arr_bar_hel[obs], label='Helicity Amplitudes', color='blue', linestyle=':')
        ax.plot(q2_arr, amp_arr_trans[obs] + amp_arr_bar_trans[obs], label='Transversity Amplitudes', color='darkturquoise', linestyle='--')
        ax.set_xlabel(r'$q^2$ [GeV$^2$]')
        ax.set_ylabel(f'$J_{obs} + '+r'\bar{J}'+f'_{obs}$')
        ax.legend()
    fig.tight_layout()
    plt.savefig('plots/comparison_J_plus_Jbar_helicity_transversity.pdf', transparent=True)
    print('Saved figure to plots/comparison_J_plus_Jbar_helicity_transversity.pdf')

    fig, axs = plt.subplots(4, 3, figsize=(25, 25))
    for j, obs in enumerate(obses):
        ax = axs[j // 3, j % 3]
        ax.plot(q2_arr, amp_arr_bs_hel[obs] - amp_arr_bar_bs_hel[obs], label='$B_s^0$ Helicity Amplitudes', color='red', linestyle='-')
        ax.plot(q2_arr, amp_arr_hel[obs] - amp_arr_bar_hel[obs], label='Helicity Amplitudes', color='blue', linestyle=':')
        ax.plot(q2_arr, (amp_arr_trans[obs] - amp_arr_bar_trans[obs]), label='Transversity Amplitudes', color='darkturquoise', linestyle='--')
        ax.set_xlabel(r'$q^2$ [GeV$^2$]')
        ax.set_ylabel(f'$J_{obs} - '+r'\bar{J}'+f'_{obs}$')
        ax.legend()
    fig.tight_layout()
    plt.savefig('plots/comparison_J_minus_Jbar_helicity_transversity.pdf', transparent=True)
    print('Saved figure to plots/comparison_J_minus_Jbar_helicity_transversity.pdf')

    fig, axs = plt.subplots(4, 3, figsize=(25, 25))
    for j, obs in enumerate(obses):
        ax = axs[j // 3, j % 3]
        ax.plot(q2_arr, amp_arr_h_hel[obs], label='$B_s^0$ Helicity Amplitudes', color='red', linestyle='-')
        ax.plot(q2_arr, amp_arr_h_trans[obs], label=r'Transversity Amplitudes $\times -1$', color='darkturquoise', linestyle='--')
        ax.set_xlabel(r'$q^2$ [GeV$^2$]')
        ax.set_ylabel(f'$h_{obs}$')
        ax.legend()
    fig.tight_layout()
    plt.savefig('plots/comparison_h_helicity_transversity.pdf', transparent=True)
    print('Saved figure to plots/comparison_h_helicity_transversity.pdf')

    fig, axs = plt.subplots(4, 3, figsize=(25, 25))
    for j, obs in enumerate(obses):
        ax = axs[j // 3, j % 3]
        ax.plot(q2_arr, amp_arr_s_hel[obs], label='$B_s^0$ Helicity Amplitudes', color='red', linestyle='-')
        ax.plot(q2_arr, amp_arr_s_trans[obs], label=r'Transversity Amplitudes $\times -1$', color='darkturquoise', linestyle='--')
        ax.set_xlabel(r'$q^2$ [GeV$^2$]')
        ax.set_ylabel(f'$s_{obs}$')
        ax.legend()
    fig.tight_layout()
    plt.savefig('plots/comparison_s_helicity_transversity.pdf', transparent=True)
    print('Saved figure to plots/comparison_s_helicity_transversity.pdf')

    # Q8 minus = s_8 / sqrt( -2 * ( J_2c + Jtilde_2c ) * ( 2 * (J_2s + Jtilde_2s) - (J_3 + Jtilde_3) ) ) -> note Jtilde_3 = -Jbar_3, others are same sign
    Q8mi = amp_arr_s_trans[8] / np.sqrt( -2 * ( amp_arr_trans['2c'] + amp_arr_bar_trans['2c'] ) * ( 2 * ( amp_arr_trans['2s'] + amp_arr_bar_trans['2s'] ) - ( amp_arr_trans[3] - amp_arr_bar_trans[3] ) ) )
    # Q9 = s_9 / 2 / (J_2s + Jtilde_2s)
    Q9 = amp_arr_s_trans[9] / 2 / (amp_arr_trans['2s'] + amp_arr_bar_trans['2s'])

    Q8mi_hel = amp_arr_s_hel[8] / np.sqrt( -2 * ( amp_arr_bs_hel['2c'] + amp_arr_bar_bs_hel['2c'] ) * ( 2 * ( amp_arr_bs_hel['2s'] + amp_arr_bar_bs_hel['2s'] ) - ( amp_arr_bs_hel[3] - amp_arr_bar_bs_hel[3] ) ) )
    Q9_hel = amp_arr_s_hel[9] / 2 / (amp_arr_bs_hel['2s'] + amp_arr_bar_bs_hel['2s'])

    q2_arr_paper = np.linspace(1, 9, 100)
    Q8mi_paper_lower, Q8mi_paper_upper = Q8mi_paper(q2_arr_paper)
    Q9_paper_lower, Q9_paper_upper = Q9_paper(q2_arr_paper)

    fig, ax = plt.subplots()
    ax.plot(q2_arr, Q8mi, label=r'Transversity Amplitude $\times -1$', color='darkturquoise', linestyle='--')
    ax.plot(q2_arr, Q8mi_hel, label='Helicity Amplitude', color='blue', linestyle=':')
    ax.fill_between(q2_arr_paper, Q8mi_paper_lower, Q8mi_paper_upper, color='orange', alpha=0.75, label='DGV [1502.05509]')
    ax.set_xlabel(r'$q^2$ [GeV$^2$]')
    ax.set_ylabel(r'$Q_8^-$')
    ax.legend()
    fig.tight_layout()
    plt.savefig('plots/comparison_Q8mi_helicity_transversity.pdf', transparent=True)
    print('Saved figure to plots/comparison_Q8mi_helicity_transversity.pdf')

    fig, ax = plt.subplots()
    ax.plot(q2_arr, Q9, label=r'Transversity Amplitude $\times -1$', color='darkturquoise', linestyle='--')
    ax.plot(q2_arr, Q9_hel, label='Helicity Amplitude', color='blue', linestyle=':')
    ax.fill_between(q2_arr_paper, Q9_paper_lower, Q9_paper_upper, color='orange', alpha=0.75, label='DGV [1502.05509]')
    ax.set_xlabel(r'$q^2$ [GeV$^2$]')
    ax.set_ylabel(r'$Q_9$')
    ax.legend()
    fig.tight_layout()
    plt.savefig('plots/comparison_Q9_helicity_transversity.pdf', transparent=True)
    print('Saved figure to plots/comparison_Q9_helicity_transversity.pdf')

    return None


def Q8mi_paper(q2_arr: np.typing.NDArray) -> list[np.typing.NDArray]:
    points_lower = [-0.16144018583042974, -0.01685393258426982,
              -0.0847851335656214, 0.1404494382022471,
              0.0058072009291521, 0.35580524344569286,
              0.07549361207897787, 0.5205992509363295,
              0.12311265969802548, 0.6179775280898877,
              0.17770034843205573, 0.7172284644194757,
              0.28222996515679427, 0.8576779026217228,
              0.38327526132404166, 0.9382022471910115,
              0.47619047619047594, 0.9831460674157304,
              0.559814169570267, 1.0112359550561798,
              0.6562137049941926, 1.0318352059925096,
              0.7317073170731706, 1.0411985018726595,
              0.8257839721254355, 1.050561797752809,
              0.9186991869918698, 1.0543071161048694,
              1.0267131242740999, 1.0543071161048692,
              1.117305458768873, 1.052434456928839,
              1.1672473867595816, 1.0486891385767791,
              -0.12427409988385599, 0.05617977528089888,
              -0.03948896631823466, 0.24719101123595497,
              0.044134727061556245, 0.44756554307116103,
              0.22996515679442497, 0.7958801498127342,
              0.3368176538908245, 0.9044943820224721]
    points_upper = [
        -0.16608594657375145, 0.013108614232209697,
        -0.1405342624854821, 0.06367041198501856,
        -0.10336817653890827, 0.1423220973782771,
        -0.07317073170731712, 0.21722846441947558,
        -0.04181184668989549, 0.29775280898876394,
        -0.016260162601626063, 0.3632958801498127,
        0.020905923344947747, 0.46067415730337086,
        0.04878048780487803, 0.5318352059925093,
        0.08478513356562131, 0.6142322097378279,
        0.12427409988385588, 0.6928838951310862,
        0.16724738675958187, 0.7677902621722847,
        0.2137049941927989, 0.8333333333333334,
        0.25783972125435534, 0.8838951310861424,
        0.3042973286875725, 0.9250936329588016,
        0.3588850174216025, 0.9606741573033709,
        0.4227642276422764, 0.9906367041198503,
        0.47619047619047605, 1.0112359550561798,
        0.5400696864111497, 1.0280898876404496,
        0.5958188153310103, 1.0393258426966294,
        0.651567944250871, 1.0468164794007488,
        0.7073170731707317, 1.0561797752808988,
        0.7688734030197445, 1.0655430711610487,
        0.826945412311266, 1.0692883895131085,
        0.889663182346109, 1.0711610486891385,
        0.9535423925667827, 1.0692883895131085,
        1.005807200929152, 1.0692883895131087,
        1.0592334494773517, 1.069288389513109,
        1.1149825783972125, 1.0655430711610485,
        1.1672473867595816, 1.063670411985019,]
    dx = [2, 8]
    dy = [-0.2, 0.4]
    xs_lower, ys_lower = (dx[0] + (dx[1] - dx[0]) * np.array(points_lower[::2]), 
                          dy[0] + (dy[1] - dy[0]) * np.array(points_lower[1::2]))
    xs_upper, ys_upper = (dx[0] + (dx[1] - dx[0]) * np.array(points_upper[::2]), 
                          dy[0] + (dy[1] - dy[0]) * np.array(points_upper[1::2]))

    sorted_indices = np.argsort(xs_lower)
    xs_sorted_lower = np.array(xs_lower)[sorted_indices]
    ys_sorted_lower = np.array(ys_lower)[sorted_indices]
    sorted_indices = np.argsort(xs_upper)
    xs_sorted_upper = np.array(xs_upper)[sorted_indices]
    ys_sorted_upper = np.array(ys_upper)[sorted_indices]

    interpolator_lower = interp1d(xs_sorted_lower, ys_sorted_lower, kind='linear', fill_value='extrapolate')
    interpolator_upper = interp1d(xs_sorted_upper, ys_sorted_upper, kind='linear', fill_value='extrapolate')
    return [interpolator_lower(q2_arr), interpolator_upper(q2_arr)] 


def Q9_paper(q2_arr: np.typing.NDArray) -> list[np.typing.NDArray]:
    points_lower = [
        -0.16697558626579806, 0.04984962884218767,
        0.005778613677869573, 0.07255067644306351,
        0.1347593607181472, 0.08381942740843729,
        0.2258815565913448, 0.08556391455647339,
        0.33239067426245456, 0.08535948246881278,
        0.4803200043612179, 0.08507554901372875,
        0.6448128764957615, 0.0885985953244112,
        0.7489505819500096, 0.09223748648476772,
        0.784458164108994, 0.08833056214281186,
        0.8826786963592917, 0.09198081064137179,
        0.980894685674308, 0.09946983945266726,
        1.0625403185506221, 0.10891005896729995,
        1.1477339838816658, 0.12026285423537848,
        1.165480960558236, 0.12406756253350405,
        -0.08533222485712466, 0.061209238513188303,
        0.07322303086470232, 0.08201815356938441,
        0.5501403767001936, 0.08686092257929692,
    ]
    points_upper = [
        -0.1670482732302996, 0.11127011384595818,
        -0.09843177874088009, 0.13033227028647762,
        -0.03809478380170989, 0.1455715466877459,
        0.022249025540382206, 0.15505265261991086,
        0.09798884255094906, 0.1549072786909078,
        0.05774979329644475, 0.15690389874705854,
        0.15006450968099513, 0.15096855380198276,
        0.2400191711868873, 0.13927958132308466,
        0.32287095338039823, 0.12952362780639834,
        0.40335586629232884, 0.11977221722499352,
        0.49330371339529905, 0.11384141521519874,
        0.5796989851082581, 0.10983681776469421,
        0.6767383541854064, 0.11156994757452682,
        0.7169705890369886, 0.11533149798747984,
        0.7524668138577699, 0.12102152442736322,
        0.7808624308338105, 0.1267251796730905,
        0.856597704909096, 0.1304185860568231,
        0.9465319232062223, 0.13600412498523545,
        1.0045134062020153, 0.14165099353994628,
        1.0778681821898766, 0.156865283797167,
        1.1654196309319378, 0.17589109675543593,
        1.112178700902227, 0.16447697186105908,
        0.19030583040314009, 0.14705254358946399,
    ]

    dx = [2, 8]
    dy = [-1, -0.94]
    xs_lower, ys_lower = (dx[0] + (dx[1] - dx[0]) * np.array(points_lower[::2]), 
                          dy[0] + (dy[1] - dy[0]) * np.array(points_lower[1::2]))
    xs_upper, ys_upper = (dx[0] + (dx[1] - dx[0]) * np.array(points_upper[::2]), 
                          dy[0] + (dy[1] - dy[0]) * np.array(points_upper[1::2]))

    sorted_indices = np.argsort(xs_lower)
    xs_sorted_lower = np.array(xs_lower)[sorted_indices]
    ys_sorted_lower = np.array(ys_lower)[sorted_indices]
    sorted_indices = np.argsort(xs_upper)
    xs_sorted_upper = np.array(xs_upper)[sorted_indices]
    ys_sorted_upper = np.array(ys_upper)[sorted_indices]

    interpolator_lower = interp1d(xs_sorted_lower, ys_sorted_lower, kind='linear', fill_value='extrapolate')
    interpolator_upper = interp1d(xs_sorted_upper, ys_sorted_upper, kind='linear', fill_value='extrapolate')
    return [interpolator_lower(q2_arr), interpolator_upper(q2_arr)]


if __name__ == "__main__":
    main()