import flavio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import logging
from flavio.physics.bdecays.bvll import observables, observables_bs
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

    B_meson = 'Bs'
    V_meson = 'phi'
    lepton = 'mu'

    amplitudes_helicity = np.array([observables.BVll_obs(get_old_amplitudes, q2, B_meson, V_meson, lepton, wc_obj, fl_pars)() for q2 in q2_arr])
    amplitudes_transversity = np.array([observables_bs.bsvll_obs_trans(get_amplitudes, q2, wc_obj, fl_pars, B_meson, V_meson, lepton) for q2 in q2_arr])

    obses = ['1s', '1c', '2s', '2c', 3, 4, 5, '6s', '6c', 7, 8, 9]
    sign = {4, '6s', 7, 9}
    amp_arr_hel = {key: np.asarray([d[0][key] for d in amplitudes_helicity]) for key in obses}
    amp_arr_trans = {key: np.asarray([d[3][key] for d in amplitudes_transversity]) for key in obses}

    amp_arr_bar_hel = {key: np.asarray([d[1][key] for d in amplitudes_helicity]) for key in obses}
    amp_arr_bar_trans = {key: np.asarray([d[4][key] for d in amplitudes_transversity]) for key in obses}

    fig, axs = plt.subplots(4, 3, figsize=(25, 25))

    for j, obs in enumerate(obses):
        ax = axs[j // 3, j % 3]
        fac = -1 if obs in sign else 1
        ax.plot(q2_arr, amp_arr_hel[obs], label='Helicity Amplitudes', color='blue')
        ax.plot(q2_arr, fac * amp_arr_trans[obs], label='Transversity Amplitudes', color='red', linestyle='--')
        ax.set_xlabel(r'$q^2$ [GeV$^2$]')
        ax.set_ylabel(f'$J_{obs}$')
        ax.legend()

    fig.tight_layout()
    plt.savefig('plots/comparison_J_helicity_transversity.pdf', transparent=True)
    print('Saved figure to plots/comparison_J_helicity_transversity.pdf')

    fig, axs = plt.subplots(4, 3, figsize=(25, 25))

    for j, obs in enumerate(obses):
        ax = axs[j // 3, j % 3]
        fac = -1 if obs in sign else 1
        ax.plot(q2_arr, amp_arr_bar_hel[obs], label='Helicity Amplitudes', color='blue')
        ax.plot(q2_arr, fac * amp_arr_bar_trans[obs], label='Transversity Amplitudes', color='red', linestyle='--')
        ax.set_xlabel(r'$q^2$ [GeV$^2$]')
        ax.set_ylabel(r'$\bar{J}'+f'_{obs}$')
        ax.legend()

    fig.tight_layout()
    plt.savefig('plots/comparison_Jbar_helicity_transversity.pdf', transparent=True)
    print('Saved figure to plots/comparison_Jbar_helicity_transversity.pdf')

    fig, axs = plt.subplots(4, 3, figsize=(25, 25))
    for j, obs in enumerate(obses):
        ax = axs[j // 3, j % 3]
        fac = -1 if obs in sign else 1
        ax.plot(q2_arr, amp_arr_hel[obs] + amp_arr_bar_hel[obs], label='Helicity Amplitudes', color='blue')
        ax.plot(q2_arr, fac * amp_arr_trans[obs] + amp_arr_bar_trans[obs], label='Transversity Amplitudes', color='red', linestyle='--')
        ax.set_xlabel(r'$q^2$ [GeV$^2$]')
        ax.set_ylabel(f'$J_{obs} + '+r'\bar{J}'+f'_{obs}$')
        ax.legend()
    fig.tight_layout()
    plt.savefig('plots/comparison_J_plus_Jbar_helicity_transversity.pdf', transparent=True)
    print('Saved figure to plots/comparison_J_plus_Jbar_helicity_transversity.pdf')

    fig, axs = plt.subplots(4, 3, figsize=(25, 25))
    for j, obs in enumerate(obses):
        ax = axs[j // 3, j % 3]
        fac = -1 if obs in sign else 1
        ax.plot(q2_arr, amp_arr_hel[obs] - amp_arr_bar_hel[obs], label='Helicity Amplitudes', color='blue')
        ax.plot(q2_arr, fac * (amp_arr_trans[obs] - amp_arr_bar_trans[obs]), label='Transversity Amplitudes', color='red', linestyle='--')
        ax.set_xlabel(r'$q^2$ [GeV$^2$]')
        ax.set_ylabel(f'$J_{obs} - '+r'\bar{J}'+f'_{obs}$')
        ax.legend()
    fig.tight_layout()
    plt.savefig('plots/comparison_J_minus_Jbar_helicity_transversity.pdf', transparent=True)
    print('Saved figure to plots/comparison_J_minus_Jbar_helicity_transversity.pdf')

    return None


if __name__ == "__main__":
    main()