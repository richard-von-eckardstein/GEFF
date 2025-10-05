DOCS = {
    "module":r"""
    This module contains several functions intended for computing and analyzing gravitational-wave spectra.

    The gravitational wave spectrum, $h^2\Omega_{\rm GW}$ can be computed using `omega_gw`, which evaluates the formula

    $$\Omega_{\rm GW}(f) \equiv  \frac{1}{3 H_0^2 M_{\rm P}^2}\frac{{\rm d} \rho_{\rm GW} (f)}{{\rm d} \ln{f}} = 
                                    \frac{\pi^2}{3 H_0^2} f^2 |\mathcal{T}_{\rm GW}(f)|^2 \mathcal{P}_T(k_f), k_f) 
                                    \, , \quad k_f = 2 \pi a_0 f \, ,$$

    where  $\mathcal{P}_T(k)$ is the tensor power spectrum with momentum $k$ at the end of inflation, and $H_0$ is the Hubble rate today.
    The transfer function $|\mathcal{T}_{\rm GW}(f)|^2$ is given by

    $$|\mathcal{T}_{\rm GW}(f)|^2 \simeq \frac{H_0^2\Omega_r}{8 \pi^2 f^2} \frac{g_{\*}(T_f)}{g_{\*}(T_0)} 
                                            \left(\frac{g_{\*,S}(T_0)}{g_{\*,S}(T_f)}\right)^{4/3}
                                            \left(1 + \frac{9}{16}\left(\frac{f_{\rm eq}}{\sqrt{2} f} \right)^2\right)
                                            |\mathcal{T}_{\rm reh}(f)|^2 \,.$$

    It accounts for the evolution of $\Omega_{\rm GW}(f)$ from the end of inflation until today.
    Here, $T_f$ is the temperature corresponding to the frequency $f$.
    For $g_{\*}$, $g_{\*,S}$, $f_{\rm eq}$, $\Omega_r$, and $T_0$, we use the corresponding functions in `GEFF.utility.cosmo`.
    The term $|\mathcal{T}_{\rm reh}(f)|^2$ accounts for the transition through reheating. For instantaneous reheating, $|\mathcal{T}_{\rm reh}(f)|^2 = 1$.
    Otherwise, we assume that reheating proceeds via coherent oscillations of the inflaton field, such that

    $$
    |\mathcal{T}_{\rm reh}(f)|^2 = \frac{ \theta(f_{\rm end} - f)}{1 - 0.22 \left(\frac{f}{f_{\rm reh}}\right)^{1.5} + 0.65 \left(\frac{f}{f_{\rm reh}}\right)^{2}} \, ,
    $$

    as given in [arXiv:1407.4785](https://arxiv.org/abs/1407.4785) and [arXiv:2011.03323](https://arxiv.org/abs/2011.03323). 
    Here, $f_{\rm reh}$ and $f_{\rm end}$ are, respectively, the frequencies at the end of reheating, and inflation.

    The frequency $f$ can be computed from a comoving momentum $k$ using `k_to_f` which evaluates

    $$
    f = \frac{k_f}{2 \pi a_0} =\frac{k_f}{2 \pi a_{\rm end}} e^{-N_{reh}} \left( \frac{g_{\*,S}(T_0)}{g_{\*,S}(T_{{\rm reh}})}\right)^{1/3} \frac{T_0}{T_{{\rm reh}}} \, .
    $$

    where

    $$
    N_{\rm reh} = \frac{1}{3(1 + w_{\rm reh})} \ln \left(\frac{90 M_{\rm P} H_{\rm end}^2}{\pi^2 g_*(T_{\rm reh}) T_{\rm reh}^4} \right) \, .
    $$
    and $T_{\rm reh}$ is the temperature of the SM plasma at the end of reheating.

    This module uses data provided by [zenodo:3689582](https://zenodo.org/records/3689582) and [zenodo:8092346](https://zenodo.org/records/8092346)
    for strain noise power spectra, $\Omega_{\rm noise}(f)$, to compute signal-to-noise ratios for several GW observatories.
    The relevant function is `compute_SNR`, which evaluates

    $$
    S/N = \left(n_{{\rm det}} t_{{\rm obs}} \int_{f_{{\rm min}}}^{f_{{\rm max}}} {\rm d} f \, \left( \frac{\Omega_{\rm signal}(f)}{\Omega_{\rm noise}(f)}\right)^2 \right)^{1/2}\, .
    $$

    Data on power-law integrated sensitivity curves (PLIS) from [zenodo:3689582](https://zenodo.org/records/3689582) can also be loaded using `get_plis`.
    Use `known_gw_obs` to learn which data is available for which observatories.
    """
}