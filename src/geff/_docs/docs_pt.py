DOCS ={
    "module":r"""
        $\newcommand{\bm}[1]{\boldsymbol{#1}}$

        This module is designed to compute the tensor power spectrum from a GEF result.

        The polarized tensor power spectrum at time $t$ with momentum $k$ is given by
        $$\mathcal{P}_{T}(t, k) = \mathcal{P}_{T}^{\mathrm{vac}}(t, k) + \frac{1}{2}\sum_{\lambda=\pm 1} \mathcal{P}_{T,\lambda}^{\mathrm{ind}}(t, k)\, ,$$
        where $\lambda$ labels the helicity of tensor modes.

        The vacuum contribution is given by

        $$\mathcal{P}_{T}^{\mathrm{vac}}(t, k) = \frac{4 k^3}{\pi^2 M_{\rm P}^2} |u_0(t, k)|^2\, ,$$

        while the induced contribution is

        $$\mathcal{P}_{T,\lambda}^{\mathrm{ind}}(t, k) = \frac{k^3}{2 \pi^2 M_{\rm P}^4} \int \frac{{\rm d}^3 \bm{p}}{(2 \pi)^3} \sum_{\alpha,\beta = \pm1} 
                \left(1 +  \lambda \alpha \frac{\bm{k} \cdot \bm{p}}{k p} \right)^2 \left(1 +  \lambda \beta \frac{k^2 - \bm{k} \cdot \bm{p}}{kq}  \right)^2 $$
        $$ \qquad \qquad \qquad \times \left|\int_{-\infty}^\infty {\rm d} s \frac{G_k(t, s)}{a^3(s)} 
            \left[A'_\alpha(s, p)A'_\beta(s, q) + \alpha \beta\, p q\, A_\alpha(s, p) A_\alpha(s, q) \right] \right|^2 \, ,
        $$

        with momentum $q = |\bm{p} + \bm{q}|$ and scale-factor $a$.

        The vacuum modes $u_0(t,k)$ obey the mode equation
        $$\mathcal{D}_k {u_0} = \ddot{u}_0 + 3 H \dot{u}_0 + \frac{k^2}{a^2} {u_0} = 0$$
        with the retarded Green function $G_k(t',t)$ defined for the operator $\mathcal{D}_k$.

        The gauge-field mode functions $A_\lambda(t,k)$ are defined as in the `geff.mode_by_mode` module.

        For details on the numerical computation, see the Appendix B of [2508.00798](https://arxiv.org/abs/2508.00798).
        """
}