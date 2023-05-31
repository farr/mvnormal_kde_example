import jax
import jax.numpy as jnp
import jax.scipy.special as jss
import numpy as np
import numpyro
import numpyro.distributions as dist

def gaussian_marginalization(samples, kde_bws, mu, sigmas):
    n_normal = mu.shape[0]
    nobs, nsamp, ndim = samples.shape

    assert kde_bws.shape == (nobs, ndim, ndim)

    mu_full = jnp.concatenate((mu, jnp.zeros(ndim-n_normal)))

    eye_full = jnp.eye(ndim)
    upper = eye_full[:, :n_normal]
    lower = eye_full[:, n_normal:]

    # We only need the upper (n_normal, n_normal) corner of Lambda to be non-zero
    Lambda = upper @ jnp.diag(jnp.square(sigmas)) @ upper.T 
    # Lambda_upper = Lambda_upper + lower @ jnp.eye(ndim-n_normal) @ lower.T

    T_axes = (0, 2, 1)
    kb_T = jnp.transpose(kde_bws, axes=T_axes)

    Q = eye_full + jnp.transpose(jnp.linalg.solve(kb_T, Lambda[jnp.newaxis, ...]), axes=T_axes)

    mue = mu_full[:, jnp.newaxis]

    U = lower
    VT = jnp.transpose(jnp.linalg.solve(kb_T, lower[jnp.newaxis, ...]), T_axes)
    VTQIU = VT @ jnp.linalg.solve(Q, U[jnp.newaxis, ...])
    QI_mu = jnp.linalg.solve(Q, mue[jnp.newaxis, ...])

    mu_term = (QI_mu - U @ jnp.linalg.solve(VTQIU, VT @ QI_mu))[...,0]



    es = samples[..., jnp.newaxis]
    kb = kde_bws[..., jnp.newaxis, :, :]
    U = upper
    UTe = U.T[jnp.newaxis, jnp.newaxis, :, :]

    samples_term = (es - kb @ U @ jnp.linalg.solve(jnp.diag(jnp.square(sigmas)) + U.T @ kb @ U, U.T @ es))[..., 0]

    a = mu_term[..., jnp.newaxis, :] + samples_term
    A = kb - kb @ U @ jnp.linalg.solve(jnp.diag(jnp.square(sigmas)) + U.T @ kb @ U, UTe) @ kb

    B = kde_bws[..., :n_normal, :n_normal] + jnp.diag(jnp.square(sigmas))

    return a, A, B

def marginal_normal_model(log_p_other, samples, prior_wts, n_normal, *args, samples_sel=None, pdraw_sel=None, n_draw_sel=None, predictive=False, **kwargs):
    r"""
    Model that marginalizes out Gaussian populations in some dimensions using a
    KDE approximation to the likelihood function.

    The model uses an independent Gaussian population for the first `n_normal`
    components of `samples`:

    .. math::
        \theta_i \sim N\left( \mu_i, \sigma_i \right)
    
    for :math:`0 \leq i < N_\mathrm{normal}` with :math:`\mu_i` and
    :math:`\sigma_i` the mean and standard deviation of the population for the
    `i`th coordinate.

    The model uses a Gaussian KDE over the provided `samples` to represent the
    likelihood function, which enables analytically marginalizing out the
    parameters in the Gaussian dimensions.  

    The population model in the other dimensions (:math:`N_\mathrm{normal} \leq
    i < N`) is encapsulated in `log_p_other`, which will be called with the
    vector of these parameters inferred after marginalizing out the Gaussian
    dimensions.

    If the optional arguments `samples_sel`, `pdraw_sel`, and `n_draw_sel` are
    given, then the model corrects for selection effects, assuming that
    `samples_sel` are samples of the parameters drawn according to the
    (normalized) density `pdraw_sel` that have been passed through an
    observational model and selected according to the same pipeline / selection
    function that produced the catalog under analysis.  `n_draw_sel` records how
    many total draws from the population were required to produce the
    `samples_sel` using the selection function.  See [Farr
    (2019)](https://ui.adsabs.harvard.edu/abs/2019RNAAS...3...66F/abstract) for
    details and formulae.

    Setting the optional argument `predictive=True` will produce samples of
    "generated quantities" based on a previous sampling of the latent parameters
    in the model.  The quantities generated and their names are:

    * `theta`: A fair draw of parameters from the posterior over each event
      conditioned on the current sample's population model; the first `n_normal`
      components are the Gaussian-population dimensions.  
    * `theta_gaussian_draw`: A single draw from the Gaussian population for the
      Gaussian population dimensions.
    * `R`: if `samples_sel` are given, then this is a draw from the "rate"
      (normalizing factor) given the population model; see [Farr
      (2019)](https://ui.adsabs.harvard.edu/abs/2019RNAAS...3...66F/abstract)
      for details.

    To produce such draws using a MCMC chain, use code like 

    .. code-block:: python
        import jax
        from numpyro.infer import MCMC, Predictive

        # Set up your model, which includes marginal_normal_model as a sub-model
        # Your model should pass the `predictive` kwarg along to #
        `marginal_normal_model` when it calls it. def full_model_function(arg1,
        arg2, ..., predictive=False, ...):
            ... marginal_normal_model(..., predictive=predictive) ...
            
        # Initialize and run an MCMC mcmc = MCMC(...) mcmc.run

        predictive = Predictive(full_model_function, mcmc.get_samples()) pred =
        predictive(jax.random.PRNGKey(42), full_model_argument_1,
        full_model_argument_2, ..., predictive=True)

        # You can now access generated draws from the posterior over, e.g.,
        `theta` via pred['theta'] # shape is (nmcmc_draws, nobs, ndim)

    :param log_p_other: Function implementing the non-Gaussian part of the
        population model.  Called with parameters `theta[n_normal:]` inferred
        after marginalizing out the Gaussian part of the population.  Should
        compute the population log-density over these components; may include
        additional `numpyro.sample(...)` statements represeting additional
        population parameters to be inferred.

    :param samples: Array of shape `(nobs, nsamp, ndim)` giving `nsamp`
        posterior samples of the `ndim` event-level parameters from individual
        analysis of the `nobs` observations.  The first `n_normal` dimensions of
        these parameters are assumed to follow a Gaussian population model.
        These samples are passed to a Gaussian KDE with Scott's rule bandwidth
        for each event to represent the event likelihood (densities) for the
        population analysis

    :param prior_wts: Array of shape `(nobs, nsamp)` giving the prior density
        (up to arbitrary normalization constant) at each sample from which the
        posterior samples `samples` were produced.

    :param n_normal: The number of dimensions for which a Gaussian population
        should be imposed.  Parameters `theta[i]` for `0 <= i < n_normal` will
        have a Gaussian population model.

    :param predictive: (default `False`).  Whether to produce samples over the
        "generated quantities" as described above.

    :param samples_sel: (default `None`).  If provided an array of shape `(nsel,
        ndim)` of true parameters of systems that have been drawn from the
        normalized density `pdraw_sel`, injected into noise realizations, and
        detected by the same pipeline / selection function that produced the
        catalog under analysis.

    :param pdraw_sel: (default `None`).  If provided, array of shape `(nsel,)`
        giving the (normalized) density from which the detected `samples_sel`
        were drawn before being passed through the pipeline / selection
        function.

    :param n_draw_sel: (default `None`).  If provided, the total number of draws
        from the density represented by `pdraw_sel` which produced the detected
        samples `samples_sel`.
    """
    samples = np.atleast_3d(samples)
    prior_wts = np.atleast_2d(prior_wts)

    nobs, nsamp, ndim = samples.shape

    wts = 1/prior_wts
    log_wts = np.log(wts)

    mu = np.mean(samples, axis=-2)
    sigma = np.sum((samples[:, :, :, np.newaxis] - mu[:, np.newaxis, :, np.newaxis])*(samples[:, :, np.newaxis, :] - mu[:, np.newaxis, np.newaxis, :]), axis=-3) / nsamp

    kde_bw = sigma / nsamp**(2/(4+ndim))

    mu = numpyro.sample('mu', dist.Normal(0,1), sample_shape=(n_normal,))
    sigmas = numpyro.sample('sigma', dist.HalfNormal(1), sample_shape=(n_normal,))

    a, A, B = gaussian_marginalization(samples, kde_bw, mu, sigmas)

    logp_normal = dist.MultivariateNormal(mu, B[..., jnp.newaxis, :, :]).log_prob(samples[..., :n_normal])
    logp_other_0 = log_p_other(a[..., n_normal:])

    logp_total = logp_normal + logp_other_0 + log_wts
    logp_sum = jss.logsumexp(logp_total, axis=-1)

    neff = numpyro.deterministic('neff', jnp.exp(2*logp_sum - jss.logsumexp(2*logp_total, axis=-1)))

    numpyro.factor('likelihood', jnp.sum(logp_sum))

    if samples_sel is not None:
        nsel, _ = samples_sel.shape
        cov_sel = np.cov(samples_sel, rowvar=False)
        ccov_sel = np.linalg.inv(np.linalg.inv(cov_sel)[:n_normal, :n_normal])
        bw_sel = ccov_sel / nsel**(2/(4+n_normal))

        B_sel = bw_sel + jnp.diag(jnp.square(sigmas))

        logp_normal_sel = dist.MultivariateNormal(mu, B_sel).log_prob(samples_sel[:, :n_normal])
        logp_other_sel = log_p_other(samples_sel[:, n_normal:])

        logp_total_sel = logp_normal_sel + logp_other_sel - jnp.log(pdraw_sel)

        log_mu_sel = jss.logsumexp(logp_total_sel) - jnp.log(n_draw_sel)
        log_m2_sel = jss.logsumexp(2*logp_total_sel) - 2*jnp.log(n_draw_sel)
        log_s2_sel = log_m2_sel + jnp.log1p(-jnp.exp(2*log_mu_sel - jnp.log(n_draw_sel) - log_m2_sel))
        neff_sel = numpyro.deterministic('neff_sel', jnp.exp(2*log_mu_sel - log_s2_sel))

        _ = numpyro.factor('selection_norm', -nobs*log_mu_sel)

    if predictive:
        inds = numpyro.sample('inds', dist.CategoricalLogits(logp_total))
        a_i = a[jnp.arange(nobs), inds, :]
        A_i = A[jnp.arange(nobs), inds, :, :]
        theta = numpyro.sample('theta', dist.MultivariateNormal(a_i, A_i))

        theta_gaussian_draw = numpyro.sample('theta_gaussian_draw', dist.Normal(mu, sigmas))

        if samples_sel is not None:
            mu = jnp.exp(log_mu_sel)
            R = numpyro.sample('R', dist.Normal(nobs/mu, np.sqrt(nobs)/mu))
