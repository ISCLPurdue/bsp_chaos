import jax
import jax.numpy as jnp

# A jax version for binned spectral loss with 3D data.
# Chakraborty, D., Mohan, A. T., & Maulik, R. (2025).
# Binned spectral power loss for improved prediction of chaotic systems. arXiv preprint arXiv:2502.00472.
# Some minute modification are made to the original as per application.

def spectral_loss(pred, target, normalize=False, eps=1e-8):
    batch_size, timesteps, h, w, c = pred.shape

    def espec(input):
        N = input.shape[0]
        fourier_image = jnp.fft.fftn(input, axes=(0, 1))          # (H, W, C)
        fourier_amplitudes = 0.5 * jnp.abs(fourier_image) ** 2    # (H, W, C)

        kfreq = jnp.fft.fftfreq(N) * N
        kfreq2D = jnp.meshgrid(kfreq, kfreq)
        knrm = jnp.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2).flatten() # (H*W,)

        kbins = jnp.arange(0.5, N//2+1, 1.)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])

        fourier_amplitudes = fourier_amplitudes.reshape(-1, c)     # (H*W, C)

        def bin_one_channel(E_c):
            Abins = jnp.histogram(knrm, kbins, weights=E_c)[0] / jnp.histogram(knrm, kbins)[0]
            Abins *= kvals**2
            if normalize:
                Abins /= jax.scipy.integrate.trapezoid(Abins, kvals)
            return Abins

        return jax.vmap(bin_one_channel, in_axes=1)(fourier_amplitudes)  # (C, N_k)

    avg_espec = lambda x: jnp.mean(jax.vmap(espec, 0)(x), axis=0)  # (C, N_k)

    spec_pred   = jax.vmap(avg_espec, 1)(pred)    # (timesteps, C, N_k)
    spec_target = jax.vmap(avg_espec, 1)(target)

    loss = jnp.mean(
        jnp.mean(
            (1 - (spec_pred + eps) / (spec_target + eps)) ** 2,
            axis=(-2, -1),   # over C and N_k
        )
    )
    return loss