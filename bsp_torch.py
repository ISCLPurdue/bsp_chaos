import torch 

# A torch version for binned spectral loss with 3D data.
# Chakraborty, D., Mohan, A. T., & Maulik, R. (2025).
# Binned spectral power loss for improved prediction of chaotic systems. arXiv preprint arXiv:2502.00472.
# Some minute modification are made to the original as per application.

def spec_loss(x, y, eps=1e-8, drop_bins=0):
    def espec_batch(batch_data):
        """
        Compute the mean energy spectrum across a batch and channels for 3D data.

        Parameters:
            batch_data (torch.Tensor): Input tensor of shape (B, C, H, W, L).

        Returns:
            torch.Tensor: Mean binned amplitudes (Abins) across the batch, shape (C, num_bins).
        """
        L = 2 * torch.pi
        B, C, ny, nx, nz = batch_data.shape
        N = min(ny, nx, nz)
        delta_x = L / N
        k_nq = torch.pi / delta_x

        # Compute the Fourier transform and amplitude squared for the batch
        fourier_image = torch.fft.fftn(batch_data, dim=(-3, -2, -1))       # (B, C, H, W, L)
        fourier_amplitudes = 0.5 * torch.abs(fourier_image) ** 2             # (B, C, H, W, L)

        # Create the k-frequency grid
        kfreq_y = torch.fft.fftfreq(ny, device=batch_data.device) * ny
        kfreq_x = torch.fft.fftfreq(nx, device=batch_data.device) * nx
        kfreq_z = torch.fft.fftfreq(nz, device=batch_data.device) * nz
        kfreq2D_x, kfreq2D_y, kfreq2D_z = torch.meshgrid(kfreq_x, kfreq_y, kfreq_z, indexing='ij')
        knrm = torch.sqrt(kfreq2D_x**2 + kfreq2D_y**2 + kfreq2D_z**2)       # (H, W, L)
        knrm = knrm.flatten()                                                  # (H * W * L)

        # Flatten Fourier amplitudes
        fourier_amplitudes = fourier_amplitudes.view(B, C, -1)                # (B, C, H * W * L)

        # Define the bins for the wavenumber
        kbins = torch.arange(0.5, N // 2 + 1, 1., device=batch_data.device)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        kvals = kvals / (0.5 * N) * k_nq

        # Bin the data for each image and channel in the batch
        Abins_batch = []
        for i in range(len(kbins) - 1):
            mask = (knrm >= kbins[i]) & (knrm < kbins[i + 1])                # (H * W * L,)
            if mask.sum() > 0:
                binned_values = torch.mean(fourier_amplitudes[:, :, mask], dim=-1)  # (B, C)
            else:
                binned_values = torch.zeros((B, C), device=batch_data.device)
            Abins_batch.append(binned_values)

        # Stack and scale the binned amplitudes
        Abins_batch = torch.stack(Abins_batch, dim=-1)                        # (B, C, num_bins)
        bin_areas = torch.pi * (kbins[1:]**3 - kbins[:-1]**3)                # 3D shell volume
        Abins_batch *= bin_areas

        # Average over batch only, keep C
        Abins_mean = Abins_batch.mean(dim=0)                                  # (C, num_bins)

        return Abins_mean

    # Compute energy spectra for x and y
    x_spec = espec_batch(x)
    y_spec = espec_batch(y)

    # Optionally drop the last N bins
    if drop_bins > 0:
        x_spec = x_spec[:, :-drop_bins]
        y_spec = y_spec[:, :-drop_bins]

    # Compute the spectral loss
    loss = torch.mean((1 - (x_spec + eps) / (y_spec + eps)) ** 2)

    return loss