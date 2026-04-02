import math
import numpy as np
import torch
import torch.nn as nn


def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == "linear":
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == "warm0.1":
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == "warm0.2":
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == "warm0.5":
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas.astype(np.float64)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device).float() / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class DenoiserMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim=1024, time_dim=256):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim + time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb], dim=1)
        return self.net(h)


class GaussianDiffusion1D:
    def __init__(self, betas):
        assert isinstance(betas, np.ndarray)
        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.alphas_cumprod = torch.from_numpy(alphas_cumprod).float()
        self.alphas_cumprod_prev = torch.from_numpy(alphas_cumprod_prev).float()
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0).float()

        betas_t = torch.from_numpy(betas).float()
        alphas_t = torch.from_numpy(alphas).float()
        posterior_variance = betas_t * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(
            torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance))
        )
        self.posterior_mean_coef1 = betas_t * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas_t) / (1.0 - self.alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        bs = t.shape[0]
        out = torch.gather(a, 0, t)
        return torch.reshape(out, [bs] + [1] * (len(x_shape) - 1))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )

    def p_mean_variance(self, denoise_fn, x, t):
        eps = denoise_fn(x, t)
        x_recon = (
            self._extract(self.sqrt_recip_alphas_cumprod.to(x.device), t, x.shape) * x
            - self._extract(self.sqrt_recipm1_alphas_cumprod.to(x.device), t, x.shape) * eps
        )
        model_mean = (
            self._extract(self.posterior_mean_coef1.to(x.device), t, x.shape) * x_recon
            + self._extract(self.posterior_mean_coef2.to(x.device), t, x.shape) * x
        )
        model_variance = self._extract(self.posterior_variance.to(x.device), t, x.shape)
        model_log_variance = self._extract(self.posterior_log_variance_clipped.to(x.device), t, x.shape)
        return model_mean, model_variance, model_log_variance

    def p_sample(self, denoise_fn, x, t):
        model_mean, _, model_log_variance = self.p_mean_variance(denoise_fn, x, t)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(x.shape[0], *([1] * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    def p_sample_loop(self, denoise_fn, shape, device):
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.num_timesteps)):
            t_b = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(denoise_fn, x, t_b)
        return x

    def p_losses(self, denoise_fn, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        eps_pred = denoise_fn(x_t, t)
        return torch.mean((noise - eps_pred) ** 2, dim=1)
