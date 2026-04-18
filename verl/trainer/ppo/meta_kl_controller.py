import torch
import torch.nn as nn


class EMANormaliser:
    def __init__(self, size: int, alpha: float = 0.01):
        self.alpha = alpha
        self.mean = torch.zeros(size)
        self.var = torch.ones(size)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        return self

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            delta = x.detach() - self.mean
            self.mean = self.mean + self.alpha * delta
            self.var = (1 - self.alpha) * (self.var + self.alpha * delta.pow(2))
        std = (self.var + 1e-6).sqrt()
        return torch.clamp((x - self.mean) / std, -5.0, 5.0)


class MetaKLController(nn.Module):
    STATE_FEATURES = ('kl_loss', 'reward_mean', 'reward_std', 'lagged_grad_norm')

    def __init__(self, cfg):
        super().__init__()

        assert cfg.min_beta < cfg.init_beta < cfg.max_beta, (
            f'Required min_beta < init_beta < max_beta, got {cfg.min_beta} / {cfg.init_beta} / {cfg.max_beta}'
        )

        if hasattr(cfg, 'state_features'):
            expected = set(self.STATE_FEATURES)
            got = set(cfg.state_features)
            assert got == expected, f'state_features mismatch. Expected {sorted(expected)}, got {sorted(got)}'

        self.input_size = len(self.STATE_FEATURES)
        self.hidden_size = cfg.lstm_hidden_dim
        self.min_beta = cfg.min_beta
        self.max_beta = cfg.max_beta
        self.warmup = cfg.warmup_steps
        self.alpha = cfg.ema_alpha
        self._step = 0
        self._beta_ema = float(cfg.init_beta)

        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.head = nn.Linear(self.hidden_size, 1)
        self._norm = EMANormaliser(self.input_size, alpha=0.01)

        self._h = None
        self._c = None

    def reset_hidden(self):
        self._h = None
        self._c = None
        self._step = 0

    def forward(self, state: dict, kl_loss: torch.Tensor) -> torch.Tensor:
        self._step += 1
        device = kl_loss.device
        dtype = kl_loss.dtype

        if self._step <= self.warmup:
            return kl_loss.new_tensor(self._beta_ema)

        x = torch.tensor([state.get(key, 0.0) for key in self.STATE_FEATURES], dtype=torch.float32, device=device)
        x_norm = self._norm.to(device).normalise(x)

        if self._h is None:
            self._h = torch.zeros(1, self.hidden_size, device=device)
            self._c = torch.zeros(1, self.hidden_size, device=device)
        else:
            self._h = self._h.to(device)
            self._c = self._c.to(device)

        h_new, c_new = self.lstm(x_norm.unsqueeze(0), (self._h, self._c))
        self._h = h_new.detach()
        self._c = c_new.detach()

        raw = self.head(h_new)
        beta_t = self.min_beta + (self.max_beta - self.min_beta) * torch.sigmoid(raw)
        beta_t = beta_t.squeeze().to(device=device, dtype=dtype)

        self._beta_ema = self.alpha * self._beta_ema + (1 - self.alpha) * beta_t.detach().item()
        return beta_t

    def get_log_dict(self) -> dict:
        return {
            'actor_adaptive_kl/beta_smoothed': self._beta_ema,
            'actor_adaptive_kl/in_warmup': int(self._step <= self.warmup),
        }
