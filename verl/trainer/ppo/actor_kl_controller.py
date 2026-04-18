import torch


class RuleBasedActorKLController:
    def __init__(self, cfg):
        assert cfg.min_beta < cfg.init_beta < cfg.max_beta, (
            f'Required min_beta < init_beta < max_beta, got {cfg.min_beta} / {cfg.init_beta} / {cfg.max_beta}'
        )

        self._beta = float(cfg.init_beta)
        self.min_beta = float(cfg.min_beta)
        self.max_beta = float(cfg.max_beta)
        self.target = float(cfg.target_kl_loss)
        self.up = float(cfg.up_gain)
        self.down = float(cfg.down_gain)
        self.alpha = float(cfg.ema_alpha)
        self.warmup = int(cfg.warmup_steps)
        self.interval = int(cfg.update_interval)
        self._step = 0
        self._beta_ema = float(cfg.init_beta)

    def __call__(self, state: dict, kl_loss: torch.Tensor) -> torch.Tensor:
        self._step += 1
        if self._step <= self.warmup or self._step % self.interval != 0:
            return kl_loss.new_tensor(self._beta_ema)

        kl_value = float(state.get('kl_loss', 0.0))
        raw = self._beta * (self.up if kl_value > self.target else self.down)
        raw = max(self.min_beta, min(self.max_beta, raw))
        self._beta_ema = self.alpha * self._beta_ema + (1 - self.alpha) * raw
        self._beta = self._beta_ema
        return kl_loss.new_tensor(self._beta_ema)

    def reset(self):
        self._step = 0

    def get_log_dict(self) -> dict:
        return {'actor_adaptive_kl/beta_smoothed': self._beta_ema}
