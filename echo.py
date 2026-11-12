"""ECHO pattern forecasting, adapted from the study implementation."""
from copy import deepcopy
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F

WINDOW = 6
HIDDEN = 12
EPOCHS = 160
MASKS = np.array([[(i >> j) & 1 for j in range(9)] for i in range(1, 512)], dtype=np.float64)
PAIR_I, PAIR_J = np.triu_indices(9, 1)
PAIR_FEATURES = MASKS[:, PAIR_I] * MASKS[:, PAIR_J]

def make_context(n, y, h, a, train_end, exposure=True):
    """Fit graph, priors and scaling exclusively within the training calendar."""
    total_a = a[:train_end].sum()
    family_totals = y[:train_end].sum(axis=0)
    q_prior = (family_totals + 0.5) / (total_a + 1.0)
    prevalence_prior = (total_a + 0.5) / (n[:train_end].sum() + 1.0)
    exposure_prior = np.median(n[:train_end]) * 0.02
    pattern_prior = (h[:train_end].sum(axis=0) + 0.5) / (total_a + 255.5)
    cooccur = np.einsum('p,pi,pj->ij', h[:train_end].sum(axis=0), MASKS, MASKS)
    together = cooccur[PAIR_I, PAIR_J]
    solo_i = family_totals[PAIR_I] - together
    solo_j = family_totals[PAIR_J] - together
    neither = total_a - family_totals[PAIR_I] - family_totals[PAIR_J] + together
    log_odds = np.log((together + 0.5) * (neither + 0.5) /
                      ((solo_i + 0.5) * (solo_j + 0.5)))
    interaction = np.clip(log_odds, -2, 2) * together / (together + 20) * 0.25
    q = (y + 2 * q_prior) / (a[:, None] + 2)
    q = np.clip(q, 1e-7, 1 - 1e-7)
    z = (a + exposure_prior * prevalence_prior) / (n + exposure_prior)
    z = np.clip(z, 1e-8, 1 - 1e-8)
    baseline = np.log(q / (1 - q))
    baseline_z = np.log(z / (1 - z)) if exposure else np.log(a + 0.5)
    # The next prediction uses the previous observed pattern distribution.
    pattern_log = np.log((h + 2 * pattern_prior) / (a[:, None] + 2))
    month_number = np.arange(len(n), dtype=float)
    calendar = np.column_stack([month_number / 54,
                                np.sin(2 * np.pi * month_number / 12),
                                np.cos(2 * np.pi * month_number / 12)])
    if exposure:
        observed_features = np.column_stack([baseline, baseline_z, np.log(n), calendar])
    else:
        observed_features = np.column_stack([baseline, baseline_z, calendar])
    scaler = StandardScaler().fit(observed_features[:train_end])
    x = scaler.transform(observed_features)
    targets = np.arange(WINDOW, len(n))
    sequence = np.stack([x[t - WINDOW:t] for t in targets])
    # Target calendar is deterministic; no contemporaneous target counts enter.
    target_calendar = calendar[targets]
    return dict(x=torch.tensor(sequence, dtype=torch.float32),
                calendar=torch.tensor(target_calendar, dtype=torch.float32),
                base=torch.tensor(baseline[targets - 1], dtype=torch.float32),
                base_z=torch.tensor(baseline_z[targets - 1], dtype=torch.float32),
                prev_log=torch.tensor(pattern_log[targets - 1], dtype=torch.float32),
                targets=targets, interaction=interaction, cooccur=cooccur,
                pattern_prior=pattern_prior, q_prior=q_prior,
                z_prior=float(prevalence_prior), exposure_prior=float(exposure_prior),
                scale_mean=scaler.mean_.tolist(), scale_scale=scaler.scale_.tolist(),
                exposure=exposure, train_end=train_end)


class Echo(nn.Module):
    def __init__(self, context, graph=True, lam=0.15):
        super().__init__()
        self.encoder = nn.GRU(context['x'].shape[2], HIDDEN, batch_first=True)
        self.head = nn.Sequential(nn.Linear(HIDDEN + 3, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 10))
        # Start at the empirical-Bayes continuation; learn its innovation.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('patterns', torch.tensor(MASKS, dtype=torch.float32))
        energy = PAIR_FEATURES @ context['interaction'] if graph else np.zeros(511)
        self.register_buffer('graph_energy', torch.tensor(energy, dtype=torch.float32))
        self.lam = float(lam)
        self.exposure = context['exposure']

    def forward(self, ctx, ids):
        _, hidden = self.encoder(ctx['x'][ids])
        innovation = self.head(torch.cat([hidden[-1], ctx['calendar'][ids]], dim=1))
        fields = ctx['base'][ids] + innovation[:, :9]
        log_u = F.log_softmax(fields @ self.patterns.T + self.graph_energy, dim=1)
        log_pattern = F.log_softmax((log_u + self.lam * ctx['prev_log'][ids]) /
                                   (1 + self.lam), dim=1)
        z_parameter = ctx['base_z'][ids] + innovation[:, 9]
        marginal = log_pattern.exp() @ self.patterns
        probability = log_pattern.exp()
        if not torch.allclose(probability.sum(dim=1), torch.ones(len(ids)), atol=2e-6, rtol=2e-6):
            raise RuntimeError('Pattern probabilities do not sum to one')
        if not torch.all((marginal >= -2e-6) & (marginal <= 1 + 2e-6)):
            raise RuntimeError('Conditional family marginal falls outside [0, 1]')
        if not torch.all(marginal.sum(dim=1) >= 1 - 2e-6):
            raise RuntimeError('Nonempty-pattern conditional marginals sum below one')
        return z_parameter, log_pattern, marginal


class PlainGRU(nn.Module):
    def __init__(self, xdim):
        super().__init__()
        self.encoder = nn.GRU(xdim, HIDDEN, batch_first=True)
        self.head = nn.Sequential(nn.Linear(HIDDEN + 3, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 9))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, ctx, ids, rate_base):
        _, hidden = self.encoder(ctx['x'][ids])
        innovation = self.head(torch.cat([hidden[-1], ctx['calendar'][ids]], dim=1))
        return rate_base[ids] + innovation


def deviance(y, mu):
    mu = np.maximum(np.asarray(mu, float), 1e-10)
    y = np.asarray(y, float)
    positive = y > 0
    term = mu - y
    term[positive] += y[positive] * np.log(y[positive] / mu[positive])
    return float(2 * np.mean(term))


def train_model(method, seed, n, y, h, a, train_end=36, val_end=42,
                lam=0.15, epochs=EPOCHS, use_validation=True, history=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    exposure = method != 'ECHO without exposure'
    ctx = make_context(n, y, h, a, train_end, exposure)
    train_ids = np.flatnonzero(ctx['targets'] < train_end)
    validation_ids = np.flatnonzero((ctx['targets'] >= train_end) & (ctx['targets'] < val_end))
    target_n = torch.tensor(n[ctx['targets']], dtype=torch.float32)
    target_a = torch.tensor(a[ctx['targets']], dtype=torch.float32)
    target_h = torch.tensor(h[ctx['targets']], dtype=torch.float32)
    target_y = torch.tensor(y[ctx['targets']], dtype=torch.float32)
    rate_prior = (y[:train_end].sum(axis=0) + 0.5) / (n[:train_end].sum() + 1)
    smooth_rate = (y + ctx['exposure_prior'] * rate_prior) / (n[:, None] + ctx['exposure_prior'])
    smooth_rate = np.clip(smooth_rate, 1e-8, 1 - 1e-8)
    rate_base = torch.tensor(np.log(smooth_rate[ctx['targets'] - 1] /
                                   (1 - smooth_rate[ctx['targets'] - 1])), dtype=torch.float32)
    model = (PlainGRU(ctx['x'].shape[2]) if method == 'Plain GRU' else
             Echo(ctx, graph=method not in ['ECHO without graph', 'Independent mask GRU'],
                  lam=0 if method in ['ECHO without entropy', 'Independent mask GRU'] else lam))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=0.002)
    best, best_epoch, best_state = float('inf'), epochs, None
    normalizer = max(float(target_a[train_ids].sum()), 1)
    start = time.perf_counter()

    def predict(ids):
        with torch.no_grad():
            if method == 'Plain GRU':
                rates = torch.sigmoid(model(ctx, ids, rate_base))
                return (rates * target_n[ids, None]).numpy(), None
            zp, logp, marginal = model(ctx, ids)
            volume = torch.sigmoid(zp) * target_n[ids] if exposure else torch.exp(zp.clamp(-20, 15))
            return (volume[:, None] * marginal).numpy(), logp.numpy()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        if method == 'Plain GRU':
            logits = model(ctx, train_ids, rate_base)
            loss = (target_n[train_ids, None] * F.softplus(logits) - target_y[train_ids] * logits).sum() / normalizer
        else:
            zp, logp, _ = model(ctx, train_ids)
            if exposure:
                volume_loss = (target_n[train_ids] * F.softplus(zp) - target_a[train_ids] * zp).sum()
            else:
                volume_loss = (torch.exp(zp.clamp(-20, 15)) - target_a[train_ids] * zp).sum()
            pattern_loss = -(target_h[train_ids] * logp).sum()
            loss = (volume_loss + pattern_loss) / normalizer
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss for {method} at epoch {epoch}')
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError(f'Nonfinite gradient for {method} at epoch {epoch}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            if use_validation:
                pred, _ = predict(validation_ids)
                validation_deviance = deviance(y[ctx['targets'][validation_ids]], pred)
                if validation_deviance < best - 1e-8:
                    best, best_epoch, best_state = validation_deviance, epoch, deepcopy(model.state_dict())
            else:
                validation_deviance = float('nan')
            if history is not None:
                history.append(dict(method=method, seed=seed, training_months=train_end,
                                    lambda_entropy=lam if method != 'ECHO without entropy' else 0,
                                    epoch=epoch, training_loss=float(loss.detach()),
                                    validation_deviance=validation_deviance,
                                    elapsed_seconds=time.perf_counter() - start))
    if use_validation and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    mu, logp = predict(np.arange(len(ctx['targets'])))
    pred_all = np.full_like(y, np.nan)
    pred_all[ctx['targets']] = mu
    pattern_all = np.full_like(h, np.nan)
    if logp is not None:
        pattern_all[ctx['targets']] = logp
    return pred_all, pattern_all, dict(method=method, seed=seed, best_epoch=best_epoch,
                                     validation_deviance=best if use_validation else None,
                                     parameter_count=sum(p.numel() for p in model.parameters()),
                                     context={k: ctx[k] for k in ['scale_mean', 'scale_scale',
                                               'z_prior', 'exposure_prior']})
