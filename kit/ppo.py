"""ppo.py — PPO mínimo (ator-crítico, GAE, recorte, taxa de aprendizado adaptativa) para o kit da FTH 1.

Os nomes e os valores padrêo seguem o rl_cfg.py da tarefa de velocidade do G1 no mjlab
(github.com/mujocolab/mjlab, commit 8bb2bc9, 18/10/2026), que usa a biblioteca rsl-rl-lib 5.5.1.
Aqui tudo roda em CPU, com poucos ambientes: serve para ver o laço, nêo para chegar numa marcha.
"""
import torch
import torch.nn as nn

PADRAO = dict(clip_param=0.2, entropy_coef=0.01, value_loss_coef=1.0, num_learning_epochs=5,
              num_mini_batches=4, learning_rate=1.0e-3, desired_kl=0.01, gamma=0.99, lam=0.95,
              max_grad_norm=1.0, num_steps_per_env=24, init_std=1.0, hidden_dims=(512, 256, 128))


class Normalizador(nn.Module):
    """Média e desvio das observações vistas até agora (obs_normalization=True no mjlab)."""

    def __init__(self, n):
        super().__init__()
        self.register_buffer("media", torch.zeros(n))
        self.register_buffer("var", torch.ones(n))
        self.register_buffer("cont", torch.tensor(1e-4))

    def atualizar(self, x):
        b_media, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_media - self.media
        tot = self.cont + b_n
        self.media += delta * b_n / tot
        self.var = (self.var * self.cont + b_var * b_n + delta ** 2 * self.cont * b_n / tot) / tot
        self.cont = tot

    def forward(self, x):
        return (x - self.media) / torch.sqrt(self.var + 1e-8)


def mlp(n_in, n_out, ocultas):
    camadas, n = [], n_in
    for h in ocultas:
        camadas += [nn.Linear(n, h), nn.ELU()]
        n = h
    camadas.append(nn.Linear(n, n_out))
    return nn.Sequential(*camadas)


class AtorCritico(nn.Module):
    def __init__(self, n_obs, n_acao, ocultas=PADRAO["hidden_dims"], init_std=PADRAO["init_std"]):
        super().__init__()
        self.norm = Normalizador(n_obs)
        self.ator = mlp(n_obs, n_acao, ocultas)
        self.critico = mlp(n_obs, 1, ocultas)
        self.std = nn.Parameter(torch.full((n_acao,), float(init_std)))

    def dist(self, obs):
        x = self.norm(obs)
        return torch.distributions.Normal(self.ator(x), self.std.clamp(min=1e-3)), self.critico(x).squeeze(-1)

    def acao_media(self, obs):
        return self.ator(self.norm(obs))


class PPO:
    def __init__(self, rede, cfg=PADRAO):
        self.rede, self.c = rede, dict(cfg)
        self.lr = self.c["learning_rate"]
        self.opt = torch.optim.Adam(rede.parameters(), lr=self.lr)

    def vantagens(self, rec, val, fim, val_final):
        """GAE (gamma, lam): quanto cada açêo foi melhor que o esperado pelo crítico."""
        T = rec.shape[0]
        adv = torch.zeros_like(rec)
        ultimo = torch.zeros_like(val_final)
        for t in reversed(range(T)):
            prox = val_final if t == T - 1 else val[t + 1]
            nao_fim = 1.0 - fim[t]
            delta = rec[t] + self.c["gamma"] * prox * nao_fim - val[t]
            ultimo = delta + self.c["gamma"] * self.c["lam"] * nao_fim * ultimo
            adv[t] = ultimo
        return adv, adv + val

    def atualizar(self, obs, acoes, logp_ant, adv, ret, val_ant, mu_ant, std_ant):
        c = self.c
        N = obs.shape[0]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        tam = N // c["num_mini_batches"]
        soma = {"perda_ator": 0.0, "perda_critico": 0.0, "kl": 0.0}
        n = 0
        for _ in range(c["num_learning_epochs"]):
            ordem = torch.randperm(N)
            for i in range(c["num_mini_batches"]):
                idx = ordem[i * tam:(i + 1) * tam]
                dist, valor = self.rede.dist(obs[idx])
                logp = dist.log_prob(acoes[idx]).sum(-1)
                # KL entre a política antiga e a nova: controla a taxa de aprendizado (schedule="adaptive")
                with torch.no_grad():
                    kl = (torch.log(dist.stddev / std_ant[idx]) + (std_ant[idx] ** 2 + (mu_ant[idx] - dist.mean) ** 2)
                          / (2 * dist.stddev ** 2) - 0.5).sum(-1).mean()
                    if kl > 2 * c["desired_kl"]:
                        self.lr = max(1e-5, self.lr / 1.5)
                    elif 0 < kl < c["desired_kl"] / 2:
                        self.lr = min(1e-2, self.lr * 1.5)
                    for g in self.opt.param_groups:
                        g["lr"] = self.lr
                razao = torch.exp(logp - logp_ant[idx])
                s1 = razao * adv[idx]
                s2 = torch.clamp(razao, 1 - c["clip_param"], 1 + c["clip_param"]) * adv[idx]
                perda_ator = -torch.min(s1, s2).mean()
                v_rec = val_ant[idx] + (valor - val_ant[idx]).clamp(-c["clip_param"], c["clip_param"])
                perda_critico = torch.max((valor - ret[idx]) ** 2, (v_rec - ret[idx]) ** 2).mean()
                perda = perda_ator + c["value_loss_coef"] * perda_critico - c["entropy_coef"] * dist.entropy().sum(-1).mean()
                self.opt.zero_grad()
                perda.backward()
                nn.utils.clip_grad_norm_(self.rede.parameters(), c["max_grad_norm"])
                self.opt.step()
                soma["perda_ator"] += perda_ator.item()
                soma["perda_critico"] += perda_critico.item()
                soma["kl"] += kl.item()
                n += 1
        return {k: v / n for k, v in soma.items()}
