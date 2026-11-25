"""treinar.py — treino curto de PPO, em CPU, com alguns G1 em paralelo no MuJoCo (aula 5).

Uso: python3 kit/treinar.py --iters 20 [--envs 16] [--seed 0] [--saida ~/fth/treino/ppo] [--recompensa kit/recompensa.yaml]
Cada iteraçêo: cada ambiente dá 24 passos (num_steps_per_env do mjlab) → GAE → 5 épocas × 4 mini-lotes.
Grava progresso.csv (uma linha por iteraçêo) e o checkpoint modelo_<iters>.pt.
Em CPU, 20 iterações nêo ensinam o G1 a andar: o que se confere é o laço, nêo a marcha.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fth  # noqa: E402
from ppo import PADRAO, PPO, AtorCritico  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--envs", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--saida", default=str(fth.RAIZ / "treino/ppo"))
ap.add_argument("--recompensa", default=None)
a = ap.parse_args()

torch.manual_seed(a.seed)
torch.set_num_threads(1)
saida = Path(os.path.expanduser(a.saida))
saida.mkdir(parents=True, exist_ok=True)
cfg_r = fth.ler_recompensa(a.recompensa)
envs = [fth.Ambiente(recompensa=cfg_r, seed=a.seed * 1000 + i) for i in range(a.envs)]
n_obs, n_acao = envs[0].n_obs, envs[0].n_acao
rede = AtorCritico(n_obs, n_acao)
ppo = PPO(rede)
T = PADRAO["num_steps_per_env"]
print(f"PPO em CPU: {a.envs} ambientes × {T} passos = {a.envs * T} amostras por iteraçêo · obs {n_obs} → açêo {n_acao}")
obs = torch.as_tensor(np.stack([e.reset() for e in envs]))
ret_ep = np.zeros(a.envs)
len_ep = np.zeros(a.envs, int)
termos_ep = [dict.fromkeys(cfg_r, 0.0) for _ in envs]
campos = ["iteracao", "recompensa_media", "episodio_medio", "episodios", "quedas", "vx_media", "std_acao", "lr", "kl", "segundos"] + list(cfg_r)
arq_csv = open(saida / "progresso.csv", "w", newline="")
w = csv.DictWriter(arq_csv, fieldnames=campos)
w.writeheader()
t_ini = time.time()
for it in range(1, a.iters + 1):
    t0 = time.time()
    B = {k: [] for k in ("obs", "acao", "logp", "val", "rec", "fim", "mu", "std")}
    fins, quedas, vxs = [], 0, []
    soma_termos = dict.fromkeys(cfg_r, 0.0)
    for _ in range(T):
        rede.norm.atualizar(obs)
        with torch.no_grad():
            dist, val = rede.dist(obs)
            acao = dist.sample()
        prox, rec, fim = [], np.zeros(a.envs), np.zeros(a.envs)
        for i, e in enumerate(envs):
            o, r, causa, termos = e.step(acao[i].numpy())
            rec[i] = r
            ret_ep[i] += r
            len_ep[i] += 1
            vxs.append(fth.vel_base(e.d)[0][0])
            for k, v in termos.items():
                termos_ep[i][k] += v
            if causa:
                fim[i] = 1.0
                quedas += causa == "queda"
                fins.append((ret_ep[i], len_ep[i], termos_ep[i]))
                ret_ep[i], len_ep[i] = 0.0, 0
                termos_ep[i] = dict.fromkeys(cfg_r, 0.0)
                o = e.reset()
            prox.append(o)
        B["obs"].append(obs); B["acao"].append(acao); B["logp"].append(dist.log_prob(acao).sum(-1))
        B["val"].append(val); B["rec"].append(torch.as_tensor(rec, dtype=torch.float32))
        B["fim"].append(torch.as_tensor(fim, dtype=torch.float32)); B["mu"].append(dist.mean); B["std"].append(dist.stddev)
        obs = torch.as_tensor(np.stack(prox))
    with torch.no_grad():
        _, val_final = rede.dist(obs)
    S = {k: torch.stack(v) for k, v in B.items()}
    adv, ret = ppo.vantagens(S["rec"], S["val"], S["fim"], val_final)
    plano = lambda x: x.reshape(T * a.envs, *x.shape[2:])  # noqa: E731
    info = ppo.atualizar(plano(S["obs"]), plano(S["acao"]), plano(S["logp"]), plano(adv), plano(ret),
                         plano(S["val"]), plano(S["mu"]), plano(S["std"]))
    if fins:
        rm, em = np.mean([f[0] for f in fins]), np.mean([f[1] for f in fins])
        for k in cfg_r:
            soma_termos[k] = np.mean([f[2][k] for f in fins])
    else:  # nenhum episódio acabou nesta iteraçêo: usa os que estêo em andamento
        rm, em = float(np.mean(ret_ep)), float(np.mean(len_ep))
        for k in cfg_r:
            soma_termos[k] = np.mean([t[k] for t in termos_ep])
    linha = {"iteracao": it, "recompensa_media": round(float(rm), 4), "episodio_medio": round(float(em), 1),
             "episodios": len(fins), "quedas": quedas, "vx_media": round(float(np.mean(vxs)), 4),
             "std_acao": round(float(rede.std.detach().mean()), 4), "lr": f"{ppo.lr:.2e}", "kl": round(info["kl"], 5),
             "segundos": round(time.time() - t0, 2), **{k: round(float(v), 4) for k, v in soma_termos.items()}}
    w.writerow(linha)
    arq_csv.flush()
    print(f"it {it:>3}/{a.iters} · recompensa média {rm:6.2f} · episódio médio {em:5.1f} passos · quedas {quedas:>2}")
arq_csv.close()
ck = saida / f"modelo_{a.iters}.pt"
torch.save({"rede": rede.state_dict(), "n_obs": n_obs, "n_acao": n_acao, "iteracao": a.iters,
            "recompensa": cfg_r, "ppo": PADRAO}, ck)
print(f"checkpoint salvo: {ck}")
print(f"log: {saida / 'progresso.csv'} ({a.iters} linhas) · tempo total: {time.time() - t_ini:.0f} s")
