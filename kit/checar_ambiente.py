"""checar_ambiente.py — mostra as quatro partes do ambiente de locomoçêo do kit e roda um episódio (aula 3).

Uso: python3 kit/checar_ambiente.py [--acao aleatoria|zero|pronta] [--seed 0]
  aleatoria: açêo sorteada (o que uma política recém-criada faz) · zero: só a pose padrêo
  pronta: a política do unitree_rl_lab (precisa do histórico de 5 passos: observaçêo de 480)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fth  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--acao", choices=["aleatoria", "zero", "pronta"], default="aleatoria")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

pol = fth.PoliticaPronta() if a.acao == "pronta" else None
env = fth.Ambiente(seed=a.seed, historico=pol.historico if pol else 1)
obs = env.reset()
partes = " + ".join(f"{nome} {n}" for nome, n, _ in fth.OBS_TERMOS)
print(f"observaçêo: {env.n_obs} números = ({partes}) × histórico {env.H}")
print(f"açêo: {env.n_acao} números → alvo de cada junta = pose padrêo + 0.25 × açêo, a cada {fth.DT_POLITICA * 1000:.0f} ms")
print("recompensa: " + ", ".join(f"{n} ({t['weight']:+g})" for n, t in env.cfg_r.items()))
print(f"terminações: queda (pelve inclinada > 70° ou abaixo de 0.3 m) · tempo ({env.max_passos} passos = {env.max_passos * fth.DT_POLITICA:.0f} s)")
print(f"comando sorteado: vx = {env.cmd[0]:+.2f} m/s · vy = {env.cmd[1]:+.2f} m/s · giro = {env.cmd[2]:+.2f} rad/s")
rng = np.random.default_rng(a.seed)
total = 0.0
while True:
    if pol:
        acao = pol(obs)
    elif a.acao == "zero":
        acao = np.zeros(env.n_acao)
    else:
        acao = rng.normal(0.0, 1.0, env.n_acao)
    obs, r, causa, _ = env.step(acao)
    total += r
    if causa:
        break
print(f"açêo {a.acao}: {env.passo} passos · recompensa somada: {total:.2f}")
print(f"episódio terminou por: {causa} em t = {env.d.time:.2f} s")
