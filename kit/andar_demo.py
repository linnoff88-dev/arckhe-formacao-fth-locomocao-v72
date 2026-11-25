"""andar_demo.py — o G1 do Menagerie andando com uma política de RL pronta (aula 1).

Uso: python3 kit/andar_demo.py [--vx 0.7] [--s 10] [--sem-politica] [--gravar arquivo.npz]
A política é a de velocidade do unitree_rl_lab (ONNX, treinada no Isaac Lab); aqui ela roda no MuJoCo.
--sem-politica mantém só o PD na pose padrêo (o robô fica de pé, mas nêo anda).
--gravar salva o estado de cada passo (usado pela aula 4).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fth  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--vx", type=float, default=0.7, help="velocidade pedida para a frente (m/s)")
ap.add_argument("--s", type=float, default=10.0, help="segundos de simulaçêo")
ap.add_argument("--sem-politica", action="store_true")
ap.add_argument("--gravar", default=None)
a = ap.parse_args()

pol = None if a.sem_politica else fth.PoliticaPronta()
env = fth.Ambiente(historico=pol.historico if pol else 1, comando=(a.vx, 0.0, 0.0), episodio_s=a.s)
obs = env.reset()
print(f"política: {pol.nome + ' · 480 → 29 · 50 Hz' if pol else 'nenhuma (só PD na pose padrêo)'}")
print(f"comando: vx = {a.vx:.1f} m/s por {a.s:.0f} s · simulaçêo a {1 / env.m.opt.timestep:.0f} Hz, política a {1 / fth.DT_POLITICA:.0f} Hz")
gravado = {k: [] for k in ("cmd", "v", "w", "g_tronco", "q", "a", "a_ant", "f_pes", "v_pes")}
caiu = None
while True:
    acao = pol(obs) if pol else np.zeros(env.n_acao)
    obs, r, causa, _ = env.step(acao)
    if a.gravar:
        e = env.estado()
        for k in gravado:
            gravado[k].append(np.array(e[k], float))
    if causa:
        if causa == "queda":
            caiu = env.d.time
        break
x, y, z = env.d.qpos[0:3]
print(f"andou: {x:.1f} m para a frente · desvio lateral: {abs(y):.1f} m · velocidade média: {x / env.d.time:.2f} m/s")
print(f"altura final da pelve: {z:.2f} m")
print("caiu: nêo" if caiu is None else f"caiu: sim, em t = {caiu:.1f} s")
if a.gravar:
    np.savez(a.gravar, q0=env.q0, lim=env.lim, **{k: np.array(v) for k, v in gravado.items()})
    print(f"gravado: {a.gravar} ({len(gravado['q'])} passos)")
