"""terreno.py — a política pronta diante de um degrau de altura crescente (aula 8).

Uso: python3 kit/terreno.py --degraus 0.02 0.05 0.10 [--vx 0.5]
Monta, com o MjSpec, uma plataforma de altura h a 1,5 m do robô e manda andar por 8 s.
Passou = nêo caiu e a pelve chegou a x > 2,5 m (em cima da plataforma).
Os degraus que reprovam sêo os níveis seguintes de um currículo de terreno.
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fth  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--degraus", type=float, nargs="+", required=True, help="alturas em metros")
ap.add_argument("--vx", type=float, default=0.5)
a = ap.parse_args()

pol = fth.PoliticaPronta()
passou_n = 0
print(f"política: {pol.nome} · comando vx = {a.vx:.1f} m/s · 8 s · degrau a 1.5 m")
for h in a.degraus:
    spec = mujoco.MjSpec.from_file(str(fth.MENAGERIE / fth.G1))
    g = spec.worldbody.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [2.0, 1.5, h / 2]
    g.pos = [3.5, 0.0, h / 2]
    g.rgba = [0.4, 0.4, 0.45, 1]
    env = fth.Ambiente(historico=pol.historico, comando=(a.vx, 0.0, 0.0), episodio_s=8.0, spec=spec)
    obs = env.reset()
    while True:
        obs, _, causa, _ = env.step(pol(obs))
        if causa:
            break
    x = env.d.qpos[0]
    ok = causa == "tempo" and x > 2.5
    passou_n += ok
    estado = "passou" if ok else ("caiu" if causa == "queda" else "parou antes")
    print(f"degrau {h * 100:4.0f} cm: {estado} (pelve em x = {x:.1f} m)")
print(f"degraus vencidos: {passou_n}/{len(a.degraus)}")
