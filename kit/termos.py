"""termos.py — quanto cada termo da recompensa pagou num rollout gravado (aula 4).

Uso: python3 kit/termos.py rollout.npz [--recompensa kit/recompensa.yaml]
O rollout vem de: python3 kit/andar_demo.py --gravar rollout.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fth  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("rollout")
ap.add_argument("--recompensa", default=None)
ap.add_argument("--comando-vx", type=float, default=None, help="recalcula como se o comando fosse este vx (m/s)")
a = ap.parse_args()
if not Path(a.rollout).exists():
    fth.sair(f"não achei {a.rollout}. Grave antes com: python3 kit/andar_demo.py --gravar {a.rollout}")
z = np.load(a.rollout)
cfg = fth.ler_recompensa(a.recompensa)
n = len(z["q"])
soma = {k: 0.0 for k in cfg}
cru = {k: 0.0 for k in cfg}
for i in range(n):
    e = {k: z[k][i] for k in ("cmd", "v", "w", "g_tronco", "q", "a", "a_ant", "f_pes", "v_pes")}
    e["q0"], e["lim"] = z["q0"], z["lim"]
    if a.comando_vx is not None:
        e["cmd"] = np.array([a.comando_vx, 0.0, 0.0])
    for k, t in cfg.items():
        v = fth.TERMOS[k](e, t)
        cru[k] += v
        soma[k] += t["weight"] * v * fth.DT_POLITICA
print(f"rollout: {Path(a.rollout).name} · {n} passos ({n * fth.DT_POLITICA:.0f} s) · comando vx = {z['cmd'][0][0] if a.comando_vx is None else a.comando_vx:.1f} m/s"
      + ("" if a.comando_vx is None else " (trocado na conta)"))
print(f"{'termo':<24}{'weight':>8}{'valor médio':>13}{'total':>9}")
for k in cfg:
    print(f"{k:<24}{cfg[k]['weight']:>8g}{cru[k] / n:>13.3f}{soma[k]:>9.2f}")
print(f"{'RECOMPENSA DO EPISÓDIO':<45}{sum(soma.values()):>9.2f}")
