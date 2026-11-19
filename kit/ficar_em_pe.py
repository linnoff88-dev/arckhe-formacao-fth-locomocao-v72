"""ficar_em_pe.py — só a camada de baixo: PD em cada junta segurando a pose padrão do G1 (aula 2).

Uso: python3 kit/ficar_em_pe.py --kp 100 --kd 2      (o mesmo kp/kd nas 29 juntas)
     python3 kit/ficar_em_pe.py --ganhos rl_lab      (kp/kd por junta do deploy.yaml do unitree_rl_lab)
Torque de cada junta = kp·(alvo − q) − kd·q̇, limitado pelo actuatorfrcrange do MJCF.
Sem política: nada corrige o equilíbrio, só a rigidez das juntas.
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fth  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kp", type=float)
ap.add_argument("--kd", type=float)
ap.add_argument("--ganhos", choices=["rl_lab"])
ap.add_argument("--s", type=float, default=5.0)
a = ap.parse_args()
if a.ganhos is None and (a.kp is None or a.kd is None):
    fth.sair("passe --kp e --kd (ex.: --kp 100 --kd 2) ou --ganhos rl_lab.")

m = fth.carregar(ganhos=a.ganhos, kp=a.kp, kd=a.kd)
d = mujoco.MjData(m)
q0, _ = fth.pose_padrao()
mujoco.mj_resetDataKeyframe(m, d, 0)
d.qpos[7:] = q0
d.ctrl[:] = q0
mujoco.mj_forward(m, d)
pelve = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
h0 = d.qpos[2]
lim = np.abs(m.jnt_actfrcrange[1:, 1])
satur = np.zeros(m.nu, bool)
tmax = 0.0
while d.time < a.s - 1e-9:
    mujoco.mj_step(m, d)
    tau = d.qfrc_actuator[6:]
    satur |= np.abs(tau) >= lim - 1e-6
    tmax = max(tmax, float(np.max(np.abs(tau))))
h = d.qpos[2]
inc = fth.inclinacao_graus(d, pelve)
dentro = 0.70 <= h <= 0.85 and inc < 20
ganho = "kp/kd por junta do unitree_rl_lab" if a.ganhos else f"kp = {a.kp:g} · kd = {a.kd:g} em todas as juntas"
print(f"ganhos: {ganho}")
print(f"altura da pelve: {h0:.2f} m no início → {h:.2f} m após {a.s:.0f} s · inclinação: {inc:.0f}°")
print(f"maior torque pedido: {tmax:.0f} N·m · juntas que bateram no limite de torque: {int(satur.sum())}")
print(f"de pé (pelve entre 0.70 e 0.85 m, inclinação < 20°): {'sim' if dentro else 'não'}")
sys.exit(0 if dentro else 1)
