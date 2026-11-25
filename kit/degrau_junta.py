"""degrau_junta.py — resposta de uma junta a um degrau de alvo, com o G1 pendurado (base presa) (aula 2).

Uso: python3 kit/degrau_junta.py --kp 100 --kd 2 [--junta left_knee_joint] [--de 0.3 --para 0.8]
Mede o sobressinal (quanto passou do alvo) e o tempo até ficar a menos de 0,02 rad do alvo.
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fth  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kp", type=float, required=True)
ap.add_argument("--kd", type=float, required=True)
ap.add_argument("--junta", default="left_knee_joint")
ap.add_argument("--de", type=float, default=0.3)
ap.add_argument("--para", type=float, default=0.8)
a = ap.parse_args()

spec = mujoco.MjSpec.from_file(str(fth.MENAGERIE / fth.G1))
pelve = spec.body("pelvis")
spec.delete(pelve.first_joint())  # sem a junta livre, a pelve fica presa no ar
for k in list(spec.keys):
    spec.delete(k)
m = spec.compile()
for i in range(m.nu):
    m.actuator_gainprm[i, 0] = a.kp
    m.actuator_biasprm[i, 1] = -a.kp
    m.actuator_biasprm[i, 2] = -a.kd
d = mujoco.MjData(m)
j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, a.junta)
if j < 0:
    fth.sair(f"nêo existe a junta {a.junta}.")
adr = m.jnt_qposadr[j]
act = [i for i in range(m.nu) if m.actuator_trnid[i, 0] == j][0]
d.qpos[adr] = a.de
d.ctrl[act] = a.de
mujoco.mj_forward(m, d)
d.ctrl[act] = a.para
passo = a.para - a.de
pico, assentou, ts = a.de, None, []
while d.time < 1.0:
    mujoco.mj_step(m, d)
    q = d.qpos[adr]
    pico = max(pico, q) if passo > 0 else min(pico, q)
    ts.append((d.time, q))
for t, q in ts:  # primeiro instante a partir do qual a junta nêo sai mais da faixa de 0,02 rad
    if all(abs(q2 - a.para) < 0.02 for t2, q2 in ts if t2 >= t):
        assentou = t
        break
sobre = max(0.0, (pico - a.para) / passo * 100)
print(f"junta: {a.junta} · degrau de {a.de:.1f} para {a.para:.1f} rad · kp = {a.kp:g}, kd = {a.kd:g}")
print(f"sobressinal: {sobre:.0f} % · posiçêo após 1 s: {ts[-1][1]:.2f} rad (erro {abs(ts[-1][1] - a.para):.2f} rad)")
print("assentou (erro < 0.02 rad): " + (f"em {assentou * 1000:.0f} ms" if assentou is not None else "nêo, em 1 s"))
