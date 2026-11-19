"""avaliar.py — avaliação de uma política em vários episódios: rastreio de velocidade, curva e empurrão (aula 7).

Uso: python3 kit/avaliar.py [--politica pronta|<checkpoint.pt>] [--episodios 20] [--empurrao 1.0] [--giro 0]
Episódio i: semente i, comando vx sorteado entre 0.3 e 1.0 m/s, 10 s; com --empurrao, aos 4 s a base
leva um empurrão horizontal de N m/s (direção sorteada). Mesmas sementes = mesmos episódios.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fth  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--politica", default="pronta")
ap.add_argument("--episodios", type=int, default=20)
ap.add_argument("--empurrao", type=float, nargs="?", const=1.0, default=0.0, help="m/s somados à base aos 4 s")
ap.add_argument("--giro", type=float, default=0.0, help="comando de giro (rad/s)")
ap.add_argument("--s", type=float, default=10.0)
a = ap.parse_args()

pol = fth.politica(a.politica)
vivos, erros, quedas_t = 0, [], []
for ep in range(a.episodios):
    rng = np.random.default_rng(ep)
    vx = rng.uniform(0.3, 1.0)
    ang = rng.uniform(0, 2 * math.pi)
    env = fth.Ambiente(historico=pol.historico, comando=(vx, 0.0, a.giro), episodio_s=a.s)
    obs = env.reset()
    v_med, n, empurrou = 0.0, 0, False
    while True:
        if a.empurrao and not empurrou and env.d.time >= 4.0:
            env.empurrar(a.empurrao * np.array([math.cos(ang), math.sin(ang)]))
            empurrou = True
        obs, _, causa, _ = env.step(pol(obs))
        if env.d.time > 2.0:  # depois de acelerar
            v_med += fth.vel_base(env.d)[0][0]
            n += 1
        if causa:
            break
    if causa == "tempo":
        vivos += 1
        erros.append(abs(v_med / max(n, 1) - vx))
    else:
        quedas_t.append((ep, env.d.time))
print(f"política: {pol.nome} · {a.episodios} episódios de {a.s:.0f} s · giro {a.giro:+.1f} rad/s"
      + (f" · empurrão de {a.empurrao:.1f} m/s aos 4 s" if a.empurrao else " · sem empurrão"))
if erros:
    print(f"erro médio de velocidade (quem não caiu): {np.mean(erros):.2f} m/s")
if quedas_t:
    print("quedas: " + ", ".join(f"ep {e} em {t:.1f} s" for e, t in quedas_t[:6]) + (" …" if len(quedas_t) > 6 else ""))
print(f"sobreviveu: {vivos}/{a.episodios}")
