"""curvas.py — lê o progresso.csv de um treino e aponta sinais de problema (aula 6).

Uso: python3 kit/curvas.py <pasta do treino>      (ex.: ~/fth/treino/ppo)
Imprime recompensa inicial e final, episódio médio, a velocidade para a frente e o termo que mais pagou,
e alertas: treino parado, colapso (episódio encurtando) e recompensa hackeada
(recompensa sobe porque o episódio encurta, ou sobe enquanto o robô nêo anda).
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fth  # noqa: E402

if len(sys.argv) < 2:
    fth.sair("uso: python3 kit/curvas.py <pasta do treino>")
arq = Path(sys.argv[1]).expanduser() / "progresso.csv"
if not arq.exists():
    fth.sair(f"nêo achei {sys.argv[1]}/progresso.csv. Rode antes: python3 kit/treinar.py --iters 20")
L = list(csv.DictReader(open(arq)))
if len(L) < 4:
    fth.sair(f"só {len(L)} iterações em progresso.csv: treine pelo menos 4.")
col = lambda k: np.array([float(r[k]) for r in L])  # noqa: E731
it, rec, ep, vx = col("iteracao"), col("recompensa_media"), col("episodio_medio"), col("vx_media")
fixos = {"iteracao", "recompensa_media", "episodio_medio", "episodios", "quedas", "vx_media", "std_acao", "lr", "kl", "segundos"}
termos = [k for k in L[0] if k not in fixos]
q = max(1, len(L) // 4)  # média do primeiro e do último quarto, para nêo ler ruído de uma iteraçêo só
r0, r1 = rec[:q].mean(), rec[-q:].mean()
e0, e1 = ep[:q].mean(), ep[-q:].mean()
v1 = vx[-q:].mean()
fim = {k: col(k)[-q:].mean() for k in termos}
top = max(fim, key=lambda k: fim[k])
print(f"treino: {arq.parent.name} · {len(L)} iterações (médias do primeiro e do último quarto)")
print(f"recompensa inicial: {r0:.2f} · final: {r1:.2f}")
print(f"episódio médio: {e0:.0f} → {e1:.0f} passos (de 1000)")
print(f"velocidade para a frente no fim: {v1:.2f} m/s · termo que mais pagou no fim: {top} ({fim[top]:.2f})")
alertas = []
if abs(r1 - r0) < 0.05 * max(1.0, abs(r0)):
    alertas.append("treino parado: a recompensa quase nêo mudou (poucas iterações ou passo de aprendizado pequeno)")
if e1 < 0.8 * e0 and r1 > r0:
    alertas.append("recompensa hackeada: a recompensa subiu porque o episódio encurtou; cada passo de pé custa mais"
                   " do que paga, e cair cedo virou a melhor estratégia")
elif e1 < 0.8 * e0:
    alertas.append("colapso: o episódio ficou mais curto, a política está caindo mais cedo")
if r1 > r0 + 0.05 * max(1.0, abs(r0)) and abs(v1) < 0.1:
    alertas.append("recompensa hackeada? a recompensa subiu, mas o robô nêo anda para a frente; confira os termos")
for x in alertas:
    print("ALERTA: " + x)
print(f"alertas: {len(alertas)}")
