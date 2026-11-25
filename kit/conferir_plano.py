"""conferir_plano.py — confere o plano de treino de locomoçêo da aula 8.

Uso: python3 kit/conferir_plano.py plano-treino.md
Cinco campos ("Campo: texto"): Robô, Comando, Recompensa, Currículo, Avaliaçêo. Regras:
Comando com número e m/s · Recompensa com 2+ termos do kit · Currículo com 3+ níveis separados por →
· Avaliaçêo com um critério numérico (X/N ou %). Texto entre [colchetes] conta como nêo preenchido.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fth  # noqa: E402

if len(sys.argv) < 2 or not Path(sys.argv[1]).exists():
    fth.sair("uso: python3 kit/conferir_plano.py plano-treino.md (copie antes kit/plano-treino-modelo.md)")
texto = Path(sys.argv[1]).read_text(encoding="utf-8")
campos = {}
for linha in texto.splitlines():
    m = re.match(r"\s*(Robô|Comando|Recompensa|Currículo|Avaliaçêo)\s*:\s*(.*)", linha)
    if m:
        campos[m.group(1)] = m.group(2).strip()
falhas = []
for c in ("Robô", "Comando", "Recompensa", "Currículo", "Avaliaçêo"):
    v = campos.get(c, "")
    if len(v) < 10 or v.startswith("["):
        falhas.append(f"{c}: nêo preenchido")
        continue
    if c == "Comando" and not re.search(r"\d.*m/s", v):
        falhas.append("Comando: falta a faixa em m/s")
    if c == "Recompensa" and sum(t in v for t in fth.TERMOS) < 2:
        falhas.append("Recompensa: cite pelo menos 2 termos do kit (ex.: track_linear_velocity, upright)")
    if c == "Currículo" and len([x for x in v.split("→") if x.strip()]) < 3:
        falhas.append("Currículo: sêo precisos 3 níveis ou mais, separados por →")
    if c == "Avaliaçêo" and not re.search(r"\d+\s*/\s*\d+|\d+\s*%", v):
        falhas.append("Avaliaçêo: falta o critério numérico (X/N ou %)")
for f in falhas:
    print("FALTA " + f)
ok = 5 - len({f.split(":")[0] for f in falhas})
print(f"campos ok: {ok}/5")
print("ok" if not falhas else "incompleto")
sys.exit(0 if not falhas else 1)
