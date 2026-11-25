# Treinos de exemplo (logs reais, 19/10/2026)

Gerados nesta máquina (aarch64, CPU, carga alta) com o `kit/treinar.py`, 32 ambientes, 150 iterações, semente 0:

- `cai-cedo/`: `python3 kit/treinar.py --iters 150 --envs 32` com `kit/recompensa.yaml` (497 s).
- `com-vivo/`: o mesmo com `--recompensa kit/recompensa-viva.yaml` (acrescenta `is_alive` com weight 5.0) (585 s).

Só o `progresso.csv` foi guardado; os checkpoints ficaram fora do repositório.
Nenhum dos dois anda: 150 iterações em CPU sêo uma fraçêo do que o mjlab usa (`max_iterations=30_000`, 4.096 ambientes na GPU).
