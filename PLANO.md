# PLANO — FTH 1 · Ensinar a andar (formacao-fth-locomocao-v72)

Curso 1 de 5 da **FTH — Formação Tecnológica em Humanoides** (v7.2, PT/EN/ES). Público: quem programa em Python, já viu ROS 2 e quer entender como um humanoide aprende a andar, sem ter um humanoide nem GPU. Toda prática roda em CPU, em minutos, com o G1 29 GDL do MuJoCo Menagerie.
Fronteira: PID básico é do curso `robot` (aqui só PD de junta); RL depois da imitação e world models são do `vla-v72` aula 7; sim2sim, gap e escada até o chão são do curso 2 (`formacao-fth-sim2real-v72`), que usa a mesma política pública do unitree_rl_lab 4960b84. Curso anterior: `formacao-fth-anatomia-v72`. Próximo: `formacao-fth-sim2real-v72`.

## Versões testadas nesta máquina (19/10/2026, aarch64 / NVIDIA GB10, Ubuntu, Python 3.12.3)
| item | versão | como |
|---|---|---|
| MuJoCo | **3.16.0** (wheel aarch64 do PyPI) | `pip install mujoco==3.16.0` |
| onnxruntime | **1.31.0** (wheel aarch64) | `pip install onnxruntime==1.31.0` |
| torch | **2.14.1+cpu** | `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu` |
| numpy / pyyaml | atuais do pip | idem |
| MuJoCo Menagerie | commit **62ab00f** (15/10/2026) | `git clone --filter=blob:none --sparse` + `sparse-checkout add unitree_g1` |
| Política pronta | **unitree_rl_lab 4960b84** (19/11/2025), `deploy/robots/g1_29dof/config/policy/velocity/v0` (policy.onnx 480 → 29, deploy.yaml), Apache-2.0 | `sparse-checkout set …/velocity` |
| mjlab | **8bb2bc9** (18/10/2026) — só lido (nomes, pesos e PPO copiados para o kit) | clone no scratchpad |
| unitree_rl_lab (código de treino) | **4960b84** — só lido (recompensas, push, currículo) | clone no scratchpad |

PPO do kit: `kit/ppo.py` (~120 linhas, ator-crítico 512-256-128 ELU, GAE, recorte, lr adaptativo por KL) com os padrões do `rl_cfg.py` do G1 no mjlab (rsl-rl-lib 5.5.1). `torch.set_num_threads(1)`.

Armadilhas achadas no teste (viraram conteúdo):
- Com os ganhos do `deploy.yaml` (tornozelo kp 40) e sem política, o G1 parado cai; com o mesmo kp em todas as juntas, só fica de pé a partir de ~200 (175 cai, 200 segura). Aula 2.
- Com a recompensa do mjlab e peso 0,5 no rastreio linear, o rollout parado (40,22) vale mais que o andando (39,65); o empate é perto de 0,57 (0,55 perde, 0,6 ganha). Aula 4.
- 150 iterações em CPU (32 ambientes) com a recompensa do mjlab dão recompensa hackeada (episódio 55 → 15 passos com a recompensa subindo); com `is_alive` 5,0 o episódio cresce, mas o robô ainda não anda (0,16 m/s). Logs reais em `kit/exemplos/`. Aula 6.
- `cp -n` imprime aviso no coreutils novo: o bloco da aula 8 usa `[ -f … ] || cp`.

**Mostrado, não testado aqui:** mjlab (`uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096`), unitree_rl_lab / Isaac Lab (`python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity`), MuJoCo Playground (`G1JoystickFlatTerrain`), Isaac Lab 3.0 EA / Isaac Sim 6.1 — todos pedem GPU NVIDIA e ambiente próprio; a GPU desta máquina é dividida com outras sessões, e o curso não os põe como pré-requisito.

## Pasta do aluno
`~/fth/` (variável `FTH`) do curso 0: `.venv/`, `mujoco_menagerie/` e, deste curso, `locomocao/` (clone deste repo), `unitree_rl_lab/` (só a pasta da política) e `treino/` (rollouts, `ppo/`, plano). Os blocos começam com `F=${FTH:-$HOME/fth}`; `kit/capturar.sh` roda o mesmo texto com `FTH=<scratchpad>` e troca o caminho por `~/fth`.

## Aulas (objetivo · diagrama · prática "pronto quando", já testada)
| # | slug | objetivo (≤ 160) | diagrama | pronto quando (comando → saída) |
|---|---|---|---|---|
| 1 | por-que-rl | rodar a política de marcha pronta do G1 no MuJoCo, medir o que ela faz e comparar com o robô sem política | modelo → MPC → limite · sim massiva → política → PD | `python3 kit/andar_demo.py --vx 0.7 --s 10; echo "saída: $?"` → `caiu: não` + `saída: 0` (6.6 m); `--sem-politica` → `caiu: sim, em t = 1.4 s` |
| 2 | pd | medir o efeito de kp e kd numa junta e deixar o G1 de pé só com o PD, entendendo por que os ganhos da política não bastam sem ela | alvo → erro → freio → limite → junta | `python3 kit/ficar_em_pe.py --kp 300 --kd 5; echo "saída: $?"` → `de pé (…): sim` + `saída: 0`; `--ganhos rl_lab` → `não` + `saída: 1` |
| 3 | ambiente | ler as quatro partes de um ambiente de locomoção e rodar um episódio com ação aleatória, zero e pronta | comando, observação, ação, recompensa, término | `python3 kit/checar_ambiente.py --acao aleatoria; echo "saída: $?"` → `observação: 96 números` + `episódio terminou por: queda em t = 1.34 s` |
| 4 | recompensa | medir quanto cada termo da recompensa paga num rollout gravado e prever o efeito de mudar um peso antes de treinar | rastrear, postura, limites, suavidade, contato | `python3 kit/termos.py $F/treino/andando.npz; echo "saída: $?"` → `RECOMPENSA DO EPISÓDIO … 53.60` + `saída: 0`; parado `42.32`; peso 0,5 → `39.65` × `40.22` |
| 5 | ppo | rodar um treino curto de PPO em CPU que grava checkpoint e log, e ler os valores de um treino de verdade no mjlab | coletar → vantagem → atualizar → passo (KL) → guardar | `timeout 900 python3 kit/treinar.py --iters 20; echo "saída: $?"` → `checkpoint salvo: ~/fth/treino/ppo/modelo_20.pt` + `saída: 0` + `arquivo existe: modelo_20.pt` + `21` |
| 6 | curvas | ler o progresso.csv de um treino, separar subida de recompensa de aprendizado real e reconhecer colapso e recompensa hackeada | recompensa, episódio, velocidade, termos + simfila (treinar → ler → ajustar → treinar → avaliar) | `python3 kit/curvas.py kit/exemplos/cai-cedo` → `ALERTA: recompensa hackeada` + `alertas: 1`; com-vivo → `alertas: 0` |
| 7 | avaliar | avaliar uma política em 20 episódios com sementes fixas, medindo erro de velocidade e sobrevivência a empurrões de força crescente | sementes, rastreio, curva, empurrão | `python3 kit/avaliar.py --episodios 20 --empurrao 1.0; echo "saída: $?"` → `sobreviveu: 20/20` + `saída: 0`; 1.5 → `16/20`; 2.0 → `9/20` |
| 8 | terreno | medir o degrau máximo da política pronta, ler as regras de currículo e escrever um plano de treino com 5 campos que o conferir_plano.py aprova | plano → degrau → rampa → irregular + mapa + gerador | `python3 kit/terreno.py --degraus 0.02 0.03 0.04 0.06 0.10` → `degraus vencidos: 2/5`; `python3 kit/conferir_plano.py $F/treino/plano-treino.md` → `campos ok: 5/5` + `ok` + `saída: 0` |

Saídas de erro comum capturadas: `e1-sem-bancada` (1), `e2-sem-kd` (2), `e4-sem-rollout` (4), `e5-sem-venv` (5). Na 3 e na 6 o "erro comum" é de leitura (ação zero com recompensa positiva; recompensa subindo com o episódio encurtando).

Exercícios conferidos à mão fora da captura (19/10/2026): aula 2 E3 (kp 175 cai, 200 segura), aula 3 E3 (sementes 1–3 diferentes, repetição igual), aula 4 E3 (peso 0,55 perde, 0,6 ganha), aula 5 E2 (`--envs 32 --iters 10` → 768 amostras, `modelo_10.pt`), aula 7 E3 (`--giro 0.2` → 20/20, erro 0.04 m/s).

## Simulação de ciclo
`simfila` na aula 6: treinar → ler as curvas (falha proposital: alerta de recompensa hackeada) → ajustar a recompensa → treinar de novo → avaliar.

## Links fixos
- Curso anterior: https://inematds.github.io/formacao-fth-anatomia-v72/ (publicado).
- Curso seguinte (aula 8, `links`): https://inematds.github.io/formacao-fth-sim2real-v72/ (publicado).
- Donos externos: robot (PID), vla-v72 aula 6 (avaliação em sim) e aula 7 (RL depois da imitação).

## Determinismo
Duas capturas completas seguidas (`kit/capturar.sh`, 26 blocos, 19/10/2026) → `diff -r` com uma única diferença de conteúdo: `tempo total: 8 s` × `9 s` no treino da aula 5 (já declarado como variável na aula; o "pronto quando" dela é estrutural). A outra diferença foi o aviso do `cp -n`, corrigido no bloco entre as duas capturas. O PPO em CPU com semente fixa e um núcleo (`torch.set_num_threads(1)`) repetiu as 20 linhas `it` idênticas.
