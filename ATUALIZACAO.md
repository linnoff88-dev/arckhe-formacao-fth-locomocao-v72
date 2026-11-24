# ATUALIZACAO — FTH 1 · Ensinar a andar (formacao-fth-locomocao-v72)

Passo "atualizar, não copiar" do BRIEF (19/10/2026). Fontes antigas lidas: FTH `niveis/nivel-2/modulo-2.md` (2.2 Sensores e controle/PID), `nivel-2/modulo-3.md` (2.3 RL introdutório), `nivel-3/modulo-1.md` (3.1 PPO e SAC), `robotica-real2sim/docs/HUMANOIDES.md` (14/10/2026). Cada afirmação foi conferida contra a PESQUISA-ESTADO-DA-ARTE.md (FTH, 19/10/2026), a PESQUISA-ECOSSISTEMA.md / PESQUISA-MODELOS.md (VLA v7.2, 19/10/2026) e contra o **código** dos repositórios clonados nesta data (quando código e prosa divergem, vale o código).

Repositórios lidos (clone no scratchpad, 19/10/2026): `unitreerobotics/unitree_rl_lab` @ **4960b84** (19/11/2025, Apache-2.0 — arquivo LICENCE) · `mujocolab/mjlab` @ **8bb2bc9** (18/10/2026) · `google-deepmind/mujoco_menagerie` @ **62ab00f** (15/10/2026, o mesmo do curso 0).

## O que foi atualizado

| fonte antiga | o que dizia | o que o curso diz agora | fonte (URL/data) |
|---|---|---|---|
| FTH 2.3 (§2.2–2.4, 7 ocorrências) | `model.fit(total_timesteps=10000)` com stable-baselines3 | **Erro do curso antigo**: o método do SB3 é `learn` (a própria FTH 3.1 usa `model.learn`). O curso novo não usa SB3: a prática roda um PPO de ~120 linhas (`kit/ppo.py`) com os nomes e valores do mjlab | FTH 3.1 linha 358 (`model.learn(`); mjlab `rl_cfg.py` @ 8bb2bc9 |
| FTH 2.3 / 3.1 | RL de humanoide = um ambiente do Gymnasium (`CartPole-v1`, `Humanoid-v4`, `HumanoidStandup-v4`) com SB3, `SubprocVecEnv`, 10 M passos | Humanoide real (Unitree G1 29 GDL do Menagerie) num ambiente "manager-based" (observações, ações, recompensas e terminações com nome), a mesma estrutura do Isaac Lab e do mjlab; em escala, PPO com milhares de ambientes na GPU (mostrado) | PESQUISA §3.1, §3.4; mjlab `velocity_env_cfg.py` @ 8bb2bc9 |
| FTH 3.1 | Hiperparâmetros PPO do SB3: lr 3e-4, clip 0.2, gamma 0.99, gae 0.95, max_grad_norm 0.5 | Os do mjlab para o G1: lr 1e-3 com ajuste adaptativo (desired_kl 0.01), clip 0.2, gamma 0.99, lam 0.95, max_grad_norm 1.0, 5 épocas, 4 mini-lotes, 24 passos por ambiente, rede 512-256-128 ELU, obs normalizadas, `max_iterations=30_000`; biblioteca `rsl-rl-lib==5.5.1` | mjlab `src/mjlab/tasks/velocity/config/g1/rl_cfg.py` e `pyproject.toml` @ 8bb2bc9 (18/10/2026) |
| FTH 3.1 | SAC como alternativa principal ao PPO para humanoide | Fora: todos os repositórios de locomoção de humanoide da pesquisa usam PPO em sim massiva | PESQUISA §3.1 ("PPO em milhares de ambientes paralelos") |
| FTH 3.1 | Ver o treino no TensorBoard | `progresso.csv` (uma linha por iteração, com cada termo da recompensa) + `kit/curvas.py` com alertas de treino parado, colapso e recompensa hackeada | kit do curso (testado 19/10/2026) |
| FTH 2.2 (§3) | PID genérico com termo integral e ajuste Ziegler-Nichols, aplicado a uma junta via ROS 2 | PD por junta (kp/kd), como nos robôs de RL: o deploy do unitree_rl_lab manda q, dq, kp, kd e tau por motor (sem termo I); `stiffness`/`damping` vêm por junta, na ordem do SDK. PID básico tem dono: curso `robot` | unitree_rl_lab `deploy/include/FSM/State_RLBase.h` e `deploy.yaml` @ 4960b84 |
| FTH 2.3 (§1.2) | Recompensa de exemplo: +1 andou, −10 bateu, 0 parado | Termos reais de marcha: mjlab (`track_linear_velocity` 2.0, `track_angular_velocity` 2.0, `upright` 1.0, `pose` 1.0, `dof_pos_limits` −1.0, `action_rate_l2` −0.1, `foot_slip` −0.1…) e unitree_rl_lab (`alive` 0.15, `base_height` −10, `flat_orientation_l2` −5, `gait` 0.5…), lidos dos arquivos | mjlab `velocity_env_cfg.py` @ 8bb2bc9; unitree_rl_lab `tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py` @ 4960b84 |
| FTH 2.3 / 3.1 (rodapé) | Links de Discord (`discord.gg/fth-brasil`, "RL Discord") e certificado | Removidos; conteúdo aberto do ARCKHÉ AI | AUDITORIA-LOCAL (Geral) |
| robotica-real2sim `docs/HUMANOIDES.md` §3 | Isaac Lab treina o G1 com 4.096 ambientes; ~82 mil passos/s na v3.0.0-beta numa RTX 4090 (250 mil/s na 2.x); "1.620×" é conta derivada | Entra como número relatado, com a ressalva do próprio documento: "(relatado) segundo a tabela de benchmarks do Isaac Lab, varia com a versão". O 4.096 está no código: `num_envs=4096` no unitree_rl_lab e `--env.scene.num-envs 4096` no mjlab | HUMANOIDES.md (14/10/2026); unitree_rl_lab `velocity_env_cfg.py` @ 4960b84; README do mjlab |
| CURRICULO (paper legged_gym) | legged_gym / Isaac Gym como base do RL de pernas | Citado como **legado** (Rudin et al., CoRL 2021, arXiv 2109.11978): unitree_rl_lab, BeyondMimic e Berkeley Humanoid Lite usam Isaac Lab; o mjlab traz a mesma API sobre MuJoCo Warp | PESQUISA §1.3, §3.2, §3.4 |
| — (novo) | — | Política pronta para a demo: velocidade do G1 29 GDL do unitree_rl_lab (`velocity/v0`, ONNX, entrada 480 = 6 termos × histórico 5, saída 29, 50 Hz). Roda no G1 do Menagerie com o PD do `deploy.yaml`: 20/20 episódios de pé sem empurrão (medido aqui) | unitree_rl_lab @ 4960b84; medição ARCKHÉ 19/10/2026 (`kit/avaliar.py`) |
| — (novo) | — | Empurrão e currículo do mjlab: empurrão a cada 1–3 s com ±0,5 m/s em x e y; currículo de terreno sobe de nível quem andou > 80 % da distância pedida e desce quem andou < 50 % | mjlab `velocity_env_cfg.py` (events.push_robot) e `mdp/curriculums.py` @ 8bb2bc9 |
| — (novo) | — | O que se usa em escala (mostrado, GPU NVIDIA): `uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096`; MuJoCo Playground `G1JoystickFlatTerrain`; unitree_rl_lab `Unitree-G1-29dof-Velocity`; Isaac Lab 3.0 EA (16/09/2026) exige Isaac Sim 6.1 e Python 3.12 + `uv` | PESQUISA §3.4; VLA PESQUISA-ECOSSISTEMA §2 |
| — (novo) | — | MPC × RL: Helix 02 (Figure, 27/01/2026) troca 109.504 linhas de C++ por um controlador aprendido a 1 kHz com 10 M parâmetros | VLA PESQUISA-MODELOS §4 |
| — (versões) | MuJoCo sem versão | MuJoCo 3.16.0 (14/10/2026), testado aqui; Menagerie fixado em 62ab00f | VLA PESQUISA-ECOSSISTEMA §2 |

## O que ficou de fora (sem fonte ou fora do escopo)

- **Comparação quantitativa MPC × RL** independente em 2026: a PESQUISA marca [incerto]. O curso só diz o que a Figure relatou do Helix 02 e que o MPC segue como conceito (ZMP, centroidal).
- **"A maioria dos repos migrou do Isaac Gym"**: síntese [incerto] na PESQUISA; o curso cita só os repositórios conferidos que usam Isaac Lab/mjlab.
- **Horas de GPU e tempo de treino do mjlab/unitree_rl_lab**: nenhum número de fonte primária; o curso não promete "uma tarde".
- **SAC, DQN, Q-learning, CartPole, OpenCV e visão** (FTH 2.2, 2.3, 3.1): fora do foco de locomoção.
- **"1.620×"** do HUMANOIDES.md: conta derivada; o curso fica com o número relatado e a ressalva.
- **Rodar mjlab, Playground ou Isaac Lab nesta máquina**: não tentado (GPU compartilhada com outras sessões, máquina com carga 50+); ficam como "mostrado", com o comando da documentação.
