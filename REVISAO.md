Nota: 9/10

# Revisêo independente — FTH 1 · Ensinar a andar (formacao-fth-locomocao-v72)

Revisor independente (nêo o autor), 19/10/2026. Lido: `conteudo/pt/curso.json` + as 8 aulas, `PLANO.md`, `ATUALIZACAO.md`, `kit/` (scripts, `blocos/`, `saidas/`, `exemplos/`). Fontes de fato: PESQUISA-ESTADO-DA-ARTE (FTH), PESQUISA-ECOSSISTEMA e PESQUISA-MODELOS (VLA), código do mjlab @ 8bb2bc9 e do unitree_rl_lab @ 4960b84, `deploy.yaml` da política velocity/v0. Rodado só o que é leve: `kit/curvas.py` nos dois exemplos e `kit/conferir_plano.py kit/exemplos/plano-treino.md` (as saídas batem com `kit/saidas/a6b-exemplos.txt` e `a8c-plano-exemplo.txt`).

**Leitura geral:** curso sólido, técnico e sem hype. Todo bloco Terminal está em `kit/blocos/`, todo número de aba bate com `kit/saidas/`, os valores copiados do mjlab e do unitree_rl_lab estêo certos quase sempre. A nota fica abaixo de 8 por **três erros de fato contra o código** (um deles do próprio kit), todos de correçêo rápida. **Corrigidos G1, G2 e G3 → nota ≥ 8,5.**

## 1. Quinze fatos sorteados

| # | Afirmaçêo | Aula | Fonte conferida | Resultado |
|---|---|---|---|---|
| 1 | Figure relatou que o Helix 02 (27/01/2026) trocou 109.504 linhas de C++ por controlador aprendido a 1 kHz, 10 M parâmetros; sem comparaçêo independente MPC × RL em 2026 | 1 | PESQUISA-MODELOS l. 69; PESQUISA FTH l. 112 ([incerto] tratado como "nêo achou") | ok (dito como relato da Figure) |
| 2 | Isaac Lab 3.0 em EA (16/09/2026), exige Isaac Sim 6.1 e Python 3.12 novo com `uv` | 5 | PESQUISA-ECOSSISTEMA l. 36; PESQUISA FTH l. 147 | ok |
| 3 | "(relatado) … o G1 com 4.096 ambientes passa de 80 mil passos/s numa RTX 4090 na 3.0 beta" | 5 | Nêo está nas 3 pesquisas. Vem de `robotica-real2sim/docs/HUMANOIDES.md` l. 47 (~82 mil, tarefa `Isaac-Velocity-Rough-G1-v0`, 250 mil na 2.x), que o CURRICULO libera "com a ressalva do próprio doc" | problema menor (M2) |
| 4 | mjlab: API manager-based sobre MuJoCo Warp 3.16, Apache-2.0, `uv`, GPU NVIDIA (macOS só avalia); `uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096` | 3, 5 | PESQUISA FTH l. 145; mjlab `README.md` l. 19, 47, 122; `pyproject.toml` (`mujoco-warp~=3.16.0`) | ok |
| 5 | Playground: `G1JoystickFlatTerrain`, Python 3.12 + `uv`, JAX/MJX "e agora também com MJWarp" | 5 | PESQUISA FTH l. 146 [FATO]; PESQUISA-ECOSSISTEMA l. 33 marca o MJWarp como beta/[2ª] | ok, com ressalva menor (M6) |
| 6 | PPO do G1 no mjlab: lr 1e-3 adaptativo, desired_kl 0,01, clip 0,2, 5 épocas, 4 mini-lotes, gamma 0,99, lam 0,95, max_grad_norm 1,0, 24 passos/env, 512-256-128 ELU, obs normalizadas, `max_iterations=30_000`, grava a cada 50; rsl-rl-lib 5.5.1 | 5 | mjlab `tasks/velocity/config/g1/rl_cfg.py` e `pyproject.toml` l. 50 @ 8bb2bc9; `kit/ppo.py` l. 10–12 idêntico | ok |
| 7 | Recompensa do mjlab: track_linear 2,0 (std 0,5), track_angular 2,0 (std 0,707), upright 1,0 (std 0,447), pose 1,0, dof_pos_limits −1,0, action_rate_l2 −0,1, foot_slip −0,1 | 4 | mjlab `velocity_env_cfg.py` l. 276–360 (std = √0,25, √0,5, √0,2) | ok (mas "os 7 termos do mjlab" é impreciso: M3) |
| 8 | Término `time_out` + `fell_over` com `limit_angle` 70°; episódio 20 s | 3 | mjlab `velocity_env_cfg.py` l. 381–384, 455 | ok |
| 9 | unitree_rl_lab: `alive` 0,15; `base_height` −10 alvo 0,78 m; `flat_orientation_l2` −5; `gait` 0,5, período 0,8 s, defasagem meio período; `num_envs=4096` | 3, 4, 6 | unitree_rl_lab `g1/29dof/velocity_env_cfg.py` l. 251, 295–305, 362 @ 4960b84 | ok |
| 10 | Ganhos do deploy.yaml: quadril 100/2, joelho 150/4, tornozelo 40/2, cintura 200/5, braços 40/10; açêo = pose padrêo + 0,25 × açêo; 480 → 29; 50 Hz; `joint_ids_map` | 1, 2, 3 | `deploy.yaml` (stiffness/damping na ordem do SDK, `scale` 0,25, `step_dt` 0,02, histórico 5 × 96) | ok |
| 11 | Empurrêo: unitree_rl_lab ±0,5 m/s em x e y a cada 5 s; mjlab a cada 1 a 3 s | 7, 8 | unitree l. 151–155; mjlab l. 226–239 (mjlab também empurra z e roll/pitch/yaw) | ok (detalhe menor M8) |
| 12 | Currículo de terreno do mjlab: sobe quem andou > 80 % da distância pedida, desce quem andou < 50 % | 8 | mjlab `mdp/curriculums.py` l. 30, 62–69 | ok (no unitree_rl_lab nêo conferível: M7) |
| 13 | Currículo de comando do unitree_rl_lab: começa em ±0,1 m/s, abre 0,1 m/s quando o rastreio passa de 80 % do peso, até −0,5 a 1,0 m/s | 8 | unitree `velocity_env_cfg.py` l. 170–175, 350–354; `mdp/curriculums.py` l. 11–36 | ok |
| 14 | "No terreno gerado do unitree_rl_lab, metade das faixas é plana (proporçêo 0,5 do sub-terreno flat)" | 8 | unitree `velocity_env_cfg.py` l. 24–37: `sub_terrains` tem **um único** sub-terreno, `flat` (MeshPlaneTerrainCfg) | **GRAVE (G1)** |
| 15 | "A política pronta foi treinada com giro de −0,2 a 0,2 rad/s" | 7 | `deploy.yaml` diz ±0,2; o código de treino começa em `ang_vel_z=(-0.1, 0.1)` e o `CurriculumCfg` só tem `terrain_levels` e `lin_vel_cmd_levels` (o `ang_vel_cmd_levels` existe no módulo mas nêo é usado) | problema menor (M1) |

Fora da amostra, também conferidos e ok: MuJoCo 3.16.0 (14/10/2026, PESQUISA-ECOSSISTEMA l. 31); mjlab 8bb2bc9 = 18/10/2026 e unitree_rl_lab 4960b84 = 19/11/2025 (`git log`); Apache-2.0 no `LICENCE` do unitree_rl_lab; comando `python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity` (README l. 78) e o id da tarefa no `__init__.py`; legged_gym/Isaac Gym só como legado; afirmações sobre cursos irmêos (curso 2 aula 6 mostra a queda em 1 s com a ordem de juntas errada; curso 2 aula 4 mede atraso; curso 0 aulas 4–6) existem nos repos locais.

## 2. Conferência dos "pronto quando" (24 exercícios)

| Aula | E1 | E2 | E3 |
|---|---|---|---|
| 1 | ok (`a1a-preparar.txt`) | ok (`a1b-andar.txt`: `caiu: nêo`, `saída: 0`) | ok (`a1c-sem-politica.txt`: `caiu: sim, em t = 1.4 s`) |
| 2 | ok (`a2a-degrau.txt`: 49 % / 1 %) | ok (`a2b-em-pe.txt`) | manual, declarado no PLANO (175 cai, 200 segura) |
| 3 | ok (`a3a-checar.txt`: 96 números, queda 1.34 s) | ok (`a3b-zero-pronta.txt`) | manual, declarado no PLANO (sementes 1–3) |
| 4 | ok (`a4b-termos.txt`: 53.60, `saída: 0`) | ok (`a4c-parado.txt`: 42.32; 42,32/53,60 = 79 %) | ok `a4d-peso.txt` (39.65 × 40.22); 0,55/0,6 manual, declarado no PLANO |
| 5 | ok (`a5a-treinar.txt`: checkpoint, `saída: 0`, arquivo existe, `21`) | manual, declarado no PLANO (768 amostras, `modelo_10.pt`) | conta (98.304; 2,95 bi; 7.680) certa; ver M9 |
| 6 | ok (`a6a-curvas.txt`) | ok (`a6b-exemplos.txt`, reconfirmado rodando `curvas.py`) | aberto (proposta do aluno), aceitável |
| 7 | ok (`a7a-avaliar.txt`: 20/20, `saída: 0`) | ok (`a7b-mais-forte.txt`: 16/20, 9/20) | manual, declarado no PLANO (`--giro 0.2` → 20/20, 0.04 m/s) |
| 8 | ok (`a8a-degraus.txt`: 2/5, `saída: 0`) | ok (`a8b-plano-vazio.txt`) | estrutura confirmada por `a8c-plano-exemplo.txt` e reconfirmada rodando `conferir_plano.py` |

Todo bloco Terminal das aulas existe, idêntico, em `kit/blocos/` (26 blocos). Os trechos de arquivo (`recompensa.yaml`, `PADRAO` do `ppo.py`, `is_alive` do `recompensa-viva.yaml`, `plano-treino-modelo.md`) sêo idênticos aos arquivos do kit. Todos os números das abas de saída batem com `kit/saidas/*.txt` (6,6 m / 0,66 m/s / 0,77 m; 9,0 m; 49 % / 992 ms / 0,03 rad; 12 N·m e 41 N·m; −5,44, 3,78, 108,61; 18,59 / 16,28 / 2,80; 9 s e 21 linhas; −4,31 → −1,12, 55 → 15, 0,16 m/s, 6,27; 0,05 m/s; 2/5; 0/5). Os números do início (8 aulas, 128 min = 8 × 16, 24 exercícios, 20/20) batem. **Nenhum "pronto quando" diverge da saída capturada.**

## 3. Regras do BRIEF

- Títulos das 8 aulas: iguais ao CURRICULO, letra a letra. ok.
- Isaac Gym/legged_gym: só como legado (aulas 1 e 5, link "legado"). ok.
- "Humble": nenhuma ocorrência. ok.
- Preço de robô: nenhum. ok.
- Placeholder `<…>` fora de código: nenhum; a prosa usa `[colchete]`. ok.
- Assunto de curso irmêo: PID → link do `robot`; polígono de apoio, atuadores, sensores → links do curso 0; sim2sim/gap → link do curso 2 aula 6 e do vla-v72 aula 6; RL depois da imitaçêo → link do vla-v72 aula 7. ok.
- Pilha: MuJoCo + Menagerie em CPU como obrigatória; mjlab, Isaac Lab e Playground só "mostrados". Treino curto verificado por formato. ok.
- Tom: técnico e sem hype. Uma generalizaçêo no lead da aula 1 (M5).
- Quizzes: 3 a 4 perguntas por aula, explicaçêo coerente com a alternativa certa em todas as 26. ok.
- Fonte única de fato: uma exceçêo, o número de passos/s (M2).

## 4. Problemas GRAVES (corrigir antes de publicar)

**G1 — Aula 8, terreno do unitree_rl_lab descrito errado.** `conteudo/pt/aulas/08-terreno.json`, `secoes[0].nos[0].detalhe` ("metade das faixas é plana (proporçêo 0,5 do sub-terreno flat)") e `nos[0].itens[0].t` ("proporçêo 0,5"). No código (4960b84, `g1/29dof/velocity_env_cfg.py` l. 24–37), o `COBBLESTONE_ROAD_CFG` tem um único sub-terreno, `flat` (`MeshPlaneTerrainCfg`). Nêo existe a outra metade: o treino do G1 29 GDL é todo em chêo plano. O nó "Degrau" ("treinada no terreno do unitree_rl_lab") passa a mesma ideia errada de terreno acidentado.
*Correçêo:* "No G1 29 GDL do unitree_rl_lab (4960b84), o gerador de terreno tem um único sub-terreno, `flat`: a política pronta foi treinada no plano. Por isso os degraus do kit já sêo terreno novo para ela." Item: `sub-terreno único: flat`. No nó "Degrau", trocar "treinada no terreno do unitree_rl_lab" por "treinada só no plano".

**G2 — Aula 6, a métrica `recompensa_media` descrita como "média por passo".** `06-curvas.json`, `secoes[0].nos[0].detalhe`: "Média por passo, em cada iteraçêo". No `kit/treinar.py` (l. 89–90), `rm = np.mean([f[0] for f in fins])`, e `f[0]` é `ret_ep`, a **soma do episódio** (média sobre os episódios que terminaram). Confere com o CSV: na 1ª linha do com-vivo, a soma dos termos dá 0,747 = `recompensa_media`. A tese da aula ("a recompensa subiu porque o episódio encurtou") só fecha com soma por episódio. Lendo "média por passo", o aluno nêo entende a lógica.
*Correçêo:* "Recompensa somada por episódio, na média dos episódios que terminaram em cada iteraçêo. Sozinha engana: …". Na aula 5 (aba da saída), dizer uma vez que "recompensa média" é a soma do episódio, na média dos episódios.

**G3 — Aula 6, "+5 por passo de pé" contradiz a regra do próprio kit.** `06-curvas.json`, `secoes[2].blocos[0].conteudo` (comentário do YAML) e `secoes[2].saida.abas[0].html` ("com +5 por passo de pé"). A aula 4 ensina, e `kit/fth.py` l. 186 aplica, que cada termo vale weight × valor × 0,02 s por passo. Entêo `is_alive` com weight 5 paga **+0,1 por passo** (5 por segundo de pé). O CSV confirma: 2,4 em 24 passos e 6,27 em ~63 passos.
*Correçêo:* "+0,1 por passo de pé (weight 5 × 0,02 s), ou seja 5 por segundo: ficar vivo passa a valer mais que a puniçêo". Corrigir também o comentário em `kit/recompensa-viva.yaml` l. 11 e depois rodar `resolver-kit.py --conferir`.

## 5. Problemas menores

- **M1 — Aula 7, giro "treinado".** `07-avaliar.json` `secoes[0].nos[2].detalhe`: "foi treinada com giro de −0,2 a 0,2 rad/s". O `deploy.yaml` diz ±0,2, mas no código de treino o giro fica em ±0,1 (o currículo só abre vx e vy). *Correçêo:* "O `deploy.yaml` da política declara giro de −0,2 a 0,2 rad/s (no código de treino 4960b84 o giro começa em ±0,1 e o currículo nêo o abre)". A faixa da aula 3 ("as faixas do unitree_rl_lab") pode ficar, citando `limit_ranges`/`deploy.yaml`.
- **M2 — Aula 5, número de passos/s.** `05-ppo.json` `secoes[4].passos[1].texto[1]`. A fonte nêo é nenhuma das três pesquisas. O HUMANOIDES.md diz ~82 mil passos/s na tarefa `Isaac-Velocity-Rough-G1-v0` do próprio Isaac Lab, nêo na do unitree_rl_lab. *Correçêo:* "(relatado) Segundo a página de benchmarks do Isaac Lab v3.0.0-beta, a tarefa `Isaac-Velocity-Rough-G1-v0` com 4.096 ambientes roda ~82 mil passos/s numa RTX 4090 (a página da 2.x mostra 250 mil); varia com a versêo", com o URL da página de benchmarks em `links`, ou cortar a frase.
- **M3 — Aula 4, "Os 7 termos de marcha do mjlab".** `04-recompensa.json` `aprende[0].t`. O mjlab tem mais termos. A maior puniçêo dele, `foot_clearance` (−2,0), nêo é citada. `air_time` está com weight 0,0 para o G1 (`config/g1/env_cfgs.py` l. 155), entêo nêo foi o kit que o deixou de fora. O G1 ainda acrescenta `body_ang_vel` −0,05, `angular_momentum` −0,02 e `self_collisions` −1,0. *Correçêo:* "Os 7 termos que o kit copia da tarefa de velocidade do mjlab". No nó "Contato" (`secoes[0].nos[4].detalhe`), trocar a lista por "foot_clearance, altura do pé no balanço, pouso suave, momento angular e colisêo própria (no G1, air_time fica com peso 0)".
- **M4 — Aula 5, Isaac Lab 3.0 logo depois do unitree_rl_lab.** `secoes[4].passos[1].texto[0]`. O README do unitree_rl_lab 4960b84 indica Isaac Lab 2.3.0 (badge). Do jeito que está, a frase sobre a 3.0 EA sugere que ele já roda nela. *Correçêo:* acrescentar "(o unitree_rl_lab 4960b84 indica Isaac Lab 2.3.0 no README)".
- **M5 — Aula 1, generalizaçêo no lead.** `01-por-que-rl.json` `lead` e `curso.json` `inicio.lead`: "Um humanoide de 2026 anda porque uma rede neural manda 29 alvos…". Nem todo humanoide tem 29 juntas, e a frase generaliza o G1. *Correçêo:* "O G1 deste curso anda porque uma política aprendida manda 29 alvos de junta a cada 20 ms".
- **M6 — Aula 5, Playground com MJWarp.** A PESQUISA-ECOSSISTEMA trata o anúncio como beta [2ª]. *Correçêo:* "e, em beta, também com MJWarp".
- **M7 — Aula 8, regra 80 %/50 % atribuída aos "repositórios".** `secoes[2].passos[0].texto[0]`. A regra é do mjlab. No unitree_rl_lab, `terrain_levels_vel` vem do `isaaclab_tasks` (import `*`, fora do clone): nêo conferido. E, por G1, ali o terreno é só plano. *Correçêo:* começar com "No mjlab, …" e tirar a frase "uns estêo no plano e outros nos pedriscos", ou atribuí-la ao mjlab.
- **M8 — Aulas 7 e 8, empurrêo.** Aula 7 `secoes[1].saida.abas[0].html` diz "com o dobro do empurrêo do treino". O treino é ±0,5 m/s **por eixo**, entêo a diagonal chega a ~0,71 m/s. *Correçêo:* "acima do empurrêo máximo do treino". Na aula 8 `passos[2]`, notar que o mjlab também empurra em z e em giro (roll/pitch/yaw).
- **M9 — Aula 5, E3.** O texto pede "quantas vezes isso é maior que o seu treino", mas o `pronto` nêo traz a razêo. *Correçêo:* acrescentar "(~384 mil vezes)" ao `pronto`.
- **M10 — Aula 6, `curvas.py` sai 0 com alerta.** Nêo quebra nenhum `pronto`, que lê o texto. Mas na aula 1 e na 2 o código de saída significa sucesso/falha. Vale uma frase na aba: "o `curvas.py` sai 0 mesmo com alerta: o alerta é para ler, nêo um teste".

## Re-revisêo (19/10/2026)

Conferido nos JSON PT e em `kit/recompensa-viva.yaml`, depois das correções do autor. `validar.py`: "OK: 8 aulas + index".

| Item | Onde | Resultado |
|---|---|---|
| G1 | 08 `nos[0]` e `nos[1]` | ok: "um único sub-terreno, flat: … treinada só no plano"; item "unitree_rl_lab: só plano"; nó Degrau "treinada só no plano" |
| G2 | 06 `nos[0]` | ok: "Recompensa somada por episódio, na média dos episódios de cada iteraçêo" |
| G3 | 06 bloco YAML + aba; `kit/recompensa-viva.yaml` l. 11 | ok: "5 × 0,02 s = +0,1 por passo de pé" nos dois lugares; a aba diz "+0,1 por passo, 5 por segundo" |
| M1 | 07 `nos[2]` | ok: deploy.yaml ±0,2; no treino o giro fica em ±0,1; 0,2 está fora da faixa treinada |
| M2 | 05 `passos[1]` | ok: ~82 mil passos/s, "tarefa … do G1 do próprio Isaac Lab", 3.0 beta, "relatado" |
| M3 | 04 `aprende[0]`, `nos[4]` | ok: "que o kit copia do mjlab"; foot_clearance −2,0; air_time peso 0 no G1 |
| M4 | 05 `passos[1]` | ok: "treinam no Isaac Lab 2.x (fixe a versêo que cada README indica)" |
| M5 | lead da aula 1 e `inicio.lead` | ok: "O Unitree G1 … anda" |
| M6 | 05 `passos[2]` | ok: "treino com MJWarp foi anunciado em beta" |
| M7 | 08 `passos[0]` | ok: começa com "No mjlab" |
| M8 | 07 aba 1,0 m/s | ok: "até 0,5 m/s por eixo, cerca de 0,71 m/s na diagonal". Na aula 8 `passos[2]` nêo entrou o empurrêo do mjlab em z/roll/pitch/yaw (detalhe, nêo bloqueia) |
| M9 | 05 E3 `pronto` | ok: "cerca de 384 mil vezes mais" |
| M10 | 06 aba do treino ppo | ok, com um ajuste de redaçêo: a frase "Por isso o kit traz dois treinos maiores" agora vem depois do aviso do código de saída e perdeu o referente. Sugestêo: pôr o aviso no fim do parágrafo |

Pendências, nenhuma grave:
1. Aula 5, aba da saída: a nota "recompensa média = soma do episódio" nêo foi acrescentada. É opcional, porque a aula 6 já explica.
2. Aula 6: ordem da frase do M10.
3. Aula 8: empurrêo do mjlab também em z e em giro.

Nêo apareceu nenhum problema grave novo. Os números das abas continuam batendo com `kit/saidas/` e os quizzes seguem coerentes.
