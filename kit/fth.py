"""fth.py — funções comuns do kit do curso FTH 1 (Ensinar a andar).

Acha a bancada em $FTH (padrão: a pasta acima do repositório do curso, ou seja ~/fth):
  ~/fth/mujoco_menagerie   (do curso 0; usamos unitree_g1/scene.xml)
  ~/fth/unitree_rl_lab     (clone parcial: só a política de velocidade do G1, Apache-2.0)
Monta um ambiente de locomoção mínimo no estilo "manager-based" (observações, ações,
recompensas e terminações com nome) em MuJoCo puro, rodando em CPU.
Testado com mujoco 3.16.0, numpy 2.5.4, onnxruntime 1.31.0, torch 2.14.1+cpu,
Menagerie 62ab00f e unitree_rl_lab 4960b84, em 19/10/2026.
"""
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

KIT = Path(__file__).resolve().parent
RAIZ = Path(os.environ.get("FTH", KIT.parents[1]))
MENAGERIE = RAIZ / "mujoco_menagerie"
POLITICA = RAIZ / "unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0"
G1 = "unitree_g1/scene.xml"
DT_POLITICA = 0.02  # 50 Hz, igual ao step_dt do deploy.yaml
PES = ("left_ankle_roll_link", "right_ankle_roll_link")


def sair(msg):
    sys.exit("ERRO: " + msg)


def deploy():
    """Lê o deploy.yaml da política do unitree_rl_lab (ganhos, pose padrão e ordem das juntas)."""
    p = POLITICA / "params/deploy.yaml"
    if not p.exists():
        sair(f"não achei {p}. Rode a preparação da aula 1.")
    return yaml.safe_load(p.read_text())


def carregar(ganhos="rl_lab", kp=None, kd=None, xml=G1, spec=None):
    """Carrega o G1 do Menagerie e troca o PD dos 29 atuadores de posição.

    O atuador <position> do MuJoCo já é um PD: força = kp·(alvo − q) − kd·q̇.
    ganhos="rl_lab": kp/kd por junta do deploy.yaml (ordem do SDK = ordem do Menagerie).
    ganhos=None com kp, kd: o mesmo valor em todas as juntas.
    """
    caminho = MENAGERIE / xml
    if spec is not None:
        m = spec.compile()
    elif not caminho.exists():
        sair(f"não achei {caminho}. Rode a preparação da aula 1 (ou a do curso 0).")
    else:
        m = mujoco.MjModel.from_xml_path(str(caminho))
    if ganhos == "rl_lab":
        c = deploy()
        kps, kds = np.array(c["stiffness"], float), np.array(c["damping"], float)
    else:
        kps, kds = np.full(m.nu, float(kp)), np.full(m.nu, float(kd))
    for a in range(m.nu):
        m.actuator_gainprm[a, 0] = kps[a]
        m.actuator_biasprm[a, 1] = -kps[a]
        m.actuator_biasprm[a, 2] = -kds[a]
    return m


def pose_padrao():
    """Pose padrão (q0) e mapa de juntas: o deploy.yaml lista na ordem do Isaac Lab; o MuJoCo usa a do SDK."""
    c = deploy()
    mapa = np.array(c["joint_ids_map"])
    q0 = np.zeros(len(mapa))
    q0[mapa] = c["default_joint_pos"]
    return q0, mapa


def gravidade_no_corpo(d, corpo):
    """Gravidade (0, 0, −1) vista no referencial do corpo: (0, 0, −1) = em pé, x/y ≠ 0 = inclinado."""
    R = d.xmat[corpo].reshape(3, 3)
    return R.T @ np.array([0.0, 0.0, -1.0])


def inclinacao_graus(d, corpo):
    g = gravidade_no_corpo(d, corpo)
    return math.degrees(math.acos(max(-1.0, min(1.0, -g[2]))))


def vel_base(d):
    """Velocidade linear (no referencial da pelve) e angular (já local no MuJoCo) da base livre."""
    R = d.xmat[1].reshape(3, 3)
    return R.T @ d.qvel[0:3], d.qvel[3:6].copy()


def contato_pes(m, d, ids_pes):
    """Força normal (N) de cada pé com o chão."""
    f6 = np.zeros(6)
    forca = np.zeros(len(ids_pes))
    for i in range(d.ncon):
        c = d.contact[i]
        for g in (c.geom1, c.geom2):
            b = m.geom_bodyid[g]
            for k, p in enumerate(ids_pes):
                if b == p:
                    mujoco.mj_contactForce(m, d, i, f6)
                    forca[k] += abs(f6[0])
    return forca


def vel_pes_xy(m, d, ids_pes):
    out = np.zeros((len(ids_pes), 2))
    v6 = np.zeros(6)
    for k, p in enumerate(ids_pes):
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, p, v6, 0)
        out[k] = v6[3:5]
    return out


# ---------------- recompensa ----------------
def ler_recompensa(caminho=None):
    p = Path(caminho) if caminho else KIT / "recompensa.yaml"
    if not p.exists():
        sair(f"não achei {caminho or 'kit/recompensa.yaml'}.")
    cfg = yaml.safe_load(p.read_text())
    for nome, t in cfg.items():
        if nome not in TERMOS:
            sair(f"termo desconhecido em {p.name}: {nome}. Termos do kit: {', '.join(TERMOS)}")
        if "weight" not in t:
            sair(f"o termo {nome} não tem weight.")
    return cfg


def _track_lin(e, t):
    err = np.sum((e["cmd"][:2] - e["v"][:2]) ** 2) + e["v"][2] ** 2
    return math.exp(-err / t.get("std", 0.5) ** 2)


def _track_ang(e, t):
    err = (e["cmd"][2] - e["w"][2]) ** 2 + np.sum(e["w"][:2] ** 2)
    return math.exp(-err / t.get("std", 0.7071) ** 2)


def _upright(e, t):
    return math.exp(-np.sum(e["g_tronco"][:2] ** 2) / t.get("std", 0.4472) ** 2)


def _pose(e, t):
    return math.exp(-np.mean((e["q"] - e["q0"]) ** 2) / t.get("std", 0.3) ** 2)


def _limites(e, t):
    lo, hi = e["lim"][:, 0], e["lim"][:, 1]
    return float(np.sum(np.clip(lo - e["q"], 0, None)) + np.sum(np.clip(e["q"] - hi, 0, None)))


def _action_rate(e, t):
    return float(np.sum((e["a"] - e["a_ant"]) ** 2))


def _foot_slip(e, t):
    em_contato = e["f_pes"] > t.get("limiar_n", 1.0)
    return float(np.sum(np.sum(e["v_pes"] ** 2, axis=1) * em_contato))


def _vivo(e, t):
    return 1.0


def _vel_frente(e, t):
    return float(e["v"][0])


TERMOS = {
    "track_linear_velocity": _track_lin,
    "track_angular_velocity": _track_ang,
    "upright": _upright,
    "pose": _pose,
    "dof_pos_limits": _limites,
    "action_rate_l2": _action_rate,
    "foot_slip": _foot_slip,
    "is_alive": _vivo,
    "forward_velocity": _vel_frente,
}


def termos_recompensa(estado, cfg):
    """Devolve {termo: valor já multiplicado pelo weight e pelo dt} para um passo da política."""
    return {n: t["weight"] * TERMOS[n](estado, t) * DT_POLITICA for n, t in cfg.items()}


# ---------------- ambiente ----------------
OBS_TERMOS = (("base_ang_vel", 3, 0.2), ("projected_gravity", 3, 1.0), ("velocity_commands", 3, 1.0),
              ("joint_pos_rel", 29, 1.0), ("joint_vel_rel", 29, 0.05), ("last_action", 29, 1.0))


class Ambiente:
    """Um G1 andando no MuJoCo, com a estrutura de um ambiente "manager-based":
    observações, ação, recompensas e terminações, cada uma com nome.
    Ação = 29 números; alvo da junta = pose padrão + 0,25 · ação (como no deploy.yaml)."""

    def __init__(self, recompensa=None, seed=0, episodio_s=20.0, historico=1, comando=None, xml=G1, spec=None):
        self.m = carregar(xml=xml, spec=spec)
        self.d = mujoco.MjData(self.m)
        self.q0, self.mapa = pose_padrao()
        self.cfg_r = ler_recompensa(recompensa) if not isinstance(recompensa, dict) else recompensa
        self.rng = np.random.default_rng(seed)
        self.max_passos = int(round(episodio_s / DT_POLITICA))
        self.dec = int(round(DT_POLITICA / self.m.opt.timestep))
        self.H = historico
        self.comando_fixo = comando
        self.tronco = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.pelve = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.pes = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, p) for p in PES]
        self.lim = self.m.jnt_range[1:].copy()
        self.n_obs = sum(n for _, n, _ in OBS_TERMOS) * self.H
        self.n_acao = self.m.nu
        self.reset()

    def sortear_comando(self):
        if self.comando_fixo is not None:
            return np.array(self.comando_fixo, float)
        return np.array([self.rng.uniform(-0.5, 1.0), self.rng.uniform(-0.3, 0.3), self.rng.uniform(-0.2, 0.2)])

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        self.d.qpos[7:] = self.q0
        self.d.ctrl[:] = self.q0
        mujoco.mj_forward(self.m, self.d)
        self.cmd = self.sortear_comando()
        self.a = np.zeros(self.n_acao)
        self.a_ant = np.zeros(self.n_acao)
        self.passo = 0
        t = self.termos_obs()
        self.hist = [[x] * self.H for x in t]
        return self.obs()

    def termos_obs(self):
        d = self.d
        _, w = vel_base(d)
        g = gravidade_no_corpo(d, self.pelve)
        jp = (d.qpos[7:] - self.q0)[self.mapa]
        jv = d.qvel[6:][self.mapa]
        crus = (w, g, self.cmd, jp, jv, self.a)
        return [x * esc for x, (_, _, esc) in zip(crus, OBS_TERMOS)]

    def obs(self):
        return np.concatenate([np.concatenate(h) for h in self.hist]).astype(np.float32)

    def estado(self):
        d = self.d
        v, w = vel_base(d)
        return {"cmd": self.cmd, "v": v, "w": w, "g_tronco": gravidade_no_corpo(d, self.tronco),
                "q": d.qpos[7:].copy(), "q0": self.q0, "lim": self.lim, "a": self.a, "a_ant": self.a_ant,
                "f_pes": contato_pes(self.m, d, self.pes), "v_pes": vel_pes_xy(self.m, d, self.pes)}

    def empurrar(self, dv):
        """Empurrão: soma uma velocidade (m/s) à base, como o push_by_setting_velocity do mjlab."""
        self.d.qvel[0:len(dv)] += dv

    def step(self, acao):
        self.a_ant = self.a
        self.a = np.asarray(acao, float)
        alvo = np.zeros(self.n_acao)
        alvo[self.mapa] = self.q0[self.mapa] + 0.25 * self.a
        self.d.ctrl[:] = alvo
        for _ in range(self.dec):
            mujoco.mj_step(self.m, self.d)
        self.passo += 1
        termos = termos_recompensa(self.estado(), self.cfg_r)
        for h, x in zip(self.hist, self.termos_obs()):
            h.pop(0)
            h.append(x)
        causa = None
        if inclinacao_graus(self.d, self.pelve) > 70.0 or self.d.qpos[2] < 0.3:
            causa = "queda"
        elif self.passo >= self.max_passos:
            causa = "tempo"
        return self.obs(), sum(termos.values()), causa, termos


class PoliticaPronta:
    """A política de velocidade do unitree_rl_lab (ONNX): 480 números de entrada (6 termos × 5 passos) → 29 ações."""

    def __init__(self):
        import onnxruntime as ort
        arq = POLITICA / "exported/policy.onnx"
        if not arq.exists():
            sair(f"não achei {arq}. Rode a preparação da aula 1.")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.s = ort.InferenceSession(str(arq), so, providers=["CPUExecutionProvider"])
        self.nome = "unitree_rl_lab velocity v0 (ONNX)"
        self.historico = 5

    def __call__(self, obs):
        return self.s.run(None, {"obs": obs[None]})[0][0]


class PoliticaTreinada:
    """Um checkpoint do kit/treinar.py (torch)."""

    def __init__(self, arq):
        import torch
        from ppo import AtorCritico
        if not Path(arq).exists():
            sair(f"não achei {arq}.")
        ck = torch.load(arq, map_location="cpu", weights_only=False)
        self.rede = AtorCritico(ck["n_obs"], ck["n_acao"])
        self.rede.load_state_dict(ck["rede"])
        self.rede.eval()
        self.torch = torch
        self.nome = f"{Path(arq).name} (iteração {ck['iteracao']})"
        self.historico = 1

    def __call__(self, obs):
        with self.torch.no_grad():
            return self.rede.acao_media(self.torch.as_tensor(obs[None])).numpy()[0]


def politica(arg):
    return PoliticaPronta() if arg in (None, "pronta") else PoliticaTreinada(arg)
