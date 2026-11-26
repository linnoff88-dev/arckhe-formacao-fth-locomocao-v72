F=${FTH:-$HOME/fth}; mkdir -p $F && cd $F
[ -d .venv ] || python3.12 -m venv .venv
. .venv/bin/activate && pip install -q mujoco==3.16.0 numpy pyyaml onnxruntime==1.31.0
pip install -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
[ -d locomocao ] || git clone -q --depth 1 https://github.com/linnoff88-dev/formacao-fth-locomocao-v72 locomocao
[ -d mujoco_menagerie ] || git clone -q --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie
(cd mujoco_menagerie && git sparse-checkout add unitree_g1 && git checkout -q 62ab00f)
[ -d unitree_rl_lab ] || git clone -q --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_rl_lab
cd unitree_rl_lab && git sparse-checkout set deploy/robots/g1_29dof/config/policy/velocity
git checkout -q 4960b84 && git log -1 --format='unitree_rl_lab %h (%ad)' --date=short
python3 -c "import mujoco, onnxruntime, torch; print('mujoco', mujoco.__version__, '· onnxruntime', onnxruntime.__version__, '· torch', torch.__version__)"
