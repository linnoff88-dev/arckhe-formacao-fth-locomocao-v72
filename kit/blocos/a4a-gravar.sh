F=${FTH:-$HOME/fth}; cd $F/locomocao && . $F/.venv/bin/activate
mkdir -p $F/treino
python3 kit/andar_demo.py --vx 0.7 --gravar $F/treino/andando.npz | tail -1
python3 kit/andar_demo.py --vx 0 --gravar $F/treino/parado.npz | tail -1
