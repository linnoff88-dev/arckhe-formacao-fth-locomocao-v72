F=${FTH:-$HOME/fth}; cd $F/locomocao && . $F/.venv/bin/activate
python3 kit/termos.py $F/treino/parado.npz --comando-vx 0.7
