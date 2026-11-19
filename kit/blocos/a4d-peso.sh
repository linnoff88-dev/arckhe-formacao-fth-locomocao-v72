F=${FTH:-$HOME/fth}; cd $F/locomocao && . $F/.venv/bin/activate
sed "s/track_linear_velocity: {weight: 2.0/track_linear_velocity: {weight: 0.5/" kit/recompensa.yaml > $F/treino/minha.yaml
python3 kit/termos.py $F/treino/andando.npz --recompensa $F/treino/minha.yaml | tail -1
python3 kit/termos.py $F/treino/parado.npz --comando-vx 0.7 --recompensa $F/treino/minha.yaml | tail -1
