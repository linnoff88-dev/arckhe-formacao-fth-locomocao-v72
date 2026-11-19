F=${FTH:-$HOME/fth}; cd $F/locomocao && . $F/.venv/bin/activate
python3 kit/termos.py $F/treino/andando.npz; echo "saída: $?"
