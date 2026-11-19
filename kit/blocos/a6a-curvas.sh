F=${FTH:-$HOME/fth}; cd $F/locomocao && . $F/.venv/bin/activate
python3 kit/curvas.py $F/treino/ppo; echo "saída: $?"
