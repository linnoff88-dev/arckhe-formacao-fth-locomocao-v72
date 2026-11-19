F=${FTH:-$HOME/fth}; cd $F/locomocao && . $F/.venv/bin/activate
python3 kit/checar_ambiente.py --acao aleatoria; echo "saída: $?"
