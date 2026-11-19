F=${FTH:-$HOME/fth}; cd $F/locomocao && . $F/.venv/bin/activate
for x in zero pronta; do python3 kit/checar_ambiente.py --acao $x | tail -2; done
