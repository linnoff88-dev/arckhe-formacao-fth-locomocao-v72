F=${FTH:-$HOME/fth}; cd $F/locomocao && . $F/.venv/bin/activate
python3 kit/terreno.py --degraus 0.02 0.03 0.04 0.06 0.10; echo "saída: $?"
