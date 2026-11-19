F=${FTH:-$HOME/fth}; cd $F/locomocao && . $F/.venv/bin/activate
python3 kit/avaliar.py --episodios 20 --empurrao 1.0; echo "saída: $?"
