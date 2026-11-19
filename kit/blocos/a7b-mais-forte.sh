F=${FTH:-$HOME/fth}; cd $F/locomocao && . $F/.venv/bin/activate
for e in 1.5 2.0; do python3 kit/avaliar.py --episodios 20 --empurrao $e | tail -1; done
