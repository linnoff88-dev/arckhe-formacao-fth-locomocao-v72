F=${FTH:-$HOME/fth}; cd $F/locomocao && . $F/.venv/bin/activate
python3 kit/avaliar.py --politica $F/treino/ppo/modelo_20.pt --episodios 5 | tail -2
