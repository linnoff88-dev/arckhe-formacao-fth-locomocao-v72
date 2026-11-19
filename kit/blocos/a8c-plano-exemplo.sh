F=${FTH:-$HOME/fth}; cd $F/locomocao && . $F/.venv/bin/activate
python3 kit/conferir_plano.py kit/exemplos/plano-treino.md; echo "saída: $?"
