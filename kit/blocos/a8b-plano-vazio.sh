F=${FTH:-$HOME/fth}; cd $F/locomocao && . $F/.venv/bin/activate
[ -f $F/treino/plano-treino.md ] || cp kit/plano-treino-modelo.md $F/treino/plano-treino.md
python3 kit/conferir_plano.py $F/treino/plano-treino.md; echo "saída: $?"
