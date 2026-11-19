F=${FTH:-$HOME/fth}; cd $F/locomocao && . $F/.venv/bin/activate
timeout 900 python3 kit/treinar.py --iters 20; echo "saída: $?"
test -f $F/treino/ppo/modelo_20.pt && echo "arquivo existe: modelo_20.pt"
wc -l < $F/treino/ppo/progresso.csv
