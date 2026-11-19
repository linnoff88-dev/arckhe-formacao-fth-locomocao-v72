F=${FTH:-$HOME/fth}; cd $F/locomocao && . $F/.venv/bin/activate
deactivate; python3 kit/treinar.py --iters 1 2>&1 | tail -1
