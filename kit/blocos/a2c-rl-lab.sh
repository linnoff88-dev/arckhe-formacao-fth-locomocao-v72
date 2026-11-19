F=${FTH:-$HOME/fth}; cd $F/locomocao && . $F/.venv/bin/activate
python3 kit/ficar_em_pe.py --ganhos rl_lab; echo "saída: $?"
