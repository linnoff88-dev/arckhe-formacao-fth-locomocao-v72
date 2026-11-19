F=${FTH:-$HOME/fth}; cd $F/locomocao && . $F/.venv/bin/activate
python3 kit/andar_demo.py --vx 0.7 --s 10; echo "saída: $?"
