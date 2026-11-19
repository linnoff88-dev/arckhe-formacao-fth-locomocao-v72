F=${FTH:-$HOME/fth}; cd $F/locomocao && . $F/.venv/bin/activate
FTH=/tmp/sem-bancada python3 kit/andar_demo.py; echo "saída: $?"
