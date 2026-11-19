F=${FTH:-$HOME/fth}; cd $F/locomocao && . $F/.venv/bin/activate
for x in cai-cedo com-vivo; do python3 kit/curvas.py kit/exemplos/$x; echo; done
