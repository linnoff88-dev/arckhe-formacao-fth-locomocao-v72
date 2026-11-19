F=${FTH:-$HOME/fth}; cd $F/locomocao && . $F/.venv/bin/activate
for kd in 0 2 10; do python3 kit/degrau_junta.py --kp 100 --kd $kd | tail -2; done
