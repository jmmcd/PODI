#!/bin/bash

for ((i=0;$i<1;i=$i+1)); do
	echo $i;
	# python ge.py > ~/Dropbox/results/Injection/GE/injection/${i}.dat
	# python bubble_down.py run_grow > ~/Dropbox/results/Injection/GP/grow/${i}.dat
	# python bubble_down.py run_bubble_down > ~/Dropbox/results/Injection/GP/bubble_down/${i}.dat
	for problem in pagie-2 vladislavleva-14; do
		echo $problem
	done
	# python gp.py > ~/Dropbox/LBYL/pagie-2d-GSGP-optimal-ms-100-10-10-3-${i}.dat
done

# python bubble_down.py grow_structure "."
# python bubble_down.py bubble_down_structure "."
