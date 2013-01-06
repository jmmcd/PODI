#!/bin/bash

for ((i=0;$i<30;i=$i+1)); do
	echo $i;
	python ge.py > ~/Dropbox/results/Injection/GE/injection/${i}.dat
	# python bubble_down.py run_grow > ~/Dropbox/results/Injection/GP/grow/${i}.dat
	# python bubble_down.py run_bubble_down > ~/Dropbox/results/Injection/GP/bubble_down/${i}.dat
done

# python bubble_down.py grow_structure "."
# python bubble_down.py bubble_down_structure "."
