#!/bin/bash

epsilon="0.0 0.05 0.1 0.15 0.2 0.25 0.3"


for eps in $epsilon
do
	sttr
	sbatch scripts/trans7.sh $eps
	sbatch scripts/trans7_3.sh $eps
	sbatch scripts/vanilla.sh $eps
	sbatch scripts/trans11.sh $eps
done
