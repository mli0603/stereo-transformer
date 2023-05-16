#!/bin/bash

epsilon="0.0 0.05 0.1 0.15 0.2 0.25 0.3"


for eps in $epsilon
do
	sttr
	sbatch scripts/vanilla_cospgd.sh $eps
done
