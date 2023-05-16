#!/bin/bash

epsilon="0.0 0.1 0.2 0.3 0.15 0.25 0.05"


for eps in $epsilon
do
	sttr
	sbatch scripts/trans11_fgsm.sh $eps
done
