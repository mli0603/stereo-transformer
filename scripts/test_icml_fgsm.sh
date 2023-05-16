#!/bin/bash

epsilon="0.03 0.05 0.1 0.15 0.2 0.25 0.3"


for eps in $epsilon
do
	job_name="STTR_FGSM_${eps}"
	sbatch -J ${job_name} scripts/fgsm.sh $eps
done
