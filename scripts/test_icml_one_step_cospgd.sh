#!/bin/bash

epsilon="0.03 0.05 0.1 0.15 0.2 0.25 0.3"
alpha="0.15"
iterations="1"


for eps in $epsilon
do
	job_name="STTR_1_Cos_${eps}"
	sbatch -J ${job_name} scripts/one_step_cospgd.sh $eps $eps
done
