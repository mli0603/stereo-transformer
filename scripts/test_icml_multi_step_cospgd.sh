#!/bin/bash

epsilon="0.03"
alpha="0.01"
iterations="3 5 10 20 40"


for it in $iterations
do
	#sttr
	job_name="STTR_m_Cos_${it}"
	sbatch -J ${job_name} scripts/multi_step_cospgd.sh $epsilon $alpha $it
done
