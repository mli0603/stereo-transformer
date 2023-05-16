#!/bin/bash

epsilon="0.03"
iterations="3 5 10 20 40 100"


for it in $iterations
do
	job_name="STTR_PGD_${it}"
	sbatch -J ${job_name} scripts/pgd.sh $epsilon $it
done
