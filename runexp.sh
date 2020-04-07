#!/bin/bash

for ((i=0;i<4;i++))
do
  echo "Current random seed: $i"
  CUDA_VISIBLE_DEVICES=4 python main.py --alg HTRPO --env SawyerLift --seed $i --eval_interval 96000 --num_steps 5000000 --num_eval 200 
done
