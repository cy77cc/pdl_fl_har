#!/bin/bash
python main.py --dataset har70 --model $1 --algorithm $2 --l2 $3 --dp_epsilon $4 --note $5 
python main.py --dataset harth --model $1 --algorithm $2 --l2 $3 --dp_epsilon $4 --note $5 
python main.py --dataset harth66 --model $1 --algorithm $2 --l2 $3 --dp_epsilon $4 --note $5 
python main.py --dataset wisdm --model $1 --algorithm $2 --l2 $3 --dp_epsilon $4 --note $5 
python main.py --dataset har --model $1 --algorithm $2 --l2 $3 --dp_epsilon $4 --note $5 
python main.py --dataset pamap2 --model $1 --algorithm $2 --l2 $3 --dp_epsilon $4 --note $5 

python attack.py --dataset har70 --model $1 --algorithm $2 --l2 $3 --note $5
python attack.py --dataset harth --model $1 --algorithm $2 --l2 $3 --note $5
python attack.py --dataset harth66 --model $1 --algorithm $2 --l2 $3 --note $5
python attack.py --dataset wisdm --model $1 --algorithm $2 --l2 $3 --note $5
python attack.py --dataset har --model $1 --algorithm $2 --l2 $3 --note $5
python attack.py --dataset pamap2 --model $1 --algorithm $2 --l2 $3 --note $5
