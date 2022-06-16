#!/bin/bash

for i in {1..20}
do
	echo "Round $i:"
	python3 random_dist_preprocessing.py
	python3 random_dist.py
	echo ""
	echo ""
done
