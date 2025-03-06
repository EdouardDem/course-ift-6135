#!/bin/bash

START_TIME=$(date +%s)

# Run all experiments
# ./question-2.sh
# ./question-3.sh
# ./question-4.sh
# ./question-5.sh
# ./question-6.sh
# ./question-7.sh
# ./question-7-bis.sh

# Run all plots
python question-2-plot.py
python question-3-plot.py
python question-4-plot.py
python question-5-plot.py
python question-6-plot.py
python question-7-plot.py
python question-7-bis-plot.py
./question-8-plot.sh

END_TIME=$(date +%s)

echo "Total execution time: $((END_TIME - START_TIME)) seconds"
