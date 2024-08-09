#!/bin/bash

DIR="/workspace" # TODO: change this to the directory you want to monitor
OUTPUT_FILE="disk_usage_log.txt"
INTERVAL=5

echo "Periodic Disk Usage Log" > $OUTPUT_FILE
echo "========================" >> $OUTPUT_FILE

while true; do
    echo "" >> $OUTPUT_FILE
    echo "Timestamp: $(date)" >> $OUTPUT_FILE
    echo "------------------------" >> $OUTPUT_FILE
    du --max-depth=2 -h $DIR >> $OUTPUT_FILE
    echo "------------------------" >> $OUTPUT_FILE
    df -h / >> $OUTPUT_FILE
    echo "========================" >> $OUTPUT_FILE
    sleep $INTERVAL
done