#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_folder> <output_file.7z>"
    exit 1
fi

SOURCE_FOLDER="$1"
OUTPUT_FILE="$2"

# Check if required tools are installed
if ! command -v 7z &> /dev/null; then
    echo "7z is not installed. Please install p7zip-full."
    exit 1
fi

if ! command -v pigz &> /dev/null; then
    echo "pigz is not installed. Please install pigz."
    exit 1
fi

# Get the number of CPU cores
NUM_CORES=$(nproc)

# Compress the folder using 7zip with ultra settings and multi-threaded gzip
7z a -t7z -m0=lzma2 -mx=9 -mfb=64 -md=32m -ms=on -mmt=$NUM_CORES "$OUTPUT_FILE" "$SOURCE_FOLDER"

echo "Compression complete. Output file: $OUTPUT_FILE"

# Optional: Calculate and display the compression ratio
ORIGINAL_SIZE=$(du -sb "$SOURCE_FOLDER" | cut -f1)
COMPRESSED_SIZE=$(du -sb "$OUTPUT_FILE" | cut -f1)
RATIO=$(echo "scale=2; $COMPRESSED_SIZE * 100 / $ORIGINAL_SIZE" | bc)

echo "Original size: $ORIGINAL_SIZE bytes"
echo "Compressed size: $COMPRESSED_SIZE bytes"
echo "Compression ratio: $RATIO%"