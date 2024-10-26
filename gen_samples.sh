#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=10GB
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=gpu

# Directory containing files
DIR="tests/test_data/"
# Number of iterations
ITERATIONS=10000

# Function to generate random float between two values
random_float() {
  echo "$(awk -v min=$1 -v max=$2 'BEGIN{srand(); print min+(rand()*(max-min))}')"
}

# Function to generate random integer between two values
random_int() {
  echo $(( ( RANDOM % ($2 - $1 + 1) ) + $1 ))
}

for ((i=1; i<=ITERATIONS; i++)); do
  # Get two random files from the directory
  file1=$(ls "$DIR" | shuf -n 1)
  file2=$(ls "$DIR" | shuf -n 1)
  
  # Ensure the two files are different
  while [ "$file1" == "$file2" ]; do
    file2=$(ls "$DIR" | shuf -n 1)
  done

  # Generate random values for -var, -temp, and -l
  #var=$(random_int 1 20)
  var=5
  temp=$(random_float 0.5 1.5)
  l=$(random_int 400 1000)

  # Run the command with the generated values
  aria sample -m large -c large-abs-inst.safetensors -p "$DIR/$file1" "$DIR/$file2" -var "$var" -trunc 0 -l "$l" -temp "$temp" -form ABA
  
done