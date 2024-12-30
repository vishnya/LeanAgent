# Function to parse the text file and count occurrences of each theorem
def count_theorem_occurrences(file_path):
    theorem_counts = {}
    
    # Open the file and read lines
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Theorem:"):
                theorem_name = line.split(":")[1].strip()
                # Increment the count of the theorem in the dictionary
                if theorem_name in theorem_counts:
                    theorem_counts[theorem_name] += 1
                else:
                    theorem_counts[theorem_name] = 1
    
    return theorem_counts

# Example usage: Assuming the text file is named 'theorem_data.txt'
file_path = 'theorem_data_SciLean.txt'
theorem_occurrences = count_theorem_occurrences(file_path)

# Display the results
print(theorem_occurrences)
