import re

# expected_results_exp3 = {
#     'task1 Test R@10: Compfiles': [
#         60.64706363237122, 61.32526532789667, 61.6708414949816, 61.08772811083543, 60.261503729020575, 60.30689658891723, 60.60123583745497, 
#         60.153335213407054, 59.7709911492489, 60.00540189142794, 59.91546810620901,  59.88792078948343, 59.64851162308727, 59.420134063412014],
#     'task2 Test R@10: Mathematics in Lean Source': [
#         65.48591635672048, 65.75192142681436, 66.4970212474497, 66.49549432082041, 68.10014741519532, 67.93505866948891, 67.88206930569434, 
#         67.97385677238191, 67.61899831570183, 67.69791601581457, 67.581090432529, 67.75405538565178, 67.55591710838813],
#     'task3 Test R@10: PrimeNumberTheoremAnd': [
#         60.24074464461104, 61.601678946721826, 61.53315591144687, 62.06460041347961, 61.59168435062894, 62.286496517728665, 63.17427181726865, 
#         62.95367306725815, 62.84812634406436, 62.796309821659605, 62.573145480848204, 62.1576880065672],
#     'task4 Test R@10: Math Workshop': [
#         63.32418885891475, 65.32513361607977, 64.5102791379126, 67.1694072669325, 66.1893594713575, 67.5335852605448, 68.069787398159,
#         68.17676215211705, 67.79894096460394, 68.11922176655432, 68.37256090708857],
#     'task5 Test R@10: FLT': [
#         67.05795410771319, 67.50657961559557, 68.07826197730941, 68.94304048302321, 69.25704440208798, 69.06493687974346, 69.13220145373448, 
#         69.16048162503891, 69.47002065174642, 69.48301090016267],
#     'task6 Test R@10: PFR': [
#         62.58480312148852, 64.14710648979965, 64.12721536773596, 63.79502213074667, 63.97614939811086, 63.862150608153875, 63.66221697616451,
#         64.11137631469667, 63.98214095462508],
#     'task7 Test R@10: SciLean': [
#         68.18567421512381, 68.94310521480176, 70.20595528030704, 70.11859835977681, 70.07787376617783, 70.21904098724055, 69.74478926670498,
#         69.6667197138306],
#     'task8 Test R@10: Debate': [68.91867342976552, 68.8826460829178, 69.18231380479203, 69.31982755616717 , 69.58427923681404, 69.13118545804842, 69.48585868047269],
#     'task9 Test R@10: Matrix Cookbook': [71.04190539789398, 71.1094077126274, 71.08151370983799,  71.44692514637913, 71.19988426372417, 71.55403847771102],
#     'task10 Test R@10: Con-nf': [65.80145354021758, 66.0667469234885, 64.94783306581058, 66.0404405207776, 65.51765650080257 ],
#     'task11 Test R@10: Foundation': [65.21872588006309,  64.71692632303098 , 65.3519623141716, 65.13856379408705 ],
#     'task12 Test R@10: Saturn': [70.44871794871794, 73.2396449704142, 71.61242603550296 ],
#     'task13 Test R@10: LeanEuclid': [82.70537124802527, 83.11611374407583],
#     'task14 Test R@10: Lean4Lean': [81.45833333333331],
# }



# # Define a dictionary to store the R@10 results for each task
# task_results_exp = {
#     'task1 Test R@10: Compfiles': [],
#     'task2 Test R@10: Mathematics in Lean Source': [],
#     'task3 Test R@10: PrimeNumberTheoremAnd': [],
#     'task4 Test R@10: Math Workshop': [],
#     'task5 Test R@10: FLT': [],
#     'task6 Test R@10: PFR': [],
#     'task7 Test R@10: SciLean': [],
#     'task8 Test R@10: Debate': [],
#     'task9 Test R@10: Matrix Cookbook': [],
#     'task10 Test R@10: Con-nf': [],
#     'task11 Test R@10: Foundation': [],
#     'task12 Test R@10: Saturn': [],
#     'task13 Test R@10: LeanEuclid': [],
#     'task14 Test R@10: Lean4Lean': []
# }

# # Define a mapping of repo names to tasks
# repo_to_task_mapping = {
#     'merged_with_new_compfiles': 'task1 Test R@10: Compfiles',
#     'merged_with_new_mathematics_in_lean_source': 'task2 Test R@10: Mathematics in Lean Source',
#     'merged_with_new_PrimeNumberTheoremAnd': 'task3 Test R@10: PrimeNumberTheoremAnd',
#     'merged_with_new_lean-math-workshop': 'task4 Test R@10: Math Workshop',
#     'merged_with_new_FLT': 'task5 Test R@10: FLT',
#     'merged_with_new_pfr': 'task6 Test R@10: PFR',
#     'merged_with_new_SciLean': 'task7 Test R@10: SciLean',
#     'merged_with_new_debate': 'task8 Test R@10: Debate',
#     'merged_with_new_lean-matrix-cookbook': 'task9 Test R@10: Matrix Cookbook',
#     'merged_with_new_con-nf': 'task10 Test R@10: Con-nf',
#     'merged_with_new_Foundation': 'task11 Test R@10: Foundation',
#     'merged_with_new_Saturn': 'task12 Test R@10: Saturn',
#     'merged_with_new_LeanEuclid': 'task13 Test R@10: LeanEuclid',
#     'merged_with_new_lean4lean': 'task14 Test R@10: Lean4Lean'
# }

# # Define a dictionary to store the R@10 results for each task
# task_results_exp = {
#     'task1 Test R@10: SciLean': [],
#     'task2 Test R@10: FLT': [],
#     'task3 Test R@10: PFR': [],
#     'task4 Test R@10: PrimeNumberTheoremAnd': [],
#     'task5 Test R@10: Compfiles': [],
#     'task6 Test R@10: Debate': [],
#     'task7 Test R@10: Mathematics in Lean Source': [],
#     'task8 Test R@10: Lean4Lean': [],
#     'task9 Test R@10: Matrix Cookbook': [],
#     'task10 Test R@10: Math Workshop': [],
#     'task11 Test R@10: LeanEuclid': [],
#     'task12 Test R@10: Foundation': [],
#     'task13 Test R@10: Con-nf': [],
#     'task14 Test R@10: Saturn': []
# }

# # Define a mapping of repo names to tasks
# repo_to_task_mapping = {
#     'merged_with_new_SciLean': 'task1 Test R@10: SciLean',
#     'merged_with_new_FLT': 'task2 Test R@10: FLT',
#     'merged_with_new_pfr': 'task3 Test R@10: PFR',
#     'merged_with_new_PrimeNumberTheoremAnd': 'task4 Test R@10: PrimeNumberTheoremAnd',
#     'merged_with_new_compfiles': 'task5 Test R@10: Compfiles',
#     'merged_with_new_debate': 'task6 Test R@10: Debate',
#     'merged_with_new_mathematics_in_lean_source': 'task7 Test R@10: Mathematics in Lean Source',
#     'merged_with_new_lean4lean': 'task8 Test R@10: Lean4Lean',
#     'merged_with_new_lean-matrix-cookbook': 'task9 Test R@10: Matrix Cookbook',
#     'merged_with_new_lean-math-workshop': 'task10 Test R@10: Math Workshop',
#     'merged_with_new_LeanEuclid': 'task11 Test R@10: LeanEuclid',
#     'merged_with_new_Foundation': 'task12 Test R@10: Foundation',
#     'merged_with_new_con-nf': 'task13 Test R@10: Con-nf',
#     'merged_with_new_Saturn': 'task14 Test R@10: Saturn'
# }

# Define a dictionary to store the R@10 results for each task
task_results_exp = {
    'task1 Test R@10: Zeta 3 Irrational': [],
    'task2 Test R@10: Formal Book': [],
    'task3 Test R@10: Formalization of Constructable Numbers': [],
    'task4 Test R@10: Carleson': [],
    'task5 Test R@10: LeanAPAP': [],
    'task6 Test R@10: Hairy Ball Theorem': [],
    'task7 Test R@10: Coxeter': [],
    'task8 Test R@10: Lean4 PDL': []
}

# Define a mapping of repo names to tasks
repo_to_task_mapping = {
    'merged_with_new_zeta_3_irrational': 'task1 Test R@10: Zeta 3 Irrational',
    'merged_with_new_formal_book': 'task2 Test R@10: Formal Book',
    'merged_with_new_Formalisation-of-constructable-numbers': 'task3 Test R@10: Formalization of Constructable Numbers',
    'merged_with_new_carleson': 'task4 Test R@10: Carleson',
    'merged_with_new_LeanAPAP': 'task5 Test R@10: LeanAPAP',
    'merged_with_new_hairy-ball-theorem-lean': 'task6 Test R@10: Hairy Ball Theorem',
    'merged_with_new_coxeter': 'task7 Test R@10: Coxeter',
    'merged_with_new_lean4-pdl': 'task8 Test R@10: Lean4 PDL'
}

# Regular expression to match R@10 lines
r_at_10_pattern = re.compile(r'R@10 = ([\d\.]+) %')

# Function to parse the text file
def parse_experiment_results(file_path):
    current_task = None

    with open(file_path, 'r') as file:
        for line in file:
            # Check if the line is an "Intermediate results" line and skip "Average" or final result lines
            if line.startswith('Intermediate results') and 'Average' not in line:
                # Check if the line contains any repository from our mapping
                for repo, task in repo_to_task_mapping.items():
                    if repo in line:
                        current_task = task
                        break
            # Once a valid task is found, capture the R@10 value from the next lines
            elif current_task:
                match = r_at_10_pattern.search(line)
                if match:
                    r_at_10_value = float(match.group(1))
                    task_results_exp[current_task].append(r_at_10_value)
                    current_task = None  # Reset current task to avoid capturing unrelated lines

# Example usage
# 1
# file_path = 'total_evaluation_results_PT_single_repo_no_ewc_no_auto.txt'  # Path to the text file containing results
# 3
# file_path = 'total_evaluation_results_PT_single_repo_no_ewc_curriculum.txt'  # Path to the text file containing results
# 7
# file_path = 'total_evaluation_results_PT_single_repo_ewc.txt'  # Path to the text file containing results
# 8
# file_path = 'total_evaluation_results_PT_single_repo_ewc_curriculum.txt'  # Path to the text file containing results

# 2
# file_path = 'total_evaluation_results_PT_merge_all_no_ewc.txt'  # Path to the text file containing results
# 4
# file_path = 'total_evaluation_results_PT_merge_all_no_ewc_curriculum.txt'  # Path to the text file containing results
# 9
# file_path = 'total_evaluation_results_PT_merge_all_ewc.txt'  # Path to the text file containing results
# 10
# file_path = 'total_evaluation_results_PT_merge_all_ewc_curriculum.txt'  # Path to the text file containing results

# 3 more
# file_path = "total_evaluation_results_PT_single_repo_no_ewc_curriculum_sorries.txt"
# 8 more
file_path = "total_evaluation_results_PT_single_repo_ewc_curriculum_sorries.txt"
parse_experiment_results(file_path)

# # Print out the results
# for task, results in task_results_exp.items():
#     print(f"{task}: {results}")

# Print the results in Python dictionary syntax
def print_results_as_python_syntax(results):
    print("# Data for Experiment")
    print("data_exp = {")
    for task, values in results.items():
        # Printing task name
        print(f"    '{task}': [")
        # Printing each value in the list with indentation
        for value in values:
            print(f"        {value},")
        print("    ],")
    print("}")

# Example usage
print_results_as_python_syntax(task_results_exp)

# # Function to compare expected and parsed results
# def compare_results(parsed_results, expected_results):
#     for task, expected_values in expected_results.items():
#         parsed_values = parsed_results.get(task, [])
        
#         # Convert both lists to sets for comparison (order doesn't matter)
#         if set(parsed_values) == set(expected_values):
#             print(f"{task}: Match")
#         else:
#             print(f"{task}: Mismatch")
#             print(f"Parsed: {parsed_values}")
#             print(f"Expected: {expected_values}")

# compare_results(task_results_exp, expected_results_exp3)