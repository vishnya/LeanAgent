data_exp3 = {
    'Repository': [
        'Zeta 3 Irrational', 'Formal Book', 'Formalization of Constructable Numbers', 
        'Carleson', 'LeanAPAP', 'Hairy Ball Theorem', 'Coxeter', 'Lean4 PDL'
    ],
    'Validation R@10 Exp3': [62.93, 66.5, 66.97, 63.32, 67.48, 72.19, 69.03, 72.15],
    'Average Test R@10 Exp3': [61.31, 63.76, 64.4, 64.8, 65.41, 66.59, 67.52, 68.02],
    'task1 Test R@10: Zeta 3 Irrational': [
        61.308370662694585,
        62.50873752152146,
        61.90408245057425,
        61.59205704797418,
        61.548903336307234,
        61.37267933820618,
        61.00003369722818,
        61.18449229278185,
    ],
    'task2 Test R@10: Formal Book': [
        65.01205506666258,
        66.31101909088257,
        66.88943612322451,
        66.35970438359516,
        67.2110481479082,
        68.23306083374344,
        68.1019931907304,
    ],
    'task3 Test R@10: Formalization of Constructable Numbers': [
        64.97169355975755,
        65.31698920218435,
        66.77817052595005,
        66.37983593704169,
        67.46729242408834,
        67.44265383809488,
    ],
    'task4 Test R@10: Carleson': [
        65.41771685089931,
        66.80184111379121,
        66.16521223848052,
        67.47689306813022,
        67.97910502896059,
    ],
    'task5 Test R@10: LeanAPAP': [
        65.56497525114236,
        67.14550091629577,
        67.94042845679527,
        67.59639957199384,
    ],
    'task6 Test R@10: Hairy Ball Theorem': [
        71.26504861695271,
        71.5225330616727,
        72.14936019943072,
    ],
    'task7 Test R@10: Coxeter': [
        69.01869231066313,
        69.12042830108524,
    ],
    'task8 Test R@10: Lean4 PDL': [
        70.58528674627745,
    ],
}

data_exp8 = {
    'Repository': [
        'Zeta 3 Irrational', 'Formal Book', 'Formalization of Constructable Numbers', 
        'Carleson', 'LeanAPAP', 'Hairy Ball Theorem', 'Coxeter', 'Lean4 PDL'
    ],
    'Validation R@10 Exp8': [62.45, 67.08, 67.96, 63.21, 67.54, 71.53, 69.23, 73.17],
    'Average Test R@10 Exp8': [61.67, 64.06, 63.86, 65.09, 65.4, 67, 67.4, 68.11],
    'task1 Test R@10: Zeta 3 Irrational': [
        61.667375556564,
        62.463529890391825,
        61.97302267070417,
        62.59627958509581,
        61.643174108040014,
        61.34060301564737,
        61.452791890228845,
        61.13298783281738,
    ],
    'task2 Test R@10: Formal Book': [
        65.65160436832791,
        65.5092897243068,
        66.55313476815184,
        67.1787037384307,
        68.22971435087476,
        68.762049668193,
        69.27217947013169,
    ],
    'task3 Test R@10: Formalization of Constructable Numbers': [
        64.08682913403007,
        65.40529147565512,
        66.18787719397397,
        66.67240746334032,
        66.51127894758307,
        66.79603972414712,
    ],
    'task4 Test R@10: Carleson': [
        65.82295683212712,
        66.43663134261816,
        66.81397437321665,
        66.72930855466825,
        66.57999209049395,
    ],
    'task5 Test R@10: LeanAPAP': [
        65.53367013064525,
        67.47539396078109,
        67.3518898619551,
        67.95632476896512,
    ],
    'task6 Test R@10: Hairy Ball Theorem': [
        71.43228617488137,
        71.48972716673704,
        71.30324110930599,
    ],
    'task7 Test R@10: Coxeter': [
        69.5226413219114,
        69.49149384003398,
    ],
    'task8 Test R@10: Lean4 PDL': [
        72.38648090815273,
    ],
}



experiments = {
    'Exp3': data_exp3,
    'Exp8': data_exp8,
}


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def calculate_expanded_bwt(data):
    N = len(data['Repository'])
    bwt_sum = 0
    count = 0

    for i in range(2, N + 1):
        for j in range(1, i):
            task_i_key = f'task{i} Test R@10: {data["Repository"][i-1]}'
            task_j_key = f'task{j} Test R@10: {data["Repository"][j-1]}'
            
            if task_i_key in data and task_j_key in data:
                R_i_j = data[task_j_key][i-j-1]  # Performance on task j after learning task i
                R_j_j = data[task_j_key][0]      # Initial performance on task j
                
                bwt_sum += R_i_j - R_j_j
                count += 1

    if count > 0:
        return bwt_sum / count
    else:
        return 0  # Return 0 if no valid comparisons were made

def calculate_AA(data, exp_name):
    test_accuracies = data[f'Average Test R@10 {exp_name}']
    return test_accuracies[-1]  # The last value represents AA_k

def calculate_AIA(data, exp_name):
    test_accuracies = data[f'Average Test R@10 {exp_name}']
    AA_values = [np.mean(test_accuracies[:i+1]) for i in range(len(test_accuracies))]
    return np.mean(AA_values)

def calculate_metrics(data, exp_name):
    metrics = {}
    
    # # 2. Area Under the Learning Curve (AULC)
    # aulc = np.trapz(data[f'Average Test R@10 {exp_name}']) / len(data['Repository'])
    # metrics['AULC'] = aulc
    
    # # 5. Backward Transfer
    # backward_transfer = []
    # for i in range(1, len(data['Repository'])):
    #     task_key = f'task{i} Test R@10: {data["Repository"][i-1]}'
    #     if task_key in data:
    #         backward_transfer.append(data[task_key][-1] - data[task_key][0])
    # metrics['Avg_Backward_Transfer'] = np.mean(backward_transfer)
    
    # Extract the task-specific accuracies
    task_accuracies = {}
    for i in range(1, len(data['Repository']) + 1):
        key = f'task{i} Test R@10: {data["Repository"][i-1]}'
        if key in data:
            task_accuracies[i] = np.array(data[key])

    # Calculate min-ACC
    min_acc_values = []
    for k in range(2, len(task_accuracies) + 1):
        min_acc_sum = 0
        count = 0
        for i in range(1, k):
            if i in task_accuracies and len(task_accuracies[i][k-1:]) > 0:
                min_acc_sum += np.min(task_accuracies[i][k-1:])
                count += 1
        if count > 0:
            min_acc_values.append(min_acc_sum / count)
    
    # metrics['min-ACC'] = np.mean(min_acc_values) if min_acc_values else 0

    # Worst-case Accuracy (WC-ACC)
    avg_test = np.array(data[f'Average Test R@10 {exp_name}'])
    tasks = len(avg_test)
    wc_acc_values = []
    for k in range(1, tasks + 1):
        if k == 1:
            wc_acc_values.append(avg_test[0])
        elif k-2 < len(min_acc_values):
            wc_acc = (1/k) * avg_test[k-1] + (1 - 1/k) * min_acc_values[k-2]
            wc_acc_values.append(wc_acc)
    
    # metrics['WC-ACC'] = np.mean(wc_acc_values)

    # 3. Windowed Forgetting (WF)
    def calculate_WF(w):
        WF = 0
        for i in range(len(avg_test)):
            if i >= w:
                WF = max(WF, np.max(avg_test[i-w:i]) - avg_test[i])
        return WF
    
    metrics['WF5'] = calculate_WF(5)

    # 7. Windowed Plasticity (WP)
    window_size = 5
    wp_values = [max(0, avg_test[i] - avg_test[max(0, i-window_size)]) for i in range(len(avg_test))]
    metrics['WP5'] = np.mean(wp_values)

    metrics['Expanded_BWT'] = calculate_expanded_bwt(data)

    return metrics

def calculate_additional_metrics(data, exp_name):
    metrics = {}
    
    # 3. Forgetting Measure (FM)
    fm_values = []
    for i in range(2, len(data['Repository']) + 1):
        task_key = f'task{i-1} Test R@10: {data["Repository"][i-2]}'
        if task_key in data:
            task_performances = [data[f'task{j} Test R@10: {data["Repository"][j-1]}'][i-j-1] for j in range(1, i) if f'task{j} Test R@10: {data["Repository"][j-1]}' in data]
            fm_values.append(np.max(task_performances) - data[task_key][-1])
    metrics['FM'] = np.mean(fm_values)
    
    # 4. Incremental Plasticity (IP)
    ip_values = np.diff(data[f'Validation R@10 {exp_name}'])
    metrics['IP'] = np.mean(ip_values)
    
    # # 10. Time-Weighted Cumulative Performance (TWCP)
    # weights = np.arange(len(data['Repository']), 0, -1)
    # metrics['TWCP'] = np.sum(weights * np.array(data[f'Average Test R@10 {exp_name}'])) / np.sum(weights)
    
    # 12. Catastrophic Forgetting Resilience (CFR)
    metrics['CFR'] = np.min(data[f'Average Test R@10 {exp_name}']) / np.max(data[f'Average Test R@10 {exp_name}'])
    
    return metrics

def calculate_all_metrics(experiments):
    all_metrics = {}
    for exp_name, data in experiments.items():
        metrics = calculate_metrics(data, exp_name)
        additional_metrics = calculate_additional_metrics(data, exp_name)
        all_metrics[exp_name] = {**metrics, **additional_metrics}
    return all_metrics

lower_is_better_metrics = {'WF5', 'FM'}

def compare_metrics(all_metrics):
    comparison = {}
    for metric in all_metrics['Exp3'].keys():
        values = {exp: metrics[metric] for exp, metrics in all_metrics.items()}
        reverse = metric not in lower_is_better_metrics
        sorted_exps = sorted(values, key=values.get, reverse=reverse)
        sorted_values = [values[exp] for exp in sorted_exps]
        
        # Calculate percentage improvement
        if len(sorted_values) > 1:
            if sorted_values[1] == 0:
                if sorted_values[0] == 0:
                    percent_improvement = 0  # Both values are zero, no improvement
                else:
                    percent_improvement = float('inf')  # Non-zero divided by zero, infinite improvement
            else:
                if metric in lower_is_better_metrics:
                    percent_improvement = ((sorted_values[1] - sorted_values[0]) / sorted_values[1]) * 100
                else:
                    percent_improvement = ((sorted_values[0] - sorted_values[1]) / sorted_values[1]) * 100
        else:
            percent_improvement = 0
        
        comparison[metric] = {
            'Ranking': list(zip(sorted_exps, sorted_values)),
            'Improvement': percent_improvement
        }
    return comparison

def format_comparison(comparison):
    formatted_output = ""
    for metric, result in comparison.items():
        formatted_output += f"{metric}:\n"
        for i, (exp, value) in enumerate(result['Ranking']):
            formatted_output += f"  {i+1}. {exp}: {value:.4f}\n"
        if result['Improvement'] == float('inf'):
            formatted_output += "  Best improvement: inf%\n\n"
        elif metric in lower_is_better_metrics:
            formatted_output += f"  Best improvement: -{result['Improvement']:.2f}%\n\n"
        else:
            formatted_output += f"  Best improvement: +{result['Improvement']:.2f}%\n\n"
    return formatted_output

# Calculate metrics for all experiments
all_metrics = calculate_all_metrics(experiments)

# Compare metrics across experiments
comparison = compare_metrics(all_metrics)

# Print the results
print("Metrics Comparison:")
print(format_comparison(comparison))


def calculate_composite_score(metrics):
    # Normalize the metrics
    normalized_metrics = {}
    for metric in ['WF5', 'FM', 'WP5', 'IP', 'Expanded_BWT', 'CFR']:
        values = [m[metric] for m in metrics.values()]
        min_val, max_val = min(values), max(values)
        if max_val - min_val == 0:
            normalized_metrics[metric] = {exp: 1 for exp in metrics}
        else:
            normalized_metrics[metric] = {exp: (metrics[exp][metric] - min_val) / (max_val - min_val) for exp in metrics}

    # Calculate composite score
    composite_scores = {}
    for exp in metrics:
        score = (0.2 * (1 - normalized_metrics['WF5'][exp])) + \
                (0.2 * (1 - normalized_metrics['FM'][exp])) + \
                (0.1 * normalized_metrics['WP5'][exp]) + \
                (0.1 * normalized_metrics['IP'][exp]) + \
                (0.2 * normalized_metrics['Expanded_BWT'][exp]) + \
                (0.2 * normalized_metrics['CFR'][exp])
        composite_scores[exp] = score

    return composite_scores

# After calculating all_metrics
composite_scores = calculate_composite_score(all_metrics)
print("\nComposite Scores:")
for exp, score in sorted(composite_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{exp}: {score:.4f}")

# SINGLE REPO:


# Metrics Comparison:

# WF5:
#   1. Exp3: 0.1800
#   2. Exp8: 0.7300
#   3. Exp7: 7.1700
#   4. Exp1: 7.6000
#   Best improvement: -75.34%

# WP5:
#   1. Exp8: 3.4200
#   2. Exp3: 2.4736
#   3. Exp7: 1.4729
#   4. Exp1: 0.8914
#   Best improvement: +38.26%

# Expanded_BWT:
#   1. Exp3: 1.2086
#   2. Exp7: 1.0397
#   3. Exp8: 0.7563
#   4. Exp1: 0.5124
#   Best improvement: +16.25%

# FM:
#   1. Exp3: 0.8455
#   2. Exp8: 2.1120
#   3. Exp7: 4.0435
#   4. Exp1: 6.5344
#   Best improvement: -59.97%

# IP:
#   1. Exp8: 1.0638
#   2. Exp3: 1.0231
#   3. Exp1: 0.3585
#   4. Exp7: 0.2562
#   Best improvement: +3.98%

# CFR:
#   1. Exp7: 0.8805
#   2. Exp3: 0.8767
#   3. Exp1: 0.8722
#   4. Exp8: 0.8458
#   Best improvement: +0.43%

# Composite Scores:
# Exp3: 0.9357
# Exp8: 0.6107
# Exp7: 0.4736
# Exp1: 0.1649


# MERGE ALL:


# Metrics Comparison:

# WF5:
#   1. Exp4: 2.2300
#   2. Exp9: 13.3400
#   3. Exp2: 15.8300
#   Best improvement: -83.28%

# WP5:
#   1. Exp4: 0.0886
#   2. Exp2: 0.0000
#   3. Exp9: 0.0000
#   Best improvement: inf%

# Expanded_BWT:
#   1. Exp4: 0.7270
#   2. Exp2: -0.1983
#   3. Exp9: -1.3354
#   Best improvement: +-466.63%

# FM:
#   1. Exp4: 4.0622
#   2. Exp2: 10.4955
#   3. Exp9: 11.4362
#   Best improvement: -61.30%

# IP:
#   1. Exp4: -0.6408
#   2. Exp2: -1.4969
#   3. Exp9: -1.7062
#   Best improvement: +-57.19%

# CFR:
#   1. Exp4: 0.9365
#   2. Exp2: 0.7618
#   3. Exp9: 0.7545
#   Best improvement: +22.93%

# Composite Scores:
# Exp4: 1.0000
# Exp2: 0.1635
# Exp9: 0.0366


# Metrics Comparison:
# WF5:
#   1. Exp4: 2.2300
#   2. Exp10: 5.8200
#   3. Exp9: 13.3400
#   4. Exp2: 15.8300
#   Best improvement: -61.68%

# WP5:
#   1. Exp10: 0.1114
#   2. Exp4: 0.0886
#   3. Exp2: 0.0000
#   4. Exp9: 0.0000
#   Best improvement: +25.81%

# Expanded_BWT:
#   1. Exp4: 0.7270
#   2. Exp2: -0.1983
#   3. Exp10: -0.3880
#   4. Exp9: -1.3354
#   Best improvement: +-466.63%

# FM:
#   1. Exp10: 3.8005
#   2. Exp4: 4.0622
#   3. Exp2: 10.4955
#   4. Exp9: 11.4362
#   Best improvement: -6.44%

# IP:
#   1. Exp4: -0.6408
#   2. Exp10: -0.8869
#   3. Exp2: -1.4969
#   4. Exp9: -1.7062
#   Best improvement: +-27.75%

# CFR:
#   1. Exp4: 0.9365
#   2. Exp10: 0.9025
#   3. Exp2: 0.7618
#   4. Exp9: 0.7545
#   Best improvement: +3.77%



# Composite Scores:
# Exp4: 0.9726
# Exp10: 0.7786
# Exp2: 0.1626
# Exp9: 0.0366