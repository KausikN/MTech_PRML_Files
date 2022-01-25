'''
Check Similarities Between Submissions
'''

# Imports
from Dataset import *

# Main Functions


# Driver Code
# Params
# preds_1 = "Runs/GradientBoost/submission_1.csv"
# preds_1 = "Submissions/AdaBoost/submission_best.csv"
preds_1 = "Submissions/Final/submission_Final_2.csv"
preds_dir = "Submissions/AdaBoost/"
# Params

# RunCode
print("MAIN:", preds_1)
print()
best_subDiffs = []
best_subDiffs_paths = []
for pred_path in os.listdir(preds_dir):
    preds_2 = preds_dir + pred_path
    if preds_2 == preds_1:
        continue
    print("WITH:", preds_2)
    
    subDiffs, mainDiff = CheckSubmissions_DifferenceSeparate(preds_1, preds_2)
    if len(best_subDiffs) == 0:
        best_subDiffs = subDiffs
        best_subDiffs_paths = [preds_2] * len(subDiffs)
    else:
        best_subDiffs = [min(best_subDiffs[i], subDiffs[i]) for i in range(len(best_subDiffs))]
        best_subDiffs_paths = [preds_2 if best_subDiffs[i] == subDiffs[i] else best_subDiffs_paths[i] for i in range(len(best_subDiffs_paths))]

    print()

print("BEST MATCHES")
for i in range(len(best_subDiffs)):
    print(best_subDiffs_paths[i], best_subDiffs[i])
print()

# AdaBoost - Extra Tree - Train Norm - 200 Est, 1 MD, 0.1 LR, Seed 3
# AdaBoost - V30