'''
Prediction Combiner
'''

# Imports
from Dataset import *

# Main Functions
def CombineSubmissions(paths, outpath):
    '''
    Combines the given submission files into one submission file.
    '''
    curInd = 0
    testCounts = [100, 100, 214, 214, 214, 214]
    preds = []
    for i in range(len(paths)):
        curPreds = ReadDataset(paths[i])["Predicted"][curInd:curInd + testCounts[i]]
        curInd = curInd + testCounts[i]
        preds.append(curPreds)
    preds = np.concatenate(preds)

    out_df = pd.DataFrame({"Id": np.arange(0, len(preds)), "Predicted": preds})
    SaveDataset(out_df, outpath)

# Driver Code
# Params
CO_1_path = "Submissions/AdaBoost/submission_AdaBoost_TNorm_ExtraTree_200Est_1MD_0.1LR_Seed3.csv"
CO_2_path = "Submissions/AdaBoost/submission_AdaBoost_RandomForest(5,0)_TNorm_200Est_1MD_0.2LR_Seed3.csv"
CO_3_path = "Submissions/AdaBoost/submission_AdaBoost_TNorm_ExtraTree_200Est_1MD_0.1LR_Seed3.csv"
CO_4_path = "Submissions/AdaBoost/submission_AdaBoost_RandomForest(5,0)_TNorm_200Est_1MD_0.2LR_Seed3.csv"
CO_5_path = "Submissions/Combined/submission_2_LR_13456_AB.csv"
CO_6_path = "Submissions/Combined/submission_SScale_1234_LR_1_56_AB_0.25_30.csv "

outpath = "Submissions\Combined\submission_BestMatchXGB.csv"
# Params

# RunCode
CombineSubmissions([CO_1_path, CO_2_path, CO_3_path, CO_4_path, CO_5_path, CO_6_path], outpath)