def error_analysis(confusion_matrix):

    TP = confusion_matrix[0]
    TN = confusion_matrix[1]
    FP = confusion_matrix[2]
    FN = confusion_matrix[3]

    total = sum(confusion_matrix)

    percision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    return f"""
        True Positives: {TP}
        True Negatives: {TN}
        False Positives: {FP}
        False Negatives: {FN}
        Accuracy: {(TP + TN) / total}
        Percision: {percision}
        Recall: {recall}
        F1_score: {2 * (percision * recall) / (percision + recall)}
    """

def confusion_matrix(true_labels, predictions):
    if (len(true_labels) != len(predictions)):
        raise TypeError("Lists must match")
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(predictions)):
        if predictions[i] == 1 and predictions[i] == true_labels[i]:
            TP += 1
        elif predictions[i] == 0 and predictions[i] == true_labels[i]:
            TN += 1
        elif predictions[i] == 1 and predictions[i] != true_labels[i]:
            FP += 1
        elif predictions[i] == 0 and predictions[i] != true_labels[i]:
            FN += 1
    return (TP, TN, FP, FN)

import random

true_labels = [random.randint(0, 1) for _ in range(10)]
predictions = [random.randint(0, 1) for _ in range(10)]

print(error_analysis(confusion_matrix(true_labels, predictions)))
# True Positives: 2
# True Negatives: 1
# False Positives: 3
# False Negatives: 4
# Accuracy: 0.3
# Percision: 0.4
# Recall: 0.3333333333333333
# F1_score: 0.3636363636363636