import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    """
	threshold 一般通过sklearn.metrics里面的roc_curve得到,具体不赘述,可以参考其他资料。
	:param threshold: array, shape = [n_thresholds]
	"""
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def find_optimal_threshold(label, y_prob):
    """
    Receiver_Operating_Characteristic, ROC
    :param label: (n, )
    :param y_prob: (n, )
    :return: fpr, tpr, roc_auc, optimal_th, optimal_point
    """
    fpr, tpr, thresholds = roc_curve(label, y_prob)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return optimal_th, optimal_point

# labels = [0,1,1,0,1,0,1]
# img_distance = [0.5,0.3,0.9,0.2,0.6,0.3,0]

# optimal_th, optimal_point = ROC(labels, img_distance)
# print(optimal_th)
# print(optimal_point)

# plt.figure(1)
# plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
# plt.plot([0, 1], [0, 1], linestyle="--")
# plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
# plt.text(optimal_point[0], optimal_point[1], f'Threshold:{optimal_th:.2f}')
# plt.title("ROC-AUC")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.legend()
# plt.show()
