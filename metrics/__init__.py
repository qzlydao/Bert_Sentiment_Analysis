import numpy as np
from sklearn.metrics import f1_score

def find_best_threshold(all_predictions, all_labels):
    '''寻找最佳分类边界, 在[0, 1]之间'''

    # 展平所有的预测结果和对应的标记
    all_predictions = np.ravel(all_predictions)
    all_labels = np.ravel(all_labels)

    # 从0到1以0.01为间隔定义99个备选阈值 0.01-0.99
    thresholds = [i / 100 for i in range(100)]
    all_f1s = []

    for threshold in thresholds:
        # 计算当前阈值的 F1 score
        preds = (all_predictions >= threshold).astype('int')
        f1 = f1_score(y_true=all_labels, y_pred=preds)
        all_f1s.append(f1)

    # 找出可以使F1 score最大的阈值
    best_threshold = thresholds[int(np.argmax(np.array(all_f1s)))]
    print('best threshold is {}'.format(str(best_threshold)))
    print(all_f1s)
    return best_threshold

