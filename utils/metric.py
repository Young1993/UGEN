import numpy as np
from collections import Counter


class Evaluator(object):

    @staticmethod
    # 现在用的
    def intent_acc(pred_intent, real_intent):
        total_count, correct_count = 0.0, 0.0
        for p_intent, r_intent in zip(pred_intent, real_intent):
            if p_intent == r_intent:
                # if sorted(p_intent) == sorted(r_intent):
                correct_count += 1.0
            total_count += 1.0

        return 1.0 * correct_count / total_count

    @staticmethod
    # 现在用的
    def overall_acc(pred_slot, real_slot, pred_intent, real_intent):
        """
        Compute the accuracy based on the whole predictions of
        given sentence, including slot and intent.
        """
        total_count, correct_count = 0.0, 0.0
        for p_slot, r_slot, p_intent, r_intent in zip(pred_slot, real_slot, pred_intent, real_intent):

            if sorted(p_slot) == sorted(r_slot) and p_intent == r_intent:
                correct_count += 1.0
            total_count += 1.0

        return 1.0 * correct_count / total_count

    @staticmethod
    def accuracy(pred_list, real_list):
        """
        Get accuracy measured by predictions and ground-trues.
        """

        pred_array = np.array(list(Evaluator.expand_list(pred_list)))
        real_array = np.array(list(Evaluator.expand_list(real_list)))
        return (pred_array == real_array).sum() * 1.0 / len(pred_array)

    @staticmethod
    def f1_score_intents(pred_array, real_array):
        pred_array = pred_array.transpose()
        real_array = real_array.transpose()
        P, R, F1 = 0, 0, 0
        for i in range(pred_array.shape[0]):
            TP, FP, FN = 0, 0, 0
            for j in range(pred_array.shape[1]):
                if (pred_array[i][j] + real_array[i][j]) == 2:
                    TP += 1
                elif real_array[i][j] == 1 and pred_array[i][j] == 0:
                    FN += 1
                elif pred_array[i][j] == 1 and real_array[i][j] == 0:
                    FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 += 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
            P += precision
            R += recall
        P /= pred_array.shape[0]
        R /= pred_array.shape[0]
        F1 /= pred_array.shape[0]
        return F1

    @staticmethod
    # 现在用的
    def f1_intent_score(pred, gold):
        tp, fp, fn = 0.0, 0.0, 0.0
        for i in range(len(pred)):
            _pred = pred[i]
            _gold = gold[i]
            _tp = 0
            for _p in _pred:
                if _p in _gold:
                    _tp += 1
                else:
                    fp += 1
            fn += len(_gold) - _tp
            tp += _tp

        p = tp / (tp + fp) if tp + fp != 0 else 0
        r = tp / (tp + fn) if tp + fn != 0 else 0
        return p, r, 2 * p * r / (p + r) if p + r != 0 else 0

    @staticmethod
    # 现在用的
    def f1_slot_score(pred_list, real_list):
        tp, fp, fn = 0.0, 0.0, 0.0
        for i in range(len(pred_list)):
            preds = pred_list[i]
            target = real_list[i]

            tp_ = 0
            for pred in preds:
                if pred in target:
                    tp_ += 1
                else:
                    fp += 1

            fn += len(target) - tp_
            tp += tp_

        p = tp / (tp + fp) if tp + fp != 0 else 0
        r = tp / (tp + fn) if tp + fn != 0 else 0
        return p, r, 2 * p * r / (p + r) if p + r != 0 else 0

    @staticmethod
    def f1_score(pred_list, real_list):
        """
        Get F1 score measured by predictions and ground-trues.
        """

        tp, fp, fn = 0.0, 0.0, 0.0
        for i in range(len(pred_list)):
            seg = set()
            result = [elem.strip() for elem in pred_list[i]]
            target = [elem.strip() for elem in real_list[i]]

            j = 0
            while j < len(target):
                cur = target[j]
                if cur[0] == 'B':
                    k = j + 1
                    while k < len(target):
                        str_ = target[k]
                        if not (str_[0] == 'I' and cur[1:] == str_[1:]):
                            break
                        k = k + 1
                    seg.add((cur, j, k - 1))  # B-slot_name, start_id, end_id
                    j = k - 1
                j = j + 1

            tp_ = 0
            j = 0
            while j < len(result):
                cur = result[j]
                if cur[0] == 'B':
                    k = j + 1
                    while k < len(result):
                        str_ = result[k]
                        if not (str_[0] == 'I' and cur[1:] == str_[1:]):
                            break
                        k = k + 1
                    if (cur, j, k - 1) in seg:
                        tp_ += 1
                    else:
                        fp += 1
                    j = k - 1
                j = j + 1

            fn += len(seg) - tp_
            tp += tp_

        p = tp / (tp + fp) if tp + fp != 0 else 0
        r = tp / (tp + fn) if tp + fn != 0 else 0
        return 2 * p * r / (p + r) if p + r != 0 else 0

    """
    Max frequency prediction. 
    """

    @staticmethod
    def max_freq_predict(sample):
        predict = []
        for items in sample:
            predict.append(Counter(items).most_common(1)[0][0])
        return predict

    @staticmethod
    def exp_decay_predict(sample, decay_rate=0.8):
        predict = []
        for items in sample:
            item_dict = {}
            curr_weight = 1.0
            for item in items[::-1]:
                item_dict[item] = item_dict.get(item, 0) + curr_weight
                curr_weight *= decay_rate
            predict.append(sorted(item_dict.items(), key=lambda x_: x_[1])[-1][0])
        return predict

    @staticmethod
    def expand_list(nested_list):
        for item in nested_list:
            if isinstance(item, (list, tuple)):
                for sub_item in Evaluator.expand_list(item):
                    yield sub_item
            else:
                yield item

    @staticmethod
    def nested_list(items, seq_lens):
        num_items = len(items)
        trans_items = [[] for _ in range(0, num_items)]

        count = 0
        for jdx in range(0, len(seq_lens)):
            for idx in range(0, num_items):
                trans_items[idx].append(items[idx][count:count + seq_lens[jdx]])
            count += seq_lens[jdx]

        return trans_items
