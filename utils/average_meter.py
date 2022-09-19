import torch, collections


class AverageMeter(object):
    """
    Computes and stores the average and current value of metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """
        String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def generate_span(start_logits, end_logits, info, args):
    seq_lens = info["seq_len"]  # including [CLS] and [SEP]
    sent_idxes = info["sent_idx"]
    _Prediction = collections.namedtuple(
        "Prediction", ["start_index", "end_index", "start_prob", "end_prob"]
    )
    output = {}
    start_probs = start_logits.softmax(-1)
    end_probs = end_logits.softmax(-1)
    start_probs = start_probs.cpu().tolist()
    end_probs = end_probs.cpu().tolist()
    for (start_prob, end_prob, seq_len, sent_idx) in zip(start_probs, end_probs, seq_lens, sent_idxes):
        output[sent_idx] = {}
        for triple_id in range(args['num_slots']):
            predictions = []
            start_indexes = _get_best_indexes(start_prob[triple_id], args['n_best_size'])
            end_indexes = _get_best_indexes(end_prob[triple_id], args['n_best_size'])
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the sentence. We throw out all
                    # invalid predictions.
                    if start_index >= (seq_len - 1):  # [SEP]
                        continue
                    if end_index >= (seq_len - 1):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1

                    if length > args['max_span_length']:
                        continue

                    predictions.append(
                        _Prediction(
                            start_index=start_index,
                            end_index=end_index,
                            start_prob=start_prob[triple_id][start_index],
                            end_prob=end_prob[triple_id][end_index],
                        )
                    )
            output[sent_idx][triple_id] = predictions
    return output


def generate_relation(pred_rel_logits, info, args):
    rel_probs, pred_rels = torch.max(pred_rel_logits.softmax(-1), dim=2)
    rel_probs = rel_probs.cpu().tolist()
    pred_rels = pred_rels.cpu().tolist()
    sent_idxes = info["sent_idx"]
    output = {}
    _Prediction = collections.namedtuple(
        "Prediction", ["pred_rel", "rel_prob"]
    )
    for (rel_prob, pred_rel, sent_idx) in zip(rel_probs, pred_rels, sent_idxes):
        output[sent_idx] = {}
        for triple_id in range(args['num_slots']):
            output[sent_idx][triple_id] = _Prediction(
                pred_rel=pred_rel[triple_id],
                rel_prob=rel_prob[triple_id])
    return output


def generate_triple(output, info, args, num_classes):
    _Pred_Triple = collections.namedtuple(
        "Pred_Triple",
        ["pred_rel", "rel_prob", "head_start_index", "head_end_index", "head_start_prob", "head_end_prob"]
    )
    pred_head_ent_dict = generate_span(output["head_start_logits"], output["head_end_logits"], info, args)
    pred_rel_dict = generate_relation(output['pred_rel_logits'], info, args)

    triples = {}
    for sent_idx in pred_rel_dict:
        triples[sent_idx] = []
        for triple_id in range(args['num_slots']):
            pred_rel = pred_rel_dict[sent_idx][triple_id]
            pred_head = pred_head_ent_dict[sent_idx][triple_id]
            triple = generate_strategy(pred_rel, pred_head, num_classes, _Pred_Triple)
            if triple:
                triples[sent_idx].append(triple)
    return triples


def generate_strategy(pred_rel, pred_head, num_classes, _Pred_Triple):
    if pred_rel.pred_rel != num_classes:
        if pred_head:
            for ele in pred_head:
                # if ele.start_prob > 0.5 and ele.end_prob > 0.5:
                    break
            head = ele
            return _Pred_Triple(pred_rel=pred_rel.pred_rel, rel_prob=pred_rel.rel_prob,
                                head_start_index=head.start_index, head_end_index=head.end_index,
                                head_start_prob=head.start_prob, head_end_prob=head.end_prob)
        else:
            return
    else:
        return
