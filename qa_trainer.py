import json
import torch, random, gc
from torch import nn, optim
from tqdm import tqdm
from transformers import AdamW
from utils.average_meter import AverageMeter, generate_triple
from typing import List, Dict
# from sklearn.metrics import f1_score, accuracy_score
from utils.metric import Evaluator
import os
# import re
import time


def eval_batchify(batch):
    intent_inputs, intent_attn_mask, intent_labels = [], [], []
    slot_inputs, slot_attn_mask, slot_labels = [], [], []
    gold_intent_list, gold_slot_list = [], []

    for b in batch:
        intent_inputs.append(b['intent']["inputs"]['input_ids'])
        intent_attn_mask.append(b['intent']["inputs"]['attention_mask'])
        intent_labels.append(b["intent"]['labels']['input_ids'])

        slot_inputs.append(b['slot']["inputs"]['input_ids'])
        slot_attn_mask.append(b['slot']["inputs"]['attention_mask'])
        slot_labels.append(b["slot"]['labels']['input_ids'])

        gold_intent_list.append(b["intent"]['gold'])
        gold_slot_list.append(b["slot"]['gold'])

    intent_inputs = torch.cat(intent_inputs)
    intent_attn_mask = torch.cat(intent_attn_mask)
    intent_labels = torch.cat(intent_labels)

    slot_inputs = torch.cat(slot_inputs)
    slot_attn_mask = torch.cat(slot_attn_mask)
    slot_labels = torch.cat(slot_labels)

    assert intent_inputs.shape == slot_inputs.shape

    intent = {
        "intent_inputs": intent_inputs,
        "intent_attn_mask": intent_attn_mask,
        "intent_labels": intent_labels
    }

    slot = {
        "slot_inputs": slot_inputs,
        "slot_attn_mask": slot_attn_mask,
        "slot_labels": slot_labels
    }

    return intent, slot, gold_intent_list, gold_slot_list


# gold intent: [play music, rate book]
# gold slot: (slot value, slot name)
def train_batchify(batch):
    inputs, attention_mask, labels = [], [], []
    # gold_intent_list, gold_slot_list = [], []

    for b in batch:
        inputs.append(b['inputs']["input_ids"])
        attention_mask.append(b['inputs']['attention_mask'])
        labels.append(b["labels"]['input_ids'])

        # gold_intent_list.append(b["gold_intent"])
        # gold_slot_list.append(b["gold_slot"])

    inputs = torch.cat(inputs)
    attention_mask = torch.cat(attention_mask)
    labels = torch.cat(labels)

    return inputs, attention_mask, labels  # , gold_intent_list, gold_slot_list


class QaTrainer(nn.Module):
    def __init__(self, model, data, args, device_id, tokenizer, base_config=None, intent_list=None,
                 slot_list: List = None):
        super().__init__()
        self.args = args
        self.model = model
        self.data = data
        self.device_id = device_id
        self.tokenizer = tokenizer
        self.intent_list = intent_list
        self.slot_list = slot_list
        self.base_config = base_config
        self.start_epoch = 0  # 默认的开始epoch
        self.slot_type_num = self.base_config['slot_type']
        self.start_metric_epoch = int(0.02 * self.args.max_epoch)  # 只有超过一定epoch时才保存
        self.slot_input_maxlength = self.base_config['slot_input_maxlength']
        self.intent_label_length = self.base_config['intent_label_length']
        self.answer_len = self.base_config['MAX_LEN_A']

        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == 'AdamW':
            self.optimizer = AdamW(model.parameters())
        else:
            raise Exception("Invalid optimizer.")
        if args.use_gpu:
            self.cuda()

    def train_model(self, resume: bool = False, saved_path: str = ''):
        best_score = 0
        # bt_loss = float('inf')
        train_loader = self.data.train_loader
        train_num = len(train_loader)
        batch_size = self.args.batch_size
        total_batch = train_num // batch_size + 1
        for epoch in tqdm(range(self.start_epoch, self.args.max_epoch), desc="train"):
            # Train
            self.model.train()
            self.model.zero_grad()
            self.optimizer = self.lr_decay(self.optimizer, epoch, self.args.lr_decay)
            print("\n=== Epoch %d train ===" % epoch, flush=True)
            avg_loss = AverageMeter()
            random.shuffle(train_loader)

            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num
                train_instance = train_loader[start:end]
                if not train_instance:
                    continue
                # data to device
                input_ids, att_mask, tgt = train_batchify(train_instance)

                logits = self.model(input_ids=input_ids.cuda(device=self.device_id),
                                    attention_mask=att_mask.cuda(device=self.device_id),
                                    labels=tgt.cuda(device=self.device_id))

                loss = logits.loss
                avg_loss.update(loss.item(), 1)
                loss.backward()

                if self.args.max_grad_norm != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                if (batch_id + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.model.zero_grad()

                if batch_id % self.args.print_steps == 0 and batch_id != 0:
                    print("\n Instance: %d; loss: %.4f" % (start, avg_loss.avg), flush=True)

            gc.collect()
            torch.cuda.empty_cache()

            # validation
            print("=== Epoch %d Validation ===" % epoch)
            valid_loss, valid_score = self.eval_model(self.data.valid_loader, epoch)  # result
            print("valid loss: %.4f" % valid_loss, flush=True)

            if epoch > self.start_metric_epoch:
                if best_score < valid_score:
                    # bt_loss = valid_loss
                    best_score = valid_score
                    best_result_epoch = epoch

                    if not os.path.exists(self.args.generated_param_directory):
                        os.makedirs(self.args.generated_param_directory)

                    ### model name need to change
                    print('=' * 50)
                    print('saved best weighted epoch:{}'.format(epoch))
                    # print('best loss:{}'.format(bt_loss))
                    print('best score:{}'.format(best_score))

                    _model_name = ''
                    if resume:
                        _model_name = saved_path
                    else:
                        _model_name = os.path.join(self.args.generated_param_directory, self.args.model_name + '.pt')

                    torch.save({'epoch': best_result_epoch,
                                'optimizer': self.optimizer.state_dict(),  # saved optimizer
                                'state_dict': self.model.state_dict()},
                               _model_name)
            gc.collect()
            torch.cuda.empty_cache()

        # Test
        print("======== Test ===========")
        test_loss, score = self.eval_model(self.data.test_loader, -1)
        print("\n test loss: %.4f, score: %.4f" % (test_loss, score), flush=True)
        print('finished!')

    def eval_model(self, eval_loader, epoch: int):
        self.model.eval()
        avg_loss = AverageMeter()
        pred_slot_pair, gold_slot_pair = [], []
        self.pred_intent_list, self.gold_intent_list = [], []

        with torch.no_grad():
            batch_size = self.args.batch_size
            eval_num = len(eval_loader)
            total_batch = eval_num // batch_size + 1

            for batch_id in tqdm(range(total_batch), desc="evaluation"):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > eval_num:
                    end = eval_num
                eval_instance = eval_loader[start:end]

                if not eval_instance:
                    continue

                _intent, _slot, gold_intents_list, gold_slot_list = eval_batchify(eval_instance)

                logits = self.model(input_ids=_intent['intent_inputs'].cuda(device=self.device_id),
                                    attention_mask=_intent['intent_attn_mask'].cuda(device=self.device_id),
                                    labels=_intent['intent_labels'].cuda(device=self.device_id))

                loss = logits.loss
                avg_loss.update(loss.item(), 1)

                slot_logits = self.model(input_ids=_slot['slot_inputs'].cuda(device=self.device_id),
                                         attention_mask=_slot['slot_attn_mask'].cuda(device=self.device_id),
                                         labels=_slot['slot_labels'].cuda(device=self.device_id))

                slot_loss = slot_logits.loss
                avg_loss.update(slot_loss.item(), 1)

                if batch_id % self.args.print_steps == 0 and batch_id != 0:
                    print("\n Instance: %d; loss: %.4f" % (start, avg_loss.avg), flush=True)

                if epoch > self.start_metric_epoch or epoch < 0:
                    # todo: 并行预测结果
                    # print(f'{batch_id} inference...')
                    intent_preds = self.model.generate(_intent['intent_inputs'].cuda(device=self.device_id),
                                                       num_beams=3, max_length=self.answer_len,
                                                       early_stopping=True)

                    _intent_pred_list = [self.tokenizer.decode(g, skip_special_tokens=True) for g in intent_preds]
                    # intent handle
                    _intent_pred_list = [self.intent_handle(o.split(',')) for o in _intent_pred_list]
                    self.pred_intent_list += _intent_pred_list
                    self.gold_intent_list += gold_intents_list
                    assert len(self.pred_intent_list) == len(self.gold_intent_list)

                    # print('slot inference...')
                    slot_preds = self.model.generate(_slot['slot_inputs'].cuda(device=self.device_id),
                                                     # _slot['slot_attn_mask'].to(self.device),
                                                     num_beams=3, max_length=self.answer_len,
                                                     early_stopping=True)

                    _slot_pred_list = [self.tokenizer.decode(g, skip_special_tokens=True) for g in slot_preds]
                    # post-process
                    _slot_pred_list = [self.post_process(o.split(',')) for o in _slot_pred_list]
                    pred_slot_pair += _slot_pred_list
                    gold_slot_pair += gold_slot_list
                    assert len(pred_slot_pair) == len(gold_slot_pair)

        if epoch > self.start_metric_epoch or epoch < 0:
            assert len(self.pred_intent_list) == len(self.gold_intent_list)
            assert len(pred_slot_pair) == len(gold_slot_pair)

            intent_acc = Evaluator.intent_acc(self.pred_intent_list,
                                              self.gold_intent_list)
            print(f"intent acc: {intent_acc}")

            intent_p, intent_r, intent_f1_score = Evaluator.f1_intent_score(self.pred_intent_list,
                                                                            self.gold_intent_list)
            print(f"intent precision: {intent_p}, recall: {intent_r}, F1: {intent_f1_score}")

            slot_p, slot_r, slot_f1_score = Evaluator.f1_slot_score(pred_slot_pair, gold_slot_pair)
            print(f"slot precision: {slot_p}, recall: {slot_r}, F1: {slot_f1_score}")

            acc = Evaluator.overall_acc(pred_slot_pair, gold_slot_pair, self.pred_intent_list,
                                        self.gold_intent_list)
            print(f"overall acc: {acc}")

            score = intent_acc * 0.25 + intent_f1_score * 0.1 + slot_f1_score * 0.25 + acc * 0.4
            # score = acc

            if epoch == -1:
                if not os.path.exists('./output/' + self.args.dataset_name):
                    os.makedirs('./output/' + self.args.dataset_name)

                with open(os.path.join('./output/', self.args.dataset_name,
                                       "eval" + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + ".txt"), 'w',
                          encoding="utf8") as f:

                    for p_slot, g_slot, p_intent, g_intent in zip(pred_slot_pair, gold_slot_pair, self.pred_intent_list,
                                                                  self.gold_intent_list):
                        f.write(f"pred slot: {str(p_slot)} \n")
                        f.write(f"gold slot: {str(g_slot)} \n")
                        f.write(f"pred intent: {str(p_intent)} \n")
                        f.write(f"gold intent: {str(g_intent)} \n")
                        f.write('=' * 100 + '\n')

                    f.close()

            return avg_loss.avg, score

        return avg_loss.avg, None

    def intent_handle(self, intent_cell):
        _intent = []
        for o in intent_cell:
            if o.strip() in self.intent_list:
                _intent.append(o.strip())
        return sorted(_intent)

    def post_process(self, slot_cell):
        _slot = []
        for o in slot_cell:
            try:
                _slot_val, _slot_name = o.split(' is one ')
                if _slot_name.strip() in self.slot_list:
                    _slot.append((_slot_val.strip(), _slot_name.strip()))
            except:
                # if len(o) and o != 'True' and o != 'False':
                    # print(o)
                continue
        return _slot

    def load_state_dict(self, checkpoint, multi_gpu, device_id):
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_epoch(checkpoint['epoch'])
        if multi_gpu:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.device_ids)
            self.model = self.model.cuda(device=device_id)

    def update_epoch(self, epoch):
        self.start_epoch = epoch

    @staticmethod
    def lr_decay(optimizer, epoch, decay_rate):
        # lr = init_lr * ((1 - decay_rate) ** epoch)
        if epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * (1 - decay_rate)
                # print(param_group['lr'])
        return optimizer
