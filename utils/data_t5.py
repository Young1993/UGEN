# import json
import os, pickle, sys, copy
# import numpy as np
from typing import List, Dict
from collections import Counter
from tqdm import tqdm
import random


class Data:
    def __init__(self, intent_list: Dict, slot_list: Dict, config, test: None):
        self.train_loader = []
        self.valid_loader = []
        self.test_loader = []
        self.intent_list = intent_list
        self.slot_list = slot_list
        self.config = config
        self.k_sample = config['k_sample']
        self.max_k_sample = self.k_sample * config['step']  # slot中最多能多多少个
        self.question_num = config['question_num']
        self.test = test
        self.count_slot = Counter()  # 统计slot对应的值
        for o in self.slot_list.values():
            self.count_slot[o] = 0
        ''' 不做样本取舍 '''
        self.full_data = config['full_data']

    def show_data_summary(self):
        print("DATA SUMMARY START:")
        # self.question_num = 5
        print("     Train  Instance Number: %s" % (len(self.train_loader) // self.question_num))
        print("     Valid  Instance Number: %s" % (len(self.valid_loader)))
        print("     Test   Instance Number: %s" % (len(self.test_loader)))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def generate_instance(self, args, tokenizer):
        self.tokenizer = tokenizer
        if "train_file" in args and not self.test:
            text, slot, intent = self.__read_file(args.train_file, args.debug, 'train')
            self.train_loader = self.data_process(text, slot, intent, 'train')
        if "valid_file" in args and not self.test:
            text, slot, intent = self.__read_file(args.valid_file, args.debug)
            self.valid_loader = self.data_process(text, slot, intent, 'dev')
        if "test_file" in args:
            text, slot, intent = self.__read_file(args.test_file, args.debug)
            self.test_loader = self.data_process(text, slot, intent, 'test')

    def supplement_data(self, key_slot, val_slot, samples, samples_ids, text, slot, intent, slot_list_asc):
        return_sample = False
        for idx, (t, s, i) in tqdm(enumerate(zip(text, slot, intent))):
            if idx in samples_ids:
                continue
            ori_utterance = "sentence: " + " ".join(t)
            _slot_len = len(s)  # [a, b, c]
            _slot_type_arr = []
            _slot_value = []
            _slot_pairs = []  # (slot value, slot name)
            _gold_intent = sorted([self.intent_list[o] for o in i])
            _i = 0
            _gold_slot = False
            while _i < _slot_len:
                if s[_i] != "O":
                    # record slot_type, start_id, end_id
                    _slot_type = self.slot_list[s[_i][2:]]
                    _start_id = _i
                    _j = _i + 1  # support single slot word: song B-music_item
                    while _j < _slot_len and s[_j][0] != 'B' and s[_j][2:] == s[_i][2:]:
                        _j += 1
                    _end_id = _j - 1
                    _i = _j
                    # 如果是需要补充的数据，则不顾一切补充；超过了k，就不补充数据了
                    if _slot_type == key_slot:
                        val_slot += 1
                        print(key_slot, val_slot)
                        if val_slot <= self.max_k_sample:
                            _gold_slot = True

                    _slot_type_arr.append(_slot_type)
                    _tmp_val = ' '.join(t[_start_id:_end_id + 1]).strip()
                    _slot_value.append(_tmp_val)
                    _slot_pairs.append((_tmp_val, _slot_type))
                    self.count_slot[_slot_type] += 1

                    if not _gold_slot and (self.count_slot[_slot_type] >= self.max_k_sample or min(
                            self.count_slot.values()) > self.k_sample):
                        return_sample = True
                        for o in _slot_type_arr:
                            self.count_slot[o] -= 1
                        break
                else:
                    _i += 1

            if return_sample:
                return samples

            samples_ids.append(idx)

            ''' generate Question '''
            question_1 = ori_utterance + ' </s>' + "question: what are the intents of the sentence?" + \
                         "</s> options: " + ",".join(self.intent_list.values())
            answer_1 = ",".join(_gold_intent)

            question_4 = ori_utterance + ' </s>' + "question: which words are the slot_values in the sentence? List them with related slot names." + \
                         " </s>"
            answer_4 = []
            for _pairs in _slot_pairs:
                answer_4.append(f"{_pairs[0]} is one {_pairs[1]}")
            answer_4 = ",".join(answer_4)
            inputs_1 = self._tokenizer(question_1, self.config['MAX_LEN_Q'])
            labels_1 = self._tokenizer(answer_1, self.config['MAX_LEN_A'])

            inputs_4 = self._tokenizer(question_4, self.config['MAX_LEN_Q'])
            labels_4 = self._tokenizer(answer_4, self.config['MAX_LEN_A'])

            samples.append({"inputs": inputs_1, "labels": labels_1})
            samples.append({"inputs": inputs_4, "labels": labels_4})

        return samples

    def data_process(self, text: List, slot: List, intent: List, name: str):
        slot_width = 0
        _c = Counter()
        samples = []
        slot_list_asc = sorted(self.slot_list.values())
        samples_ids = []
        for idx, (t, s, i) in tqdm(enumerate(zip(text, slot, intent))):
            if len(t) < 5:
                print(f"length is too small: {t}")

            ori_utterance = "sentence: " + " ".join(t)
            _slot_len = len(s)  # [a, b, c]
            _slot_type_arr = []
            _slot_value = []
            _slot_pairs = []  # (slot value, slot name)
            _gold_intent = sorted([self.intent_list[o] for o in i])
            _i = 0
            skip_sample = False
            while _i < _slot_len:
                if s[_i] != "O":
                    # record slot_type, start_id, end_id
                    _slot_type = self.slot_list[s[_i][2:]]
                    _start_id = _i
                    _j = _i + 1  # support single slot word: song B-music_item
                    while _j < _slot_len and s[_j][0] != 'B' and s[_j][2:] == s[_i][2:]:
                        _j += 1
                    _end_id = _j - 1
                    slot_width = max(slot_width, _end_id - _start_id)
                    if _end_id - _start_id > 10:
                        # print(f"slot width: {slot_width}, context: {s[_start_id:_end_id]}")
                        print(f'slot value: {t[_start_id:_end_id + 1]}')
                    _i = _j
                    _slot_type_arr.append(_slot_type)
                    _tmp_val = ' '.join(t[_start_id:_end_id + 1]).strip()
                    _slot_value.append(_tmp_val)
                    _slot_pairs.append((_tmp_val, _slot_type))  # todo: slot 是否需要排序
                    self.count_slot[_slot_type] += 1
                    if name == 'train' and not self.full_data and self.count_slot[_slot_type] > self.k_sample:
                        for o in _slot_type_arr:
                            self.count_slot[o] -= 1
                        skip_sample = True
                        break
                else:
                    _i += 1

            if name == 'train' and skip_sample:
                continue

            if name == 'train' and not self.full_data and min(self.count_slot.values()) > self.k_sample:
                ''' if the sample large than self.k_sample, over! '''
                break

            samples_ids.append(idx)
            ''' generate question '''
            ''' 
            question: what are the intents of the sentence 
            options: play music, rate book, search creative work, search screening event, etc.
            '''
            question_1 = ori_utterance + " </s> question: what are the intents of the sentence?" + "</s> options: " + ",".join(
                self.intent_list.values())
            answer_1 = ",".join(_gold_intent)

            ''' 
            question: which words are the slot_values in the sentence? List them with related slot names.
            options: artist, object name, rating unit, sort, etc.
            '''
            question_4 = ori_utterance + ' </s>' + "question: which words are the slot_values in the sentence? List them with related slot names." + \
                         " </s>"
            answer_4 = []
            for _pairs in _slot_pairs:
                answer_4.append(f"{_pairs[0]} is one {_pairs[1]}")
            answer_4 = ",".join(answer_4)

            _c["q1_len"] = max(_c["q1_len"], len(self.tokenizer.tokenize(question_1)))
            inputs_1 = self._tokenizer(question_1, self.config['MAX_LEN_Q'])
            _c["a1_len"] = max(_c["a1_len"], len(self.tokenizer.tokenize(answer_1)))
            labels_1 = self._tokenizer(answer_1, self.config['MAX_LEN_A'])

            _c["q4_len"] = max(_c["q4_len"], len(self.tokenizer.tokenize(question_4)))
            inputs_4 = self._tokenizer(question_4, self.config['MAX_LEN_Q'])
            _c["a4_len"] = max(_c["a4_len"], len(self.tokenizer.tokenize(answer_4)))
            labels_4 = self._tokenizer(answer_4, self.config['MAX_LEN_A'])

            if name == 'train':
                samples.append({"inputs": inputs_1, "labels": labels_1})
                samples.append({"inputs": inputs_4, "labels": labels_4})
            else:
                samples.append({
                    "intent": {"inputs": inputs_1, "labels": labels_1, "gold": _gold_intent},
                    "slot": {"inputs": inputs_4, "labels": labels_4, "gold": _slot_pairs}
                })

        if name == 'train' and not self.full_data:
            for k, v in self.count_slot.items():
                if v < self.k_sample:
                    print(k, v)
                    # todo: supplement data
                    samples = self.supplement_data(k, v, samples, samples_ids, text, slot, intent, slot_list_asc)
            print(f' min count slot: {min(self.count_slot.values())}')
            random.shuffle(samples)
            print(f'train data distribution: {self.count_slot}, len {len(self.count_slot)}')
            print(f"train sample numbers: {len(samples) // self.question_num}")
        else:
            print(f"sample numbers: {len(samples)}")

        print(f"question statistics: {_c}")
        print(f"max slot width: {slot_width}")

        return samples

    def _tokenizer(self, context, max_len):
        return self.tokenizer.encode_plus(context, max_length=max_len, padding="max_length",
                                          truncation=True, return_tensors='pt', add_special_tokens=False)

    @staticmethod
    def __read_file(file_path: str, debug: bool = False, is_train=None):
        """ Read data file of given path.
        :param file_path: path of data file.
        :return: list of sentence, list of slot and list of intent.
        """

        texts, slots, intents = [], [], []
        text, slot = [], []
        count_num = 0

        with open(file_path, 'r', encoding="utf8") as fr:
            for line in fr.readlines():
                items = line.strip().split()

                if len(items) == 1:
                    texts.append(text)
                    slots.append(slot)
                    if "#" not in items[0]:  # single
                        intents.append(items)
                    else:
                        new = items[0].split("#")
                        intents.append(new)
                        count_num += 1

                        # if count_num > 18000:
                        #     break

                        if debug and count_num > 24:
                            break

                    # clear buffer lists.
                    text, slot = [], []

                elif len(items) == 2:
                    text.append(items[0].strip())
                    slot.append(items[1].strip())

        return texts, slots, intents


# 处理数据
def build_data(args, tokenizer, intent_list: Dict, slot_list: Dict, config, test: str = None, ):
    if args.debug:
        # 调试使用
        file = args.cache_data_directory + args.dataset_name + '_' + args.mode + "_debug.pkl"
    elif test == 'test':
        # 测试使用
        file = args.cache_data_directory + args.dataset_name + '_' + args.mode + "_test.pkl"
    else:
        # 训练使用
        file = args.cache_data_directory + args.dataset_name + '_' + args.mode + "_train.pkl"

    if os.path.exists(file) and not args.refresh:
        # from cache
        print('data is from cache...')
        data = load_data_setting(args)
    else:
        data = Data(intent_list, slot_list, config, test)
        data.generate_instance(args, tokenizer)
        save_data_setting(data, args, file)
    return data


def save_data_setting(data, args, file):
    new_data = copy.deepcopy(data)
    data.show_data_summary()
    if not os.path.exists(args.cache_data_directory):
        os.makedirs(args.cache_data_directory)
    with open(file, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting is saved to file: ", file)


def get_index(lst=None, item=''):
    return [index for (index, value) in enumerate(lst) if value == item]


def load_data_setting(args):
    if args.debug:
        saved_path = args.cache_data_directory + args.dataset_name + '_' + args.mode + "_debug.pkl"
    else:
        saved_path = args.cache_data_directory + args.dataset_name + '_' + args.mode + "_train.pkl"

    with open(saved_path, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting is loaded from file: ", saved_path)

    if args.debug:
        data.test_loader = data.test_loader[:15]
        data.valid_loader = data.valid_loader[:15]
        data.train_loader = data.train_loader[:15]

    ratio = int(len(data.train_loader) * args.data_ratio)
    data.train_loader = data.train_loader[:ratio]
    data.show_data_summary()
    return data
