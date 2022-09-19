# @Time : 23/08/2022
# @Author : Young

import argparse, os, torch
import random
import numpy as np
from utils.data_qa import build_data
from qa_trainer import QaTrainer

from transformers import AutoTokenizer
from model.idsf import IDSF
import json
from config.base import base_config
import namegenerator
import pathlib


def str2bool(v):
    return v.lower() in ('true')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg


def get_args():
    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    data_arg = add_argument_group('Data')
    data_arg.add_argument('--dataset_name', type=str, default="MixSNIPS")  # dataset name, control cache pkl name;
    data_arg.add_argument('--dataset_dir', type=str, default="MixSNIPS_clean")
    data_arg.add_argument('--mode', type=str, default="qa_5")

    data_arg.add_argument('--train_file', type=str, default="./data/MixSNIPS_clean/train.txt")
    data_arg.add_argument('--valid_file', type=str, default="./data/MixSNIPS_clean/dev.txt")
    data_arg.add_argument('--test_file', type=str, default="./data/MixSNIPS_clean/test.txt")
    data_arg.add_argument('--data_ratio', type=float, default=1.0)

    data_arg.add_argument('--cache_data_directory', type=str, default="./cache/")
    data_arg.add_argument('--generated_param_directory', type=str, default="./output")

    data_arg.add_argument('--lm', type=str, default="t5-base")
    learn_arg = add_argument_group('Learning')
    learn_arg.add_argument('--model_name', type=str, default="id_sf_")

    learn_arg.add_argument('--max_span_length', type=int, default=19)
    learn_arg.add_argument('--batch_size', type=int, default=32)
    learn_arg.add_argument('--max_epoch', type=int, default=20)
    learn_arg.add_argument('--debug', type=bool, default=False)

    learn_arg.add_argument('--gradient_accumulation_steps', type=int, default=5)
    learn_arg.add_argument('--print_steps', type=int, default=5)
    learn_arg.add_argument('--lr', type=float, default=3e-5)
    learn_arg.add_argument('--lr_decay', type=float, default=0.01)
    learn_arg.add_argument('--weight_decay', type=float, default=1e-5)
    learn_arg.add_argument('--max_grad_norm', type=float, default=1)
    learn_arg.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'AdamW'])

    evaluation_arg = add_argument_group('Evaluation')
    # evaluation_arg.add_argument('--max_length', type=int, default=48)
    misc_arg = add_argument_group('MISC')
    misc_arg.add_argument('--refresh', type=str2bool, default=False)
    misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
    misc_arg.add_argument('--visible_gpu', type=int, default=0)
    misc_arg.add_argument('--device_ids', type=list, default=[0, 1])
    misc_arg.add_argument('--random_seed', type=int, default=2022)

    args, unparsed = get_args()
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(args.visible_gpu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_id = args.device_ids[0]

    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    set_seed(args.random_seed)

    print('load dataset ...')
    intent_list = json.load(
        open(os.path.join('./data/', args.dataset_dir, "intent.json")))  # ./data/MixSNIPS_clean/intent.json
    slot_list = json.load(open(os.path.join('./data/', args.dataset_dir, "slot.json")))

    tokenizer = AutoTokenizer.from_pretrained(args.lm)

    data = build_data(args, tokenizer, intent_list, slot_list, base_config[args.dataset_name])

    _name = namegenerator.gen()
    model_name = args.model_name + _name + args.mode
    # To prevent model_name duplicated
    while pathlib.Path(os.path.join(args.generated_param_directory, model_name + '.pt')).exists():
        model_name = args.model_name + namegenerator.gen() + args.mode

    args.model_name = model_name
    print(f'model_name: {args.model_name}')
    print(f"question_valid: {base_config[args.dataset_name]['question_valid']}")

    # model = IDSF.from_pretrained(args.lm, base_config[args.dataset_name], bert_config, loss_weight)
    model = IDSF.from_pretrained(args.lm)
    print(f"k sample: {base_config[args.dataset_name]['k_sample']}")
    trainer = QaTrainer(model, data, args, device_id, tokenizer, base_config[args.dataset_name],
                        list(intent_list.values()),
                        list(slot_list.values()))
    trainer.train_model()
