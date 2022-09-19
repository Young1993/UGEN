base_config = {
    "MixSNIPS": {
        "full_data": True,
        "MAX_LEN_Q": 320,  # 5-shot:256, 10-shot:284,
        "MAX_LEN_A": 140,  # 5-shot:110, 10-shot:120,
        "k_sample": 5,  # 5-shot 10-shot 78 (10%)  400
        "question_num": 5,  # 2 6
        "step": 50,
        "intent_label_length": 13,
        "slot_input_maxlength": 112,
        "slot_label_length": 20,
        "slot_type": 39,
        "question_valid": [1, 2, 3, 4, 5]
    },
    "MixATIS": {
        "full_data": False,
        "step": 500,
        "MAX_LEN_Q": 610,  # 256
        "MAX_LEN_A": 220,  # 110
        "k_sample": 5,  # 5 10 78
        "question_num": 6,
        "slot_type": 78,
        "intent_label_length": 13,
        "slot_input_maxlength": 112,
        "slot_label_length": 20,
        "question_valid": [1, 3, 4, 5]
    }
}
