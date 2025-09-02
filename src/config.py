from sacred import Experiment

ex = Experiment("ViLT")


def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "mppd": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
        "mmimdb": 0,
        "hatememes": 0,
        "food101": 0,        
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "src"
    seed = 42
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # eval config (for bash execution)
    test_ratio = None
    test_type = None
    test_exp_name = None
    
    # fix backbone model (ViLT) weights
    fix_model = True
    
    # missing modality config
    missing_ratio = {'train': 0.7, 'val': 0.7, 'test': 0.7}
    missing_type = {'train': 'image', 'val': 'image', 'test': 'image'}  # ['text', 'image', 'both'] in VL taskss
    both_ratio = 0.5   # missing both ratio
    missing_table_root = './datasets/missing_tables/'
    simulate_missing = False
    prompt_type = 'input'
    learnt_p = True
    multi_layer_prompt = True

    # prompts config
    gen_prompt_length = 16
    gen_prompt_layers = [0,1,2,3,4,5]

    shared_prompt_length = 16
    shared_prompt_layers = [0,1,2,3,4,5]

    mem_size = 16
    top_k = 5
    bottleneck_dim = 8

    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 384
    max_image_len = -1
    patch_size = 32
    draw_false_image = 1
    image_only = False

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    vit = "vit_base_patch32_384"
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

    # Downstream Setting
    get_recall_metric = False
    mmimdb_class_num = 23
    hatememes_class_num = 2
    food101_class_num = 101

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    finetune_first = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16


@ex.named_config
def task_finetune_mmimdb():
    exp_name = "finetune_mmimdb_both"
    datasets = ["mmimdb"]
    loss_names = _loss_names({"mmimdb": 1})
    batch_size = 256
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-3
    val_check_interval = 0.2
    weight_decay = 2e-2
    #     optim_type = "adam"
    max_text_len = 1024

@ex.named_config
def task_finetune_food101():
    exp_name = "finetune_food101_both111"
    datasets = ["Food101"]
    loss_names = _loss_names({"food101": 1})
    batch_size = 256
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-3
    val_check_interval = 0.2
    weight_decay = 2e-2
    #     optim_type = "adam"
    max_text_len = 512


@ex.named_config
def task_finetune_hatememes():
    exp_name = "finetune_hatememes"
    datasets = ["Hatefull_Memes"]
    loss_names = _loss_names({"hatememes": 1})
    batch_size = 256
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-3
    val_check_interval = 0.11
    weight_decay = 2e-2
    #     optim_type = "adam"
    max_text_len = 128