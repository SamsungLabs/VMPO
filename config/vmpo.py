import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config(name):
    return globals()[name]()

def _get_config(base_model="sd3", n_gpus=1, gradient_step_per_epoch=1, dataset="ocr", reward_fn={}, name=""):
    config = base.get_config()
    assert base_model in ["sd3"]
    assert dataset in ["ocr"]   # only support ocr for now

    config.base_model = base_model
    config.dataset = os.path.join(os.getcwd(), f"dataset/{dataset}")
    if base_model == "sd3":
        config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
        config.sample.num_steps = 10
        config.sample.eval_num_steps = 40
        config.sample.guidance_scale = 4.5
        config.resolution = 512
        config.train.beta = 1.0
        config.sample.noise_level = 0.7
        bsz = 4

    num_image_per_epoch = 256   # effective sample size = num_image_per_epoch / gradient_step_per_epoch (larger -> more stable)
    assert num_image_per_epoch % (n_gpus * bsz) == 0, "num_image_per_epoch must be divisible by n_gpus * bsz"
    n_batch_per_epoch = num_image_per_epoch // (n_gpus * bsz)

    config.sample.train_batch_size = bsz
    config.train.batch_size = config.sample.train_batch_size
    config.sample.num_batches_per_epoch = n_batch_per_epoch
    config.train.gradient_accumulation_steps = (
        config.sample.num_batches_per_epoch // gradient_step_per_epoch
    )

    # special design, the test set has a total of 1018 for ocr, followed diffusionNFT & flow_grpo
    config.sample.test_batch_size = 16
    if n_gpus > 32:
        config.sample.test_batch_size = config.sample.test_batch_size // 2

    config.prompt_fn = "general_ocr"

    config.run_name = f"vmpo_{base_model}_{name}"
    config.save_dir = f"logs/vmpo/{base_model}/{name}"
    config.reward_fn = reward_fn

    # reward rescaling
    config.train.reward_anneal = 'linear'
    config.train.reward_exp = 10.0
    config.train.reward_warmup_epochs = 50


    config.sample.deterministic = False
    config.sample.solver = "flow"
    return config

def sd3_ocr():
    reward_fn = {
        "ocr": 1.0,
    }
    config = _get_config(
        base_model="sd3", n_gpus=4, gradient_step_per_epoch=2, dataset="ocr", reward_fn=reward_fn, name="ocr"
    )
    config.debug = False # True -> disable evaluation during training
    return config