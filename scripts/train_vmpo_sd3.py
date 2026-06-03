from collections import defaultdict
import os
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
import logging
from diffusers import StableDiffusion3Pipeline
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from ml_collections import config_flags
from torch.amp import GradScaler, autocast as torch_autocast

import flow_grpo.rewards
from flow_grpo.ema import EMAModuleWrapper

from flow_grpo.mphi import ConditionalFPsiSD3
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import compute_log_prob, pred_orig_latent
from .utils import setup_distributed, cleanup_distributed, is_main_process, set_seed, decode_latents, \
    gather_tensor_to_all, compute_text_embeddings, TextPromptDataset

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def eval_fn(
    pipeline,
    test_dataloader,
    text_encoders,
    tokenizers,
    config,
    device,
    rank,
    world_size,
    global_step,
    reward_fn,
    executor,
    mixed_precision_dtype,
    ema,
    transformer_trainable_parameters,
):
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    pipeline.transformer.eval()

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_rewards = defaultdict(list)

    test_sampler = (
        DistributedSampler(test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    eval_loader = DataLoader(
        test_dataloader.dataset,
        batch_size=config.sample.test_batch_size,  # This is per-GPU batch size
        sampler=test_sampler,
        collate_fn=test_dataloader.collate_fn,
        num_workers=test_dataloader.num_workers,
    )

    for test_batch in tqdm(
        eval_loader,
        desc="Eval: ",
        disable=not is_main_process(rank),
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
        )
        current_batch_size = len(prompt_embeds)
        if current_batch_size < len(sample_neg_prompt_embeds):  # Handle last batch
            current_sample_neg_prompt_embeds = sample_neg_prompt_embeds[:current_batch_size]
            current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:current_batch_size]
        else:
            current_sample_neg_prompt_embeds = sample_neg_prompt_embeds
            current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds

        with torch_autocast(device_type="cuda", enabled=(config.mixed_precision in ["fp16", "bf16"]), dtype=mixed_precision_dtype):
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=current_sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=current_sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution, 
                    noise_level=0,
                )

        rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, reward_metadata = rewards_future.result()

        for key, value in rewards.items():
            rewards_tensor = torch.as_tensor(value, device=device).float()
            gathered_value = gather_tensor_to_all(rewards_tensor, world_size)
            all_rewards[key].append(gathered_value.numpy())

    if is_main_process(rank):
        final_rewards = {key: np.concatenate(value_list) for key, value_list in all_rewards.items()}

        images_to_log = images.cpu()
        prompts_to_log = prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples_to_log = min(15, len(images_to_log))
            for idx in range(num_samples_to_log):
                image = images_to_log[idx].float()
                pil = Image.fromarray((image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

            sampled_prompts_log = [prompts_to_log[i] for i in range(num_samples_to_log)]
            sampled_rewards_log = [{k: final_rewards[k][i] for k in final_rewards} for i in range(num_samples_to_log)]

            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | "
                            + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts_log, sampled_rewards_log))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in final_rewards.items()},
                },
                step=global_step,
            )

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)

    if world_size > 1:
        dist.barrier()


def save_ckpt(
    save_dir, transformer_ddp, mphi_model_ddp, global_step, rank, ema, transformer_trainable_parameters, config, optimizer, scaler
):
    if is_main_process(rank):
        save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
        save_root_lora = os.path.join(save_root, "lora")
        os.makedirs(save_root_lora, exist_ok=True)

        model_to_save = transformer_ddp.module

        if config.train.ema and ema is not None:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

        model_to_save.save_pretrained(save_root_lora)  # For LoRA/PEFT models

        torch.save(mphi_model_ddp.module.state_dict(), os.path.join(save_root, "mphi_model.pt"))

        torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))
        if scaler is not None:
            torch.save(scaler.state_dict(), os.path.join(save_root, "scaler.pt"))

        if config.train.ema and ema is not None:
            ema.copy_temp_to(transformer_trainable_parameters)
        logger.info(f"Saved checkpoint to {save_root}")


def main(_):
    config = FLAGS.config
    
    # --- Distributed Setup ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # --- WandB Init (only on main process) ---
    if is_main_process(rank):
        log_dir = os.path.join(config.logdir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        wandb.init(project="flow-grpo", name=config.run_name, config=config.to_dict(), dir=log_dir)
    logger.info(f"\n{config}")

    set_seed(config.seed, rank)  # Pass rank for different seeds per process

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler('cuda', enabled=enable_amp)

    # --- Load pipeline and models ---
    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_main_process(rank),
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32

    pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_3.to(device, dtype=text_encoder_dtype)

    transformer = pipeline.transformer.to(device)

    if config.use_lora:
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=target_modules
        )
        if config.train.lora_path:
            transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, transformer_lora_config)

    transformer_ddp = DDP(transformer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    transformer_ddp.module.set_adapter("default")
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))

    mphi_model = ConditionalFPsiSD3(
        caption_projection_dim=4096,
        pooled_projection_dim=2048,
        num_attention_heads=32,
        attention_head_dim=128,
        num_layers=4,
        output_dim=1,
    ).to(device, dtype=torch.float32)
    mphi_model_ddp = DDP(mphi_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    mphi_trainable_parameters = list(filter(lambda p: p.requires_grad, mphi_model_ddp.module.parameters()))

    trainable_params = [
        {"params": transformer_trainable_parameters, "lr": config.train.learning_rate},
        {"params": mphi_trainable_parameters, "lr": config.train.learning_rate_mphi},
    ]

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Optimizer ---
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        trainable_params,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # --- Datasets and Dataloaders ---
    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, "train")
        test_dataset = TextPromptDataset(config.dataset, "test")
    else:
        raise NotImplementedError("Prompt function not supported with dataset")

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.sample.train_batch_size,  # Per-GPU
        sampler=train_sampler,  # Use distributed sampler for training
        collate_fn=TextPromptDataset.collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    test_sampler = (
        DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,   # Per-GPU
        sampler=test_sampler,  # Use distributed sampler for testing
        collate_fn=TextPromptDataset.collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # --- Prompt Embeddings ---
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    executor = futures.ThreadPoolExecutor(max_workers=8)  # Async reward computation

    # Train!
    samples_per_epoch = config.sample.train_batch_size * world_size * config.sample.num_batches_per_epoch
    total_train_batch_size = config.train.batch_size * world_size * config.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    reward_fn = getattr(flow_grpo.rewards, "multi_score")(device, config.reward_fn)  # Pass device
    eval_reward_fn = getattr(flow_grpo.rewards, "multi_score")(device, config.reward_fn)  # Pass device

    # --- Resume from checkpoint ---
    first_epoch = 0
    global_step = 0
    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        # Assuming checkpoint dir contains lora, optimizer.pt, scaler.pt
        lora_path = os.path.join(config.resume_from, "lora")
        if os.path.exists(lora_path):  # Check if it's a PEFT model save
            transformer_ddp.module.load_adapter(lora_path, adapter_name="default", is_trainable=True)
        else:  # Try loading full state dict if it's not a PEFT save structure
            model_ckpt_path = os.path.join(config.resume_from, "transformer_model.pt")  # Or specific name
            if os.path.exists(model_ckpt_path):
                transformer_ddp.module.load_state_dict(torch.load(model_ckpt_path, map_location=device))

        mphi_path = os.path.join(config.resume_from, "mphi_model.pt")
        if os.path.exists(mphi_path):
            mphi_model_ddp.module.load_state_dict(torch.load(mphi_path, map_location=device))

        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))

        scaler_path = os.path.join(config.resume_from, "scaler.pt")
        if os.path.exists(scaler_path) and enable_amp:
            scaler.load_state_dict(torch.load(scaler_path, map_location=device))

        # Extract epoch and step from checkpoint name, e.g., "checkpoint-1000" -> global_step = 1000
        try:
            global_step = int(os.path.basename(config.resume_from).split("-")[-1])
            logger.info(f"Resumed global_step to {global_step}. Epoch estimation might be needed.")
        except ValueError:
            logger.warning(
                f"Could not parse global_step from checkpoint name: {config.resume_from}. Starting global_step from 0."
            )
            global_step = 0

    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=1, device=device)

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    logger.info("***** Running training *****")

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    for epoch in range(first_epoch, config.num_epochs):

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples_data_list = []

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not is_main_process(rank),
            position=0,
        ):
            try:
                prompts, prompt_metadata = next(train_iter)
            except StopIteration:
                # Recreate iterator when exhausted
                train_iter = iter(train_dataloader)
                prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
            )
            prompt_ids = tokenizers[0](
                prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt"
            ).input_ids.to(device)

            if i == 0 and epoch % config.eval_freq == 0 and not config.debug:
                eval_fn(
                    pipeline,
                    test_dataloader,
                    text_encoders,
                    tokenizers,
                    config,
                    device,
                    rank,
                    world_size,
                    global_step,
                    eval_reward_fn,
                    executor,
                    mixed_precision_dtype,
                    ema,
                    transformer_trainable_parameters,
                )

            if i == 0 and epoch % config.save_freq == 0 and is_main_process(rank) and not config.debug:
                save_ckpt(
                    config.save_dir,
                    transformer_ddp,
                    mphi_model_ddp,
                    global_step,
                    rank,
                    ema,
                    transformer_trainable_parameters,
                    config,
                    optimizer,
                    scaler,
                )

            with torch_autocast(device_type="cuda", enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    images, latents, log_probs = pipeline_with_logprob(
                            pipeline,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_prompt_embeds=sample_neg_prompt_embeds[: len(prompts)],
                            negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[: len(prompts)],
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            output_type="pt",
                            height=config.resolution,
                            width=config.resolution, 
                            noise_level=config.sample.noise_level,
                    )

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 16, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps)
            timesteps = pipeline.scheduler.timesteps.repeat(
                len(prompts), 1
            )  # (batch_size, num_steps)
            next_timesteps = torch.cat(
                [timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1
            )  # (batch_size, num_steps)

            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)

            samples_data_list.append(
                {
                    "prompts": prompts, # tuple of strings
                    "prompt_metadata": prompt_metadata,
                    # "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "next_timesteps": next_timesteps,
                    "latents": latents[:, :-1],     # each entry is the latent before timestep t
                    "next_latents": latents[:, 1:], # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards_future": rewards_future,  # Store future
                }
            )
        
        # wait for all rewards to be computed
        for sample_item in tqdm(
            samples_data_list, desc="Waiting for rewards", disable=not is_main_process(rank), position=0
        ):
            rewards, reward_metadata = sample_item["rewards_future"].result()
            sample_item["rewards"] = {k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()}
            del sample_item["rewards_future"]

        # Collate samples
        collated_samples = {
            k: (
                [item for s in samples_data_list for item in s[k]] # list of tuples [('cat', 'dog'), ('cat', 'tiger'), ...] -> list ['cat', 'dog', 'cat', 'tiger', ...]
                if k in ["prompts", "prompt_metadata"]
                else (
                    torch.cat([s[k] for s in samples_data_list], dim=0)
                    if not isinstance(samples_data_list[0][k], dict)
                    else {sk: torch.cat([s[k][sk] for s in samples_data_list], dim=0) for sk in samples_data_list[0][k]}
                )
            )
            for k in samples_data_list[0].keys()
        }

        # Logging images (main process)
        if epoch % 10 == 0 and is_main_process(rank):
            images_to_log = images.cpu()  # from last sampling batch on this rank
            prompts_to_log = prompts  # from last sampling batch on this rank
            rewards_to_log = collated_samples["rewards"]["avg"][-len(images_to_log) :].cpu()

            with tempfile.TemporaryDirectory() as tmpdir:
                num_to_log = min(15, len(images_to_log))
                for idx in range(num_to_log):  # log first N
                    img_data = images_to_log[idx]
                    pil = Image.fromarray((img_data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompts_to_log[idx]:.100} | avg: {rewards_to_log[idx]:.2f}",
                            )
                            for idx in range(num_to_log)
                        ],
                    },
                    step=global_step,
                )

         # Gather rewards across processes
        gathered_rewards_dict = {}
        for key, value_tensor in collated_samples["rewards"].items():
            gathered_rewards_dict[key] = gather_tensor_to_all(value_tensor, world_size).numpy()

        if is_main_process(rank):  # logging
            wandb.log(
                {
                    "epoch": epoch,
                    **{
                        f"reward_{k}": v.mean()
                        for k, v in gathered_rewards_dict.items()
                        if "_strict_accuracy" not in k and "_accuracy" not in k
                    },
                },
                step=global_step,
            )

        collated_samples["rewards"] = collated_samples["rewards"]['avg']
        if is_main_process(rank):
            logger.info(f"Rewards mean: {collated_samples['rewards'].mean().item()}")


        num_batches = config.sample.num_batches_per_epoch * config.sample.train_batch_size // config.train.batch_size
        filtered_samples = collated_samples
        total_batch_size_filtered, num_timesteps_filtered = filtered_samples["timesteps"].shape

        if config.train.reward_anneal == 'linear':
            ratio = min(1, epoch / config.train.reward_warmup_epochs)
        else:
            ratio = 1.0
        reward_exp_ep = config.train.reward_exp * ratio
        def reward_transform(value):
            return value * reward_exp_ep

        #################### TRAINING ####################
        transformer_ddp.train()  # Sets DDP model and its submodules to train mode.
        mphi_model_ddp.train()

        # Total number of backward passes before an optimizer step
        effective_grad_accum_steps = config.train.gradient_accumulation_steps * num_train_timesteps

        current_accumulated_steps = 0  # Counter for backward passes
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled_filtered_samples = {
                k: [v[i] for i in perm]
                if k in ["prompts", "prompt_metadata"]
                else v[perm] for k, v in filtered_samples.items()
            }

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [torch.randperm(num_timesteps_filtered, device=device) for _ in range(total_batch_size_filtered)]
            )
            for key in ["timesteps", "next_timesteps", "latents", "next_latents", "log_probs"]:
                shuffled_filtered_samples[key] = shuffled_filtered_samples[key][
                    torch.arange(total_batch_size_filtered, device=device)[:, None], perms,
                ]

            # rebatch for training
            training_batch_size = total_batch_size_filtered // num_batches
            samples_batched_list = []
            for k_batch in range(num_batches):
                batch_dict = {}
                start = k_batch * training_batch_size
                end = (k_batch + 1) * training_batch_size
                for key, val_tensor in shuffled_filtered_samples.items():
                    batch_dict[key] = val_tensor[start:end]
                samples_batched_list.append(batch_dict)

            info = defaultdict(list)  # For accumulating stats over one grad acc cycle

            for i, train_sample_batch in tqdm(
                list(enumerate(samples_batched_list)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not is_main_process(rank),
            ):
                current_micro_batch_size = len(train_sample_batch["prompt_embeds"])

                if config.sample.guidance_scale > 1.0:
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:current_micro_batch_size], train_sample_batch["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [
                            train_neg_pooled_prompt_embeds[:current_micro_batch_size],
                            train_sample_batch["pooled_prompt_embeds"],
                        ]
                    )
                else:
                    embeds = train_sample_batch["prompt_embeds"]
                    pooled_embeds = train_sample_batch["pooled_prompt_embeds"]

                # Loop over timesteps for this micro-batch
                for j_idx, j_timestep_orig_idx in tqdm(
                    enumerate(range(num_train_timesteps)),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not is_main_process(rank),
                ):
                    assert j_idx == j_timestep_orig_idx

                    with torch_autocast(device_type="cuda", enabled=enable_amp, dtype=mixed_precision_dtype):
                        _, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(transformer_ddp, pipeline, train_sample_batch, j_idx, embeds, pooled_embeds, config)
                        if config.train.beta > 0:
                            with torch.no_grad():
                                with transformer_ddp.module.disable_adapter():
                                    _, log_prob_ref, prev_sample_mean_ref, _ = compute_log_prob(transformer_ddp, pipeline, train_sample_batch, j_idx, embeds, pooled_embeds, config)

                    # Compute reward at current timestep (noisier)
                    with torch_autocast(device_type="cuda", enabled=enable_amp, dtype=mixed_precision_dtype), torch.no_grad():
                        orig_latent = pred_orig_latent(transformer_ddp, pipeline, train_sample_batch["latents"][:, j_idx], train_sample_batch["timesteps"][:, j_idx], embeds, pooled_embeds, config)
                    with torch.inference_mode():
                        rewards = executor.submit(reward_fn, decode_latents(orig_latent, pipeline), train_sample_batch["prompts"], train_sample_batch["prompt_metadata"], only_strict=False)
                    time.sleep(0)
                    logr_t = torch.as_tensor(rewards.result()[0]['avg'], device=device).float()
                    logr_t = reward_transform(logr_t)

                    # Compute reward at next timestep (cleaner)
                    with torch_autocast(device_type="cuda", enabled=enable_amp, dtype=mixed_precision_dtype), torch.no_grad():
                        next_orig_latent = pred_orig_latent(transformer_ddp, pipeline, train_sample_batch["next_latents"][:, j_idx], train_sample_batch["next_timesteps"][:, j_idx], embeds, pooled_embeds, config)
                    with torch.inference_mode():
                        next_rewards = executor.submit(reward_fn, decode_latents(next_orig_latent, pipeline), train_sample_batch["prompts"], train_sample_batch["prompt_metadata"], only_strict=False)
                    time.sleep(0)
                    logr_t_minus_1 = torch.as_tensor(next_rewards.result()[0]['avg'], device=device).float()
                    logr_t_minus_1 = reward_transform(logr_t_minus_1)
                    end_mask = train_sample_batch["timesteps"][:, j_idx] == pipeline.scheduler.timesteps[-1]
                    logr_t_minus_1[end_mask] = reward_transform(train_sample_batch['rewards'][end_mask].to(logr_t))

                    logr_diff = logr_t_minus_1 - logr_t
                    logp_diff = log_prob_ref - log_prob

                    with torch_autocast(device_type="cuda", enabled=False):
                        mphi_value = mphi_model_ddp(
                            timestep=train_sample_batch["timesteps"][:, j_idx].float(),
                            encoder_hidden_states=(embeds if not pipeline.do_classifier_free_guidance else embeds[len(train_sample_batch["prompt_embeds"]):]).float(),  # Remove negative prompts (first half of the batch)
                            pooled_projections=(pooled_embeds if not pipeline.do_classifier_free_guidance else pooled_embeds[len(train_sample_batch["pooled_prompt_embeds"]):]).float()
                        )

                    log_w = logr_diff + logp_diff
                    # we use two different losses to update transformer and mphi
                    # this enables us to use a small coefficient (e.g., 0.01) on logvar_loss_2
                    # this is important to stabilise the gradient norm of mphi during training
                    logvar_loss_1 = torch.mean((log_w - mphi_value.detach()) ** 2)
                    logvar_loss_2 = torch.mean((log_w.detach() - mphi_value) ** 2)
                    logvar_loss = logvar_loss_1 + 0.01 * logvar_loss_2

                    if config.train.beta > 0:
                        kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
                        kl_loss = torch.mean(kl_loss)
                        loss = logvar_loss + config.train.beta * kl_loss
                    else:
                        loss = logvar_loss

                    info["approx_kl"].append(0.5 * torch.mean((log_prob - train_sample_batch["log_probs"][:, j_idx]) ** 2).detach())
                    info["logr_diff"].append(torch.mean(logr_diff).detach())
                    info["logp_diff"].append(torch.mean(logp_diff).detach())
                    info["mphi_value"].append(torch.mean(mphi_value).detach())
                    info["log_w"].append(torch.mean(log_w).detach())
                    info["logvar_loss"].append(logvar_loss.detach())
                    info["loss"].append(loss.detach())
                    if config.train.beta > 0:
                        info["kl_loss"].append(kl_loss.detach())

                    # Scale loss for gradient accumulation and DDP (DDP averages grads, so no need to divide by world_size here)
                    scaled_loss = loss / effective_grad_accum_steps
                    if mixed_precision_dtype == torch.float16:
                        scaler.scale(scaled_loss).backward()  # one accumulation
                    else:
                        scaled_loss.backward()
                    current_accumulated_steps += 1

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        if mixed_precision_dtype == torch.float16:
                            scaler.unscale_(optimizer)

                        transformer_grad_norm = torch.nn.utils.clip_grad_norm_(
                            transformer_trainable_parameters, 
                            config.train.max_grad_norm
                        )
                        mphi_grad_norm = torch.nn.utils.clip_grad_norm_(
                            mphi_trainable_parameters,
                            config.train.max_grad_norm
                        )
                        info['transformer_grad_norm'] = (transformer_grad_norm.detach(),)
                        info['mphi_grad_norm'] = (mphi_grad_norm.detach(),)

                        if mixed_precision_dtype == torch.float16:
                            scaler.step(optimizer)
                        else:
                            optimizer.step()
                        gradient_update_times += 1
                        if mixed_precision_dtype == torch.float16:
                            scaler.update()
                        optimizer.zero_grad()

                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info_tensor = torch.tensor([info[k] for k in sorted(info.keys())], device=device)
                        dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG)
                        reduced_log_info = {k: info_tensor[ki].item() for ki, k in enumerate(sorted(info.keys()))}
                        if is_main_process(rank):
                            wandb.log(
                                {
                                    "step": global_step,
                                    "gradient_update_times": gradient_update_times,
                                    "epoch": epoch,
                                    "inner_epoch": inner_epoch,
                                    **reduced_log_info,
                                }
                            )

                        global_step += 1  # gradient step
                        info = defaultdict(list)  # Reset for next accumulation cycle

                if (
                    config.train.ema
                    and ema is not None
                    and (current_accumulated_steps % effective_grad_accum_steps == 0)
                ):
                    ema.step(transformer_trainable_parameters, global_step)

        if world_size > 1:
            dist.barrier()

    if is_main_process(rank):
        wandb.finish()
    cleanup_distributed()

if __name__ == "__main__":
    app.run(main)
