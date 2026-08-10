import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import pickle
import tqdm
import numpy as np
import math
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from umi_day.train_network.utils.load_env import load_env_runner, env_rollout
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
import accelerate
from accelerate import Accelerator
from datetime import timedelta
from accelerate import InitProcessGroupKwargs
from umi_day.train_network.utils.training_utils import get_gradient_norm

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainPolicyWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.model)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # if not rolling out, montiro the training loss instead for checkpointing
        if cfg.checkpoint.topk.monitor_key == 'test_mean_score' and cfg.training.rollout_every == -1:
            print('updating checkpoint monitor key to `train_loss` since no evaluation is done')
            cfg.checkpoint.topk.monitor_key = 'train_loss'
            cfg.checkpoint.topk.format_str = cfg.checkpoint.topk.format_str.replace('test_mean_score', 'train_loss')

        os.environ['TOKENIZERS_PARALLELISM'] = 'false' # disable parallelism to remove warnings about tokenizer issues after process is forked due to dataloader having multiple worker processes
        timeout = InitProcessGroupKwargs(timeout=timedelta(minutes=120)) # two hour timeout on multi GPU NCCL (default is 30 min). This helps with long setup before training where processes have to wait for env_runners to initialize
        accelerator = Accelerator(log_with='wandb', kwargs_handlers=[timeout], mixed_precision=cfg.training.mixed_precision)
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # ensure all processes use the same output directory
        output_dirs = accelerate.utils.gather_object([self.output_dir] if accelerator.is_main_process else [''])
        main_output_dir = [x for x in output_dirs if x][0]
        self._output_dir = main_output_dir
        accelerator.print(f'Started training. Run dir: {self.output_dir}')

        # configure optimizer
        if cfg.training.scale_lr_with_gpus:
            cfg.optimizer.lr *= accelerator.num_processes # scale LR by num GPUs; see see https://huggingface.co/docs/accelerate/concept_guides/performance
        obs_encorder_lr = cfg.optimizer.lr
        if cfg.model.obs_encoder.pretrained:
            obs_encorder_lr *= 0.1
            accelerator.print('==> reduce pretrained obs_encorder\'s lr')
        obs_encorder_params = list()
        for param in self.model.obs_encoder.parameters():
            if param.requires_grad:
                obs_encorder_params.append(param)
        accelerator.print(f'obs_encorder params: {len(obs_encorder_params)}')
        param_groups = [
            {'params': self.model.model.parameters()},
            {'params': obs_encorder_params, 'lr': obs_encorder_lr}
        ]
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, param_groups)

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset) or isinstance(dataset, BaseDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            pickle.dump(normalizer, open(normalizer_path, 'wb'))

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        accelerator.print('train dataset:', len(dataset), 'train dataloader:', len(train_dataloader))
        accelerator.print('val dataset:', len(val_dataset), 'val dataloader:', len(val_dataloader))

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        if cfg.training.rollout_every != -1 and accelerator.is_main_process:
            env_runners = load_env_runner(cfg, self.output_dir)
        accelerator.wait_for_everyone() # since loading envs can take a while, halt all processes until it's finished to prevent timeout

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # accelerator
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
        )
        device = self.model.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                accelerator.wait_for_everyone() # since some of the evaluations can take a while (env rollout happens only on main process), halt all processes until it's finished to prevent timeout

                self.model.train()

                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        step_log = {}
                        
                        # always use the latest batch
                        train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        accelerator.backward(loss)

                        # log grad norms
                        if cfg.training.log_grad_norm_every != -1 and self.global_step % cfg.training.log_grad_norm_every == 0:
                            step_log['grad_norm'] = get_gradient_norm(self.model)
                            if cfg.training.clip_grad_norm:
                                step_log['grad_norm_clipped'] = min(cfg.training.clip_grad_norm, step_log['grad_norm'])

                        if cfg.training.clip_grad_norm and accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(self.model.parameters(), cfg.training.clip_grad_norm)

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(accelerator.unwrap_model(self.model))

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log.update({
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        })

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0 and cfg.training.rollout_every != -1 and accelerator.is_main_process:
                    runner_log = env_rollout(cfg, env_runners, policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0 and len(val_dataloader) > 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = policy(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                
                def log_action_mse(step_log, category, pred_action, batch):
                    """Computes and logs action MSE error per task"""
                    gt_action = batch['action']

                    B, T, _ = pred_action.shape
                    pred_action = pred_action.view(B, T, -1, 10)
                    gt_action = gt_action.view(B, T, -1, 10)

                    # compute overall loss
                    step_log[f'{category}/action_mse_error/all'] = torch.nn.functional.mse_loss(pred_action, gt_action)
                    step_log[f'{category}/action_mse_error_pos/all'] = torch.nn.functional.mse_loss(pred_action[..., :3], gt_action[..., :3])
                    step_log[f'{category}/action_mse_error_rot/all'] = torch.nn.functional.mse_loss(pred_action[..., 3:9], gt_action[..., 3:9])
                    step_log[f'{category}/action_mse_error_width/all'] = torch.nn.functional.mse_loss(pred_action[..., 9], gt_action[..., 9])

                    # compute loss by task name
                    if 'task_idx' in batch['metadata']:
                        task_indices = batch['metadata']['task_idx']
                        unique_task_indices = dataset.task_idx_to_unique_task_idx[task_indices.cpu().numpy()]

                        for unique_task_idx, unique_task_name in enumerate(dataset.unique_task_names):
                            cur_task_batch_indices = torch.from_numpy(np.where(unique_task_indices == unique_task_idx)[0]).to(gt_action.device)
                            task_entry_count = len(cur_task_batch_indices)

                            if task_entry_count > 0:
                                cur_pred_actions = pred_action[cur_task_batch_indices]
                                cur_gt_actions = gt_action[cur_task_batch_indices]

                                step_log[f'{category}/action_mse_error/{unique_task_name}'] = torch.nn.functional.mse_loss(cur_pred_actions, cur_gt_actions)
                                step_log[f'{category}/action_mse_error_pos/{unique_task_name}'] = torch.nn.functional.mse_loss(cur_pred_actions[..., :3], cur_gt_actions[..., :3])
                                step_log[f'{category}/action_mse_error_rot/{unique_task_name}'] = torch.nn.functional.mse_loss(cur_pred_actions[..., 3:9], cur_gt_actions[..., 3:9])
                                step_log[f'{category}/action_mse_error_width/{unique_task_name}'] = torch.nn.functional.mse_loss(cur_pred_actions[..., 9], cur_gt_actions[..., 9])

                    del gt_action

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        pred_action = policy.predict_action(batch['obs'], None)['action_pred']
                        log_action_mse(step_log, 'train', pred_action, batch)

                        if len(val_dataloader) > 0:
                            val_sampling_batch = next(iter(val_dataloader))
                            batch = dict_apply(val_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            pred_action = policy.predict_action(batch['obs'], None)['action_pred']
                            log_action_mse(step_log, 'val', pred_action, batch)

                        del batch
                        del pred_action
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and accelerator.is_main_process:
                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                    # recover the DDP model
                    self.model = model_ddp
                # ========= eval end for this epoch ==========
                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        accelerator.end_training()
