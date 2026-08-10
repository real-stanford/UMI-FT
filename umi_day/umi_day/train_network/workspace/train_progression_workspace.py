import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import pickle
import tqdm
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi_day.train_network.model.progression_net import ProgressionNet
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from umi_day.common.plot_util import plot_progressions, plot_progression_and_frame
from umi_day.train_network.common.torch_util import add_batch_dim
from umi_day.train_network.dataset.umi_task_dataset import UmiTaskDataset
import accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed
import cv2
import wandb
from umi_day.train_network.utils.training_utils import get_gradient_norm

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainProgressionWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        set_seed(seed)

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # configure model
        self.model: ProgressionNet = hydra.utils.instantiate(cfg.model)

        self.ema_model: ProgressionNet = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # configure accelerator + logging
        os.environ['TOKENIZERS_PARALLELISM'] = 'false' # disable parallelism to remove warnings about tokenizer issues after process is forked due to dataloader having multiple worker processes
        accelerator = Accelerator(log_with='wandb', mixed_precision=cfg.training.mixed_precision)
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
            cfg.optimizer.lr *= accelerator.num_processes # scale LR by num GPUs; see https://huggingface.co/docs/accelerate/concept_guides/performance
        obs_encorder_lr = cfg.optimizer.lr
        if cfg.model.obs_encoder.pretrained:
            obs_encorder_lr *= 0.1
            accelerator.print('==> reduce pretrained obs_encorder\'s lr')
        obs_encoder_params = list()
        for param in self.model.obs_encoder.parameters():
            if param.requires_grad:
                obs_encoder_params.append(param)
        accelerator.print(f'obs_encorder params: {len(obs_encoder_params)}')
        param_groups = [
            {'params': self.model.model.parameters(), 'initial_lr': cfg.optimizer.lr},
            {'params': obs_encoder_params, 'lr': obs_encorder_lr, 'initial_lr': obs_encorder_lr}
        ]
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, param_groups)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: UmiTaskDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process and not cfg.training.resume:
            normalizer = dataset.get_normalizer()
            pickle.dump(normalizer, open(normalizer_path, 'wb'))

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        accelerator.print('train dataset:', len(dataset), 'train dataloader:', len(train_dataloader))
        accelerator.print('val dataset:', len(val_dataset), 'val dataloader:', len(val_dataloader))

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

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # progression visualization
        @torch.inference_mode()
        def visualize_progression(out_fname, is_train, include_video=False):
            """
            Plot the progression for each unique language label in the validation dataset.
            Args:
            - out_fname: str, the name of the output file (excluding extension)
            - is_train: bool, whether this is a training or validation visualization
            - include_video: bool, whether to include a video of the progression and RGB observations side by side
            """
            if not accelerator.is_main_process:
                return
            
            split_name = 'train' if is_train else 'val'
            split_dataset = dataset if is_train else val_dataset
            split_mask = dataset.train_mask if is_train else dataset.val_mask
            out_fname = f'{split_name}_{out_fname}'

            out_path_base = os.path.join(self.output_dir, 'progression_vis', out_fname)
            os.makedirs(os.path.dirname(out_path_base), exist_ok=True)
            
            model = accelerator.unwrap_model(self.model)
            if cfg.training.use_ema:
                model = self.ema_model
            model.eval()

            # find the indices of the tasks that have a unique task name
            task_names = dataset.replay_buffer.task_names
            unique_indices = {}
            for task_idx in np.where(split_mask)[0]:
                task_name = task_names[task_idx]
                if task_name not in unique_indices:
                    unique_indices[task_name] = task_idx
            keys_sorted = sorted(unique_indices.keys())
            unique_task_idxs = [unique_indices[k] for k in keys_sorted]

            # Iterate through one task and plot the progression
            pred_progressions = []
            true_progressions = []
            titles = []

            def compute_task_idx_progression(task_idx):
                episode_idx = split_dataset.replay_buffer.get_episode_idx_from_task_idx(task_idx)
                episode_name = split_dataset.replay_buffer.episode_names[episode_idx]
                task_name = split_dataset.replay_buffer.get_task_name(task_idx)
                title = f'{out_fname}\n\"{task_name}\"\n{episode_name}'
                titles.append(title)
            
                pred_progression = []
                true_progression = []            

                start_idx = split_dataset.sampler.start_of_segment_indices[task_idx]
                end_idx = split_dataset.sampler.end_of_segment_indices[task_idx]
                iterable = range(start_idx, end_idx)
                if include_video:
                    video_path = out_path_base + f'_{task_name.replace(" ", "_")}.mp4'
                    iterable = tqdm.tqdm(iterable, desc=f"Generating progression video \"{os.path.basename(video_path)}\"", leave=False)
                for i in iterable:
                    entry = split_dataset[i]
                    entry = add_batch_dim(entry)

                    # ensure that we are going through a single task (assumes val dataset is not shuffled)
                    assert entry['metadata']['task_idx'].item() == task_idx
                    
                    # need some mapping from demonstration idx to demonstration name
                    progression = model(entry)
                    pred_progression.append(progression.item())
                    true_progression.append(entry['labels']['progression'].item())

                    if include_video:
                        # include all frames that have _rgb in the key
                        video_frames = []
                        for obs_name in cfg.shape_meta.obs:
                            obs_cfg = cfg.shape_meta.obs[obs_name]
                            if obs_cfg.type == 'rgb':
                                video_frames.append(entry['obs'][obs_name][0][-1].moveaxis(0,2).cpu().numpy()) # index into batch=1 and horizon (get last entry which is current frame) # (H, W, 3)
                        video_frame = np.hstack(video_frames) 
                        video_frame = (video_frame*255).astype(np.uint8)

                        combined_frame = plot_progression_and_frame(video_frame, pred_progression, true_progression, end_idx-start_idx, title)
                        combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)

                        if i == start_idx:
                            video_path = out_path_base + f'_{task_name.replace(" ", "_")}.mp4'
                            video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('a','v','c','1'), 60, (combined_frame.shape[1], combined_frame.shape[0]))

                        video.write(combined_frame)

                if include_video:
                    video.release()
                    accelerator.get_tracker("wandb").log({f"{split_name}/progression_video/{task_name}": wandb.Video(video_path)}, commit=False)

                pred_progressions.append(pred_progression)
                true_progressions.append(true_progression)

            for task_idx in tqdm.tqdm(unique_task_idxs, desc=f"Visualizing progression \"{out_fname}\"", leave=False):
                compute_task_idx_progression(task_idx)
            
            # Plot the progression vs ground truth
            progression_plot_out_path = out_path_base + '.png'
            plot_progressions(progression_plot_out_path, pred_progressions, true_progressions, titles)
            accelerator.get_tracker("wandb").log({f"{split_name}/progression_plot/all": wandb.Image(progression_plot_out_path)}, commit=False)

        def compute_completion_metrics(pred, gt):
            threshold = cfg.task.task_completion_threshold
            false_positives = torch.sum((pred > threshold) & (gt < threshold)).item()
            false_negatives = torch.sum((pred < threshold) & (gt > threshold)).item()
            total_positives = torch.sum(gt > threshold).item()
            total_negatives = torch.sum(gt < threshold).item()

            completion_metrics = torch.tensor([false_positives, false_negatives, total_positives, total_negatives], dtype=torch.int32)
            return completion_metrics

        def compute_metrics(pred, batch):
            """Compute total loss, and loss by task index."""
            # compute raw loss
            gt = batch['labels']['progression'].to(pred.dtype)
            raw_loss = torch.nn.functional.mse_loss(pred, gt)

            with torch.inference_mode():
                count = len(pred)

                # compute total loss
                loss = torch.nn.functional.mse_loss(pred, gt, reduction='sum').item()
                loss = torch.tensor([loss, count], dtype=torch.float32)

                # compute total completion
                completion = compute_completion_metrics(pred, gt)

                # compute total average error
                error = torch.sum(torch.abs(pred - gt)).item()
                error = torch.tensor([error, count], dtype=torch.float32)

                # compute metrics by task name
                task_indices = batch['metadata']['task_idx']
                unique_task_indices = dataset.task_idx_to_unique_task_idx[task_indices.cpu().numpy()]
                loss_by_task_name = {}
                completion_by_task_name = {}
                error_by_task_name = {}

                for unique_task_idx, unique_task_name in enumerate(dataset.unique_task_names):
                    cur_task_batch_indices = torch.from_numpy(np.where(unique_task_indices == unique_task_idx)[0]).to(gt.device)
                    task_entry_count = len(cur_task_batch_indices)

                    if task_entry_count > 0:
                        cur_task_preds = pred[cur_task_batch_indices]
                        cur_task_labels = gt[cur_task_batch_indices]

                        # loss
                        task_loss = torch.nn.functional.mse_loss(cur_task_preds, cur_task_labels, reduction='sum').item()
                        loss_by_task_name[unique_task_name] = torch.tensor([task_loss, task_entry_count], dtype=torch.float32)

                        # completion
                        task_completion_metrics = compute_completion_metrics(cur_task_preds, cur_task_labels)
                        completion_by_task_name[unique_task_name] = task_completion_metrics

                        # average error
                        task_error = torch.sum(torch.abs(cur_task_preds - cur_task_labels)).item()
                        error_by_task_name[unique_task_name] = torch.tensor([task_error, task_entry_count], dtype=torch.float32)
            metrics = {
                'raw_loss': raw_loss,
                'loss': loss,
                'completion': completion,
                'error': error,
                'loss_by_task_name': loss_by_task_name,
                'completion_by_task_name': completion_by_task_name,
                'error_by_task_name': error_by_task_name
            }
            
            return metrics
        
        def aggregate_metrics_across_batch(existing_metrics, new_metrics):
            for key, value in new_metrics.items():
                if key in existing_metrics:
                    if type(value) == dict:
                        aggregate_metrics_across_batch(existing_metrics[key], new_metrics[key])
                    else:   
                        existing_metrics[key] += value # uses the fact that you can add the loss, completion, and error metrics since they have summed value and counts
                else:
                    existing_metrics[key] = value

        def get_step_log(metrics, split_name):
            """
            `metrics` is a dictionary returned by `compute_metrics`.
            `split_name` is a string indicating the split name (e.g. 'train', 'val').
            `additional_metrics` is a dictionary of additional metrics to log.
            Note `raw_loss` is not logged since it is only for backward gradient computation.
            """
            log_metrics = {}

            # loss
            log_metrics['loss/all'] = (metrics['loss'][0] / metrics['loss'][1]).item() # divide summed loss by number of entries

            # loss_by_task_name
            for task_name, loss_metrics in metrics['loss_by_task_name'].items():
                log_metrics[f'loss/{task_name}'] = (loss_metrics[0] / loss_metrics[1]).item() # divide summed loss by number of entries

            # completion_metrics
            completion = metrics['completion']
            false_positives, false_negatives, total_positives, total_negatives = completion
            if total_negatives > 0:
                false_positive_rate = false_positives / total_negatives
                log_metrics[f'completion/false_positive_rate/all'] = false_positive_rate
            if total_positives > 0:
                false_negative_rate = false_negatives / total_positives
                log_metrics[f'completion/false_negative_rate/all'] = false_negative_rate

            # completion_metrics_by_task_name
            for task_name, completion_for_task in metrics['completion_by_task_name'].items():
                false_positives, false_negatives, total_positives, total_negatives = completion_for_task
                if total_negatives > 0:
                    false_positive_rate = false_positives / total_negatives
                    log_metrics[f'completion/false_positive_rate/{task_name}'] = false_positive_rate
                if total_positives > 0:
                    false_negative_rate = false_negatives / total_positives
                    log_metrics[f'completion/false_negative_rate/{task_name}'] = false_negative_rate

            # error
            log_metrics['error/all'] = (metrics['error'][0] / metrics['error'][1]).item() # divide summed error by number of entries

            # error_by_task_name
            for task_name, error_metrics in metrics['error_by_task_name'].items():
                log_metrics[f'error/{task_name}'] = (error_metrics[0] / error_metrics[1]).item() # divide summed error by number of entries

            # add split name to keys
            log_metrics = {f'{split_name}/{k}': v for k, v in log_metrics.items()}

            return log_metrics
        
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
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

                        # compute loss
                        pred = self.model(batch)
                        metrics = compute_metrics(pred, batch)
                        raw_loss = metrics['raw_loss']
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        accelerator.backward(loss)

                        # log grad norms
                        if cfg.training.log_grad_norm_every != -1 and self.global_step % cfg.training.log_grad_norm_every == 0:
                            step_log['grad_norm'] = get_gradient_norm(self.model)

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
                        step_log.update(get_step_log(metrics, 'train'))
                        step_log.update({
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

                # at the end of each epoch add epoch average loss
                train_loss = np.mean(train_losses)
                step_log['train/loss/all_epoch'] = train_loss

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()
                self.model.eval()

                # run validation
                # This differs from training epochs in that we aggregate metrics across the entire epoch (compared to logging metrics for every batch in training). It also differes in that we aggregate metrics from batches across all processes, while in training we only use metrics from the batches in the main process.
                if (self.epoch % cfg.training.val_every) == 0 and len(val_dataloader) > 0:
                    with torch.inference_mode():
                        metrics_aggregated = {}
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                pred = self.model(batch)
                                gt = batch['labels']['progression']
                                task_idx = batch['metadata']['task_idx']

                                # gather preds and other items from batch that are relevant for computing metrics
                                pred, gt, task_idx = accelerator.gather_for_metrics((pred, gt, task_idx))
                                batch_for_metrics = {'metadata': {'task_idx': task_idx}, 'labels': {'progression': gt}}

                                metrics = compute_metrics(pred, batch_for_metrics)
                                aggregate_metrics_across_batch(metrics_aggregated, metrics)

                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        
                        if metrics_aggregated:
                            # add validation metrics to step log
                            step_log.update(get_step_log(metrics_aggregated, 'val'))
                    
                    visualize_progression(f'epoch_{local_epoch_idx}', is_train=False, include_video=False)

                # visualize train progression
                visualize_progression(f'epoch_{local_epoch_idx}', is_train=True, include_video=False)
                
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

        # end of script validation
        visualize_progression('final', is_train=True, include_video=True)
        visualize_progression('final', is_train=False, include_video=True)

        accelerator.print(f'Finished training. Run dir: {self.output_dir}')

        accelerator.end_training()
