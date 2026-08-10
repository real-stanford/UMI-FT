import hydra
import torch
import random
import copy
import numpy as np
from accelerate import Accelerator
from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from umi_day.train_network.utils.load_env import load_env_runner, env_rollout

OmegaConf.register_new_resolver("eval", eval, replace=True)

class RolloutPolicyWorkspace(BaseWorkspace):

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

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        accelerator = Accelerator(log_with='wandb', mixed_precision=cfg.training.mixed_precision)
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )
        accelerator.print(f'Started rollout. Run dir: {self.output_dir}')

        # load checkpoint
        if cfg.rollout.checkpoint_path:
            print(f"Going to evaluate checkpoint {cfg.rollout.checkpoint_path}")
            saved_output_dir = self.output_dir
            self.load_checkpoint(path=cfg.rollout.checkpoint_path)
            self._output_dir = saved_output_dir
        elif cfg.rollout.policy_only_checkpoint_path:
            print(f"Going to evaluate policy only checkpoint {cfg.rollout.policy_only_checkpoint_path}")
            checkpoint = torch.load(cfg.rollout.policy_only_checkpoint_path, map_location='cpu')
            
            if cfg.training.use_ema:
                self.ema_model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print("No checkpoint path provided, using random weights")

        # unwrap the policy from accelerator
        policy = accelerator.unwrap_model(self.model)
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        
        # load env runners
        print('Starting to load env runners')
        env_runners = load_env_runner(cfg, self.output_dir)
        print('Finished loading env runners')

        # put policy on GPU
        policy.to(accelerator.device)
        
        # rollout policy in env
        print('Starting to rollout policy')
        step_log = {}
        runner_log = env_rollout(cfg, env_runners, policy)
        # log all
        step_log.update(runner_log)
        accelerator.log(step_log, step=0)
        print(f'Rollout finished, step log: {step_log}')

        accelerator.end_training()
