# From https://github.com/real-stanford/detached-umi-policy

import sys
import os
import time
import click
import numpy as np
import torch
import dill
import hydra
import zmq

from umi_day.common import import_umi_source

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi_day.train_network.model.progression_net import ProgressionNet
from umi.real_world.real_inference_util import get_real_obs_resolution, get_real_umi_action
from diffusion_policy.common.pytorch_util import dict_apply
import omegaconf
import traceback
import subprocess

def echo_exception():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    # Extract unformatted traceback
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    # Print line of code where the exception occurred

    return "".join(tb_lines)
class PolicyInferenceNode:
    def __init__(self, ckpt_path: str, ip: str, port: int, device: str):
        self.ckpt_path = ckpt_path
        if not self.ckpt_path.endswith('.ckpt'):
            self.ckpt_path = os.path.join(self.ckpt_path, 'checkpoints', 'latest.ckpt')
        payload = torch.load(open(self.ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
        self.cfg = payload['cfg']
        # export cfg to yaml
        cfg_path = self.ckpt_path.replace('.ckpt', '.yaml')
        with open(cfg_path, 'w') as f:
            f.write(omegaconf.OmegaConf.to_yaml(self.cfg))
            print(f"[policy] Exported config to {cfg_path}")
        print(f"[policy] Loading configure: {self.cfg.name}, workspace: {self.cfg._target_}, policy: {self.cfg.model._target_}, model_name: {self.cfg.model.obs_encoder.model_name}")
        self.obs_res = get_real_obs_resolution(self.cfg.task.shape_meta)
        self.get_class_start_time = time.monotonic()

        cls = hydra.utils.get_class(self.cfg._target_)
        self.workspace = cls(self.cfg)
        self.workspace: BaseWorkspace
        self.workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        self.policy:BaseImagePolicy = self.workspace.model
        if self.cfg.training.use_ema:
            self.policy = self.workspace.ema_model
            print("[policy] Using EMA model")
        self.policy.num_inference_steps = 16
        
        obs_pose_rep = self.cfg.task.pose_repr.obs_pose_repr
        action_pose_repr = self.cfg.task.pose_repr.action_pose_repr
        print('[policy] obs_pose_rep', obs_pose_rep)
        print('[policy] action_pose_repr', action_pose_repr)
        
        self.device = torch.device(device)
        self.policy.eval().to(self.device)
        self.policy.reset()
        self.ip = ip
        self.port = port

    def predict_action(self, obs_dict_np: dict):
        with torch.no_grad():
            obs_dict = dict_apply(obs_dict_np, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
            result = self.policy.predict_action(obs_dict)
            action = result['action_pred'][0].detach().to('cpu').numpy()
            del result
            del obs_dict
        return action
    
    def run_node(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://{self.ip}:{self.port}")
        print(f"PolicyInferenceNode is listening on {self.ip}:{self.port}")
        while True:
            obs_dict_np = socket.recv_pyobj()
            try:
                start_time = time.monotonic()
                action = self.predict_action(obs_dict_np)
                print(f'Inference time: {time.monotonic() - start_time:.3f} s')
            except Exception as e:
                err_str = echo_exception()
                print(f'Error: {err_str}')
                action = err_str
            send_start_time = time.monotonic()
            socket.send_pyobj(action)
            print(f'Send time: {time.monotonic() - send_start_time:.3f} s')

class ProgressionInferenceNode:
    def __init__(self, ckpt_path: str, ip: str, port: int, device: str):
        self.ckpt_path = ckpt_path
        if not self.ckpt_path.endswith('.ckpt'):
            self.ckpt_path = os.path.join(self.ckpt_path, 'checkpoints', 'latest.ckpt')
        payload = torch.load(open(self.ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
        self.cfg = payload['cfg']
        # export cfg to yaml
        cfg_path = self.ckpt_path.replace('.ckpt', '.yaml')
        with open(cfg_path, 'w') as f:
            f.write(omegaconf.OmegaConf.to_yaml(self.cfg))
            print(f"[progression] Exported config to {cfg_path}")
        print(f"[progression] Loading configure: {self.cfg.name}, workspace: {self.cfg._target_}, model: {self.cfg.model._target_}, model_name: {self.cfg.model.obs_encoder.model_name}")
        self.obs_res = get_real_obs_resolution(self.cfg.task.shape_meta)
        self.get_class_start_time = time.monotonic()

        cls = hydra.utils.get_class(self.cfg._target_)
        self.workspace = cls(self.cfg)
        self.workspace: BaseWorkspace
        self.workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        self.model:ProgressionNet = self.workspace.model
        if self.cfg.training.use_ema:
            self.model = self.workspace.ema_model
            print("[progression] Using EMA model")
        
        obs_pose_rep = self.cfg.task.pose_repr.obs_pose_repr
        action_pose_repr = self.cfg.task.pose_repr.action_pose_repr
        print('[progression] obs_pose_rep', obs_pose_rep)
        print('[progression] action_pose_repr', action_pose_repr)
        
        self.device = torch.device(device)
        self.model.eval().to(self.device)
        self.ip = ip
        self.port = port

    def predict_progression(self, obs_dict_np: dict):
        with torch.no_grad():
            obs_dict = dict_apply(obs_dict_np, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
            obs_dict = {'obs': obs_dict}
            result = self.model(obs_dict)
            progression = result.detach().to('cpu').numpy()
            del result
            del obs_dict
        return progression
    
    def run_node(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://{self.ip}:{self.port}")
        print(f"ProgressionInferenceNode is listening on {self.ip}:{self.port}")
        while True:
            obs_dict_np = socket.recv_pyobj()
            try:
                start_time = time.monotonic()
                progression = self.predict_progression(obs_dict_np)
                print(f'Inference time: {time.monotonic() - start_time:.3f} s')
            except Exception as e:
                err_str = echo_exception()
                print(f'Error: {err_str}')
                progression = err_str
            send_start_time = time.monotonic()
            socket.send_pyobj(progression)
            print(f'Send time: {time.monotonic() - send_start_time:.3f} s')

def run_policy_inference_node(input, ip, port, device):
    node = PolicyInferenceNode(input, ip, port, device)
    node.run_node()

def run_progression_inference_node(input, ip, port, device):
    node = ProgressionInferenceNode(input, ip, port, device)
    node.run_node()
    
@click.command()
@click.option('--policy_input', '-i', required=False, help='Path to policy checkpoint')
@click.option('--progression_input', '-g', required=False, help='Path to progression checkpoint')
@click.option('--ip', default="0.0.0.0")
@click.option('--policy_port', default=8766, help="Port to listen on for policy inference")
@click.option('--progression_port', default=8764, help="Port to listen on for progression inference")
@click.option('--device', default="cuda", help="Device to run on")
def main(policy_input, progression_input, ip, policy_port, progression_port, device):
    # run policy inference and progression inference in separate processes (only do the fork if both are present)
    policy_present = policy_input is not None
    progression_present = progression_input is not None

    if policy_present and progression_present:
        child_pid = os.fork() # both policy and progression
    elif policy_present:
        child_pid = 0 # only policy
    elif progression_present:
        child_pid = 1 # only progression
    else:
        assert False, "At least one of policy_input or progression_input must be provided"

    if child_pid == 0:
        run_policy_inference_node(policy_input, ip, policy_port, device)
    else:
        run_progression_inference_node(progression_input, ip, progression_port, device)
                

if __name__ == '__main__':
    main()