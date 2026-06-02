import sys
import os
import datetime
import re
import numpy as np
import torch
import pandas as pd
from termcolor import colored
from omegaconf import OmegaConf
from zoneinfo import ZoneInfo
import imageio
#from datetime import datetime


CONSOLE_FORMAT = [('episode', 'E', 'int'), ('env_step', 'S', 'int'), ('episode_reward', 'R', 'float'), ('total_time', 'T', 'time')]
AGENT_METRICS = ['consistency_loss', 'reward_loss', 'value_loss', 'total_loss', 'weighted_loss', 'pi_loss', 'grad_norm']


def make_dir(dir_path):
	"""Create directory if it does not already exist."""
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def print_run(cfg, reward=None):
	"""Pretty-printing of run information. Call at start of training."""
	prefix, color, attrs = '  ', 'green', ['bold']
	def limstr(s, maxlen=32):
		return str(s[:maxlen]) + '...' if len(str(s)) > maxlen else s
	def pprint(k, v):
		print(prefix + colored(f'{k.capitalize()+":":<16}', color, attrs=attrs), limstr(v))
	kvs = [('task', cfg.task_title),
		   ('train steps', f'{int(cfg.train_steps*cfg.action_repeat):,}'),
		   ('observations', 'x'.join([str(s) for s in cfg.obs_shape])),
		   ('actions', cfg.action_dim),
		   ('experiment', cfg.exp_name)]
	if reward is not None:
		kvs.append(('episode reward', colored(str(int(reward)), 'white', attrs=['bold'])))
	w = np.max([len(limstr(str(kv[1]))) for kv in kvs]) + 21
	div = '-'*w
	print(div)
	for k,v in kvs:
		pprint(k, v)
	print(div)


def cfg_to_group(cfg, return_list=False):
	"""Return a wandb-safe group name for logging. Optionally returns group name as list."""
	lst = [cfg.task, cfg.modality, re.sub('[^0-9a-zA-Z]+', '-', cfg.exp_name)]
	return lst if return_list else '-'.join(lst)


class VideoRecorder:
	"""Utility class for logging evaluation videos."""
	def __init__(self, root_dir, wandb, render_size=384, fps=15):
		self.save_dir = (root_dir / 'eval_video') if root_dir else None
		self._wandb = wandb
		self.render_size = render_size
		self.fps = fps
		self.frames = []
		self.enabled = False

	def init(self, env, enabled=True):
		self.frames = []
		self.enabled = self.save_dir and enabled
		self.record(env)

	def record(self, env):
		if self.enabled:
			frame = env.render(recompute=False)
			self.frames.append(frame)

	def save(self, step):
		out = self.save_dir / f"step_{int(step)}.mp4"
		# imageio 需要 (T,H,W,3)
		frames_hwc = np.stack(self.frames, axis=0)
		imageio.mimsave(out, frames_hwc, fps=self.fps)
		print(f"[VideoRecorder] Saved video to: {out}")
		# 可选：清空缓存，避免占内存
		self.frames = []


class Logger(object):
	"""Primary logger object. Logs either locally or using wandb."""
	def __init__(self, log_dir, cfg):
		self._log_dir = make_dir(log_dir) # "logs"/ algo_cfg.exp_name / str(algo_cfg.seed)
		self._model_dir = make_dir(self._log_dir / 'models') # "logs"/ algo_cfg.exp_name / str(algo_cfg.seed)/models
		self._save_model = cfg.save_model
		self._group = cfg_to_group(cfg)
		self._seed = cfg.seed
		self._cfg = cfg
		self._eval = []
		print_run(cfg)
		
		
		print(colored('Logs will be saved locally.', 'yellow', attrs=['bold']))
		self._wandb = None
		self._video = VideoRecorder(log_dir, self._wandb) if cfg.save_video else None

	@property
	def video(self):
		return self._video

	def finish(self, agent):
		if self._save_model:
			now = datetime.datetime.now(ZoneInfo("Europe/Berlin"))
			stamp = now.strftime("%Y-%m-%d_%H-%M-%S") 
			fp = self._model_dir / f'{stamp}.pt'
			torch.save(agent.state_dict(), fp)

		# 如果没有 eval 数据就跳过打印
		if len(self._eval) > 0 and len(self._eval[-1]) > 0:
			print_run(self._cfg, self._eval[-1][-1])
		else:
			print("[INFO] No evaluation results to print at finish.")
		import os, sys, socket
		print("[whoami]", os.geteuid(), os.getlogin() if hasattr(os, "getlogin") else "n/a")
		print("[host]", socket.gethostname())
		print("[cwd]", os.getcwd())
		print("[python]", sys.executable)
		print("[model path]", fp)
		print("[exists right after save?]", fp.exists())


	def _format(self, key, value, ty):
		# 如果是 tensor，先取标量
		if isinstance(value, torch.Tensor):
			if value.numel() == 1:  # 只有一个元素
				value = value.item()
			else:
				value = value.detach().cpu().numpy()  # 多元素 tensor 转 numpy 数组

		if ty == 'int':
			return f'{colored(key+":", "grey")} {int(value):,}'
		elif ty == 'float':
			return f'{colored(key+":", "grey")} {float(value):.1f}'  # 转 float
		elif ty == 'time':
			value = str(datetime.timedelta(seconds=int(value)))
			return f'{colored(key+":", "grey")} {value}'
		else:
			raise ValueError(f'invalid log format type: {ty}')


	def _print(self, d, category):
		category = colored(category, 'blue' if category == 'train' else 'green')
		pieces = [f' {category:<14}']
		for k, disp_k, ty in CONSOLE_FORMAT:
			pieces.append(f'{self._format(disp_k, d.get(k, 0), ty):<26}')
		print('   '.join(pieces))

	def log(self, d, category='train'):
		assert category in {'train', 'eval'}
		if category == 'eval':
			keys = ['env_step', 'episode_reward']
			self._eval.append(np.array([d[keys[0]], d[keys[1]]]))
			pd.DataFrame(np.array(self._eval)).to_csv(self._log_dir / 'eval.log', header=keys, index=None)
		self._print(d, category)
