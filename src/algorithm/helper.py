import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


__REDUCE__ = lambda b: 'mean' if b else 'none'


def l1(pred, target, reduce=False):
	"""Computes the L1-loss between predictions and targets."""
	return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))


def mse(pred, target, reduce=False):
	"""Computes the MSE loss between predictions and targets."""
	return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))


def orthogonal_init(m):
	"""Orthogonal layer initialization."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if m.bias is not None:
			nn.init.zeros_(m.bias)

def ema(m, m_target, tau):
	"""Update slow-moving average of online network (target network) at rate tau."""
	with torch.no_grad():
		# zip 会把两个网络对应的参数成一对一对的元组，然后把对应的参数组优化
		for p, p_target in zip(m.parameters(), m_target.parameters()):
			# TOLD step 19
			# lerp_ 其实是 PyTorch 自带的张量方法;Tensor 都有一个 .data 属性，表示它的 原始张量数据，不包含梯度信息
			# p_target.data = (1 - tau) * p_target.data + tau * p.data
			# .lerp_ 末尾的 _ 表示 in-place 原地修改，参数直接改了，不需要返回
			p_target.data.lerp_(p.data, tau)


def set_requires_grad(net, value):
	"""Enable/disable gradients for a given (sub)network."""
	for param in net.parameters():
		param.requires_grad_(value)


class TruncatedNormal(pyd.Normal): 	# 实现截断正态分布（Truncated Normal Distribution）的采样
	"""Utility class implementing the truncated normal distribution."""
	def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
		super().__init__(loc, scale, validate_args=False)
		self.low = low # 保存下界
		self.high = high # 保存上界
		self.eps = eps	# 保存边界偏移量

	def _clamp(self, x):
		# 私有方法，将输入 x 限制在 [low + eps, high - eps] 范围内
        # x: 输入张量（采样值）
		clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
		# x.detach() 移除梯度，clamped_x.detach() 移除梯度后的值，保持 x 的计算图
		x = x - x.detach() + clamped_x.detach()
		return x

	def sample(self, clip=None, sample_shape=torch.Size()): # 生成并返回一个截断正态分布（Truncated Normal Distribution）的样本
        # clip: 可选裁剪范围，限制标准正态分布的输出
        # sample_shape: 采样张量的形状
		shape = self._extended_shape(sample_shape)
		# eps 不是均值，它是从标准正态分布采样得到的随机噪声
		# _standard_normal 生成标准正态分布（均值=0，标准差=1）的随机样本张量,返回形状为 shape 的随机张量
		eps = _standard_normal(shape, 
							   dtype=self.loc.dtype,
							   device=self.loc.device)
		# scale：正态分布的标准差（standard deviation）；self.scale = std, self.loc = mu
		eps *= self.scale # 缩放噪声样本到标准差为 self.scale
		if clip is not None:
			eps = torch.clamp(eps, -clip, clip)
		# self.loc 是目标正态分布的均值，x 是最终的正态分布样本
		# 将缩放后的噪声加上均值，得到正态分布样本
		x = self.loc + eps 
		# # 限制样本到 [low + eps, high - eps] 范围内
		return self._clamp(x)


def enc(cfg):
	"""Returns a TOLD encoder."""
	layers = [nn.Linear(cfg.obs_shape[0], cfg.enc_dim), nn.ELU(), # obs.shape = (batch_size, obs_dim)
				  nn.Linear(cfg.enc_dim, cfg.latent_dim)]
	return nn.Sequential(*layers)


def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
	"""Returns an MLP."""
	if isinstance(mlp_dim, int):
		mlp_dim = [mlp_dim, mlp_dim]
	# x.shape = (batch_size, latent_dim+action_dim) ;in_dim = cfg.latent_dim + cfg.action_dim; mlp_dim = cfg.mlp_dim
	return nn.Sequential(
		nn.Linear(in_dim, mlp_dim[0]), act_fn,  # 输入层 是(B, in_dim)
		nn.Linear(mlp_dim[0], mlp_dim[1]), act_fn,
		nn.Linear(mlp_dim[1], out_dim)) 		# 输出：(B, out_dim) ; out_dim = cfg.latent_dim

def q(cfg, act_fn=nn.ELU()):
	"""Returns a Q-function that uses Layer Normalization."""
	# x.shape(batch_size,latent_dim+action_dim)
	return nn.Sequential(nn.Linear(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim), nn.LayerNorm(cfg.mlp_dim), nn.Tanh(),
						 nn.Linear(cfg.mlp_dim, cfg.mlp_dim), nn.ELU(),
						 nn.Linear(cfg.mlp_dim, 1))


class Episode(object):
	"""Storage object for a single episode."""
	def __init__(self, cfg, init_obs):
		self.cfg = cfg
		self.device = torch.device(cfg.device)
		num_envs = init_obs.shape[0]
		obs_dim = init_obs.shape[1]
		dtype = torch.float32 if cfg.modality == 'state' else torch.uint8
		# obs: (T+1, B, obs_dim)
		# 这里之所以是T+1而不是T 因为obs[0] 是 init_obs，obs[1] 是执行第 0 个 action 后的 next_obs
		self.obs = torch.empty((cfg.episode_length+1, num_envs, obs_dim), dtype=dtype, device=self.device)
		self.obs[0] = torch.tensor(init_obs, dtype=torch.float32, device=self.device)
		# action: (T, B, action_dim)
		self.action = torch.empty((cfg.episode_length, num_envs, cfg.action_dim), dtype=torch.float32, device=self.device)
   		# reward: (T, B)
		self.reward = torch.empty((cfg.episode_length, num_envs), dtype=torch.float32, device=self.device)
		self.cumulative_reward = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
		self.done = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
		self._idx = 0
	
	def __len__(self):
		return self._idx

	@property
	def first(self):
		return len(self) == 0
	
	def __add__(self, transition):
		# obs.shape = (num_envs, 7)， reward.shape = (num_envs,), done.shape = (num_envs,)
		# transition = (obs, action, reward, done)
		self.add(*transition) 
		return self

	def add(self, obs, action, reward, done):
     	# obs: (B, obs_dim), action: (B, act_dim), reward: (B,), done: (B,)
		# self.obs:(L,num_envs,obs_dim), self.action:(L, num_envs, actionim), self.done:(L,num_envs,)
      	# self._idx+1表示从1开始的，因为0拿去存init_obs了，而idx=1存的是action 0执行后的obs
		self.obs[self._idx + 1] = obs.detach().to(device=self.obs.device, dtype=self.obs.dtype)
		self.action[self._idx] = action
		self.reward[self._idx] = reward
		self.cumulative_reward += reward # reward: (B,)
		self.done = done
		self._idx += 1 # self._idx 表示当前 episode 已经写入了多少 step,最后self._idx == cfg.episode_length


class ReplayBuffer():
	"""
	Storage and sampling functionality for training TD-MPC / TOLD.
	The replay buffer is stored in GPU memory when training from state.
	Uses prioritized experience replay by default."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device(cfg.device)
		# 把num_envs放到capacity 相当于展平
		base_capacity = min(cfg.train_steps, cfg.max_buffer_size) * cfg.num_envs
		# base_capacity // cfg.episode_length 算出恰好能存多少条完整的轨迹，多余的不要
		self.capacity = (base_capacity // cfg.episode_length) * cfg.episode_length
  
		# (self.capacity+1, obs_dim)
		self._obs = torch.empty((self.capacity+1, *cfg.obs_shape), dtype=torch.float32, device=self.device)
		# (self.capacity//cfg.episode_length,obs_dim)
		self._last_obs = torch.empty((self.capacity//cfg.episode_length, *cfg.obs_shape), dtype=torch.float32, device=self.device)
		# (self.capacity, action_dim)
		self._action = torch.empty((self.capacity, cfg.action_dim), dtype=torch.float32, device=self.device)
		# (self.capacity,)
		self._reward = torch.empty((self.capacity,), dtype=torch.float32, device=self.device)
		# (self.capacity,)
		self._priorities = torch.ones((self.capacity,), dtype=torch.float32, device=self.device)
		self._eps = 1e-6
		self._full = False
		self.idx = 0

	def __add__(self, episode: Episode):
		self.add(episode)
		return self

	def add(self, episode: Episode):
		"""
        episode.obs:   (T+1, B, *obs_dim)
        episode.action:(T,   B, act_dim)
        episode.reward:(T,   B)
        
        Prioritized Replay Buffer (PER)。
		PER 的核心：每条 transition 都有一个 priority (优先级)，采样的时候概率 ∝ priority。
		新数据 → 模型从没见过 → 应该优先采样。
		老数据 → priority 可能下降。
		因此，刚写入 Replay Buffer 的 transition，要给比较高的优先级
        """
		# 新数据的初始优先级 要尽量高，这样它有更大概率被采样出来（因为模型还没学过它）。
		if self._full:
			max_priority = self._priorities.max().to(self.device).item()
		else:
			max_priority = 1. if self.idx == 0 else self._priorities[:self.idx].max().to(self.device).item()
		mask_tail = torch.arange(self.cfg.episode_length, device=self.device) >= (self.cfg.episode_length - self.cfg.horizon)
		new_prios_one_env = torch.full((self.cfg.episode_length,), max_priority, device = self.device)
		new_prios_one_env[mask_tail] = 0
  
		# 逐个env写入: 每个env占据连续self.cfg.episode_length个槽位
		for env in range(self.cfg.num_envs):
			start = self.idx
			end = start + self.cfg.episode_length
			# episode.obs.shape = (L,num_envs,obs_dim) 存储的是一条完整的轨迹所以取全部的length
			self._obs[start:end] = episode.obs[:-1, env].to(self._obs.dtype)
			# 存每个轨迹最后一步 obs
			# 为什么要把最后一个obs单独存放：(init_obs,a0)->(o1,a1)->(o2,a2)->o3
			# 因为配对的问题，_obs=[init_obs, o1, o2]必须与_action[start:end] = [a0, a1, a2]对齐,_last_obs=o3它是最后一个action的next_obs要单独放
			# start//self.cfg.episode_length 这里取整表示第n个episode，把对应的last_obs存入对应第n个位置处
			# episode.obs.shape = (T+1, num_envs, obs_dim)
			self._last_obs[start//self.cfg.episode_length] = episode.obs[-1, env].to(self._last_obs.dtype)
			# episode.action.shape = (T, B, action_dim); self._action.shape=(self.capacity, action_dim)
			self._action[start:end] = episode.action[:, env]
			# episode.reward.shape = (T, B); self._reward,shape = (self.capacity,)
			self._reward[start:end] = episode.reward[:, env]
			# 给这一段数据的优先级赋值
			self._priorities[start:end] = new_prios_one_env
			# 移动 buffer 写入位置； % self.capacity 超过capacity时，会回到buffer最开头
			# self.idx + self.cfg.episode_length 表示存完一整个episode后idx移动episode作为下一个完整的episode的起点
			self.idx = (self.idx + self.cfg.episode_length) % self.capacity
			self._full = self._full or (self.idx == 0)
	
	def update_priorities(self, idxs, priorities):
		# priority.shape = (Batch_size, 1) --> (Batch_size,)
		self._priorities[idxs] = priorities.squeeze(1).to(self.device) + self._eps

	def sample(self):
		"""
		Horizon = MPC 的规划窗口长度，是“在 episode 的某个时刻 t 向前看多少步
		Episode: [s0, a0, r0, s1, a1, r1, ..., s_T]   (T = episode_length)
		在某个时间 t，MPC horizon=5：
		使用 [s_t] 作为起点，预测未来 5 步轨迹 → [a_t, a_{t+1}, ..., a_{t+4}],只执行 a_t,下一步再滚动 horizon
		"""
		probs = (self._priorities if self._full else self._priorities[:self.idx]) ** self.cfg.per_alpha
		probs /= probs.sum()
		total = len(probs)
		# idxs.shape = (batch_size,) 长度为Batch_size的数组
		idxs = torch.from_numpy(np.random.choice(total, self.cfg.batch_size, p=probs.cpu().numpy(), replace=not self._full)).to(self.device)
		# probs[idxs] 按照 idxs 从 probs 中挑出 batch_size 个概率 → probs[idxs].shape = (batch_size,)
		weights = (total * probs[idxs]) ** (-self.cfg.per_beta)
		weights /= weights.max()
  
		# idxs.shape = (batch_size,) --->self._obs[idxs] → 取多个索引，shape 会变成 (len(idxs), obs_dim)
		# _obs.shape = (self.capacity+1, obs_dim)； obs.shape = (batch_size, obs_dim)
		obs = self._obs[idxs] # 这里 obs 是rollout 的起点状态，也就是在 t=0 时刻的观测
		# next_obs_shape = obs_dim
		next_obs_shape = self._obs.shape[1:] 
		# next_obs.shape = (H+1, Batch_size,obs_dim)
		next_obs = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, *next_obs_shape), dtype=obs.dtype, device=obs.device)
		# action.shape = (H, Batch_size, action_dim) ; self._action.shape=(self.capacity, action_dim)
		action = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, *self._action.shape[1:]), dtype=torch.float32, device=self.device)
		# reward.shape = (H, Batch_size)
		reward = torch.empty((self.cfg.horizon+1, self.cfg.batch_size), dtype=torch.float32, device=self.device)
		# Horizon = MPC 的规划窗口长度，是“在 episode 的某个时刻 t 向前看多少步
		for t in range(self.cfg.horizon+1):
			_idxs = idxs + t
			# _obs.shape = (self.capacity+1, obs_dim) ; next_obs.shape = (H+1, Batch_size,obs_dim)
			# obs = s_t 单独存 ; next_obs = [s_t, s_{t+1}, ..., s_{t+H}] 也就是说Obs只存了起点，next_obs存了起点+Horizon
			
			# 有bug
			next_obs[t] = self._obs[_idxs + 1]




			# self._action.shape=(self.capacity, action_dim);  action.shape = (H, Batch_size, action_dim)
			# action = [a_t, ..., a_{t+H-1}]
			action[t] = self._action[_idxs]
			# self._reward = (self.capacity,) ; reward.shape = (H, Batch_size)
			reward[t] = self._reward[_idxs]

		# 创建一个布尔掩码 (mask)，检查轨迹的下一步索引 (_idxs + 1) 是否正好是 episode 边界的起点
		mask = (_idxs+1) % self.cfg.episode_length == 0


		# 有bug
		next_obs[-1, mask] = self._last_obs[(_idxs[mask]) // self.cfg.episode_length].to(next_obs.dtype).to(self.device)
		# next_obs[-1, mask] = self._last_obs[_idxs[mask]//self.cfg.episode_length].cuda().float()
		
		
		if not action.is_cuda:
			action, reward, idxs, weights = action.cuda(), reward.cuda(), idxs.cuda(), weights.cuda()
		# obs.shape = (batch_size, obs_dim);  next_obs.shape = (H+1, Batch_size,obs_dim);
  		# action.shape = (H, Batch_size, action_dim) ;reward.shape = (H, Batch_size,1); weight.shape(batch_size,)
		return obs, next_obs, action, reward.unsqueeze(2), idxs, weights


def linear_schedule(schdl, step):
	"""
	Outputs values following a linear decay schedule.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	try: # 尝试将 schdl 转换为浮点数。schdl = "linear(1.0, 0.0, 10)" init=1.0, final=0.0, duration=10-->10个步骤从0衰减到10
		return float(schdl)
	 # 如果转换失败（ValueError），则检查是否是线性调度格式。
	except ValueError:
		# 使用正则表达式匹配 'linear(init,final,duration)' 格式的字符串。
        # match.groups() 会提取 init、final、duration 的值。
		match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
		if match: # 如果匹配成功，提取并转换为浮点数
			init, final, duration = [float(g) for g in match.groups()] # init=1.0, final=0.0, duration=10-->10个步骤从0衰减到10
			# 计算混合比例 mix = step / duration，夹在 [0,1] 之间。
            # 这决定了从 init 到 final 的线性插值进度。
			mix = np.clip(step / duration, 0.0, 1.0)
			# 返回线性插值结果：(1 - mix) * init + mix * final
            # 当 step=0 时，返回 init；step>=duration 时，返回 final
			return (1.0 - mix) * init + mix * final # 刚开始return 是init 后面return final 可以理解成刚开始是探索 后面变成利用
	# 如果 schdl 不是数字也不是线性格式，抛出异常。
	raise NotImplementedError(schdl)
