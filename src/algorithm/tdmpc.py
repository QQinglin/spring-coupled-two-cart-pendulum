import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import algorithm.helper as h


class TOLD(nn.Module):
	"""Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._encoder = h.enc(cfg)
		self._dynamics = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)
		self._reward = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, 1)
		self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
		self._Q1, self._Q2 = h.q(cfg), h.q(cfg)
		self.apply(h.orthogonal_init)
		for m in [self._reward, self._Q1, self._Q2]:
			m[-1].weight.data.fill_(0)
			m[-1].bias.data.fill_(0)

	def track_q_grad(self, enable=True):
		"""Utility function. Enables/disables gradient tracking of Q-networks."""
		for m in [self._Q1, self._Q2]:
			h.set_requires_grad(m, enable)

	def h(self, obs):
		"""Encodes an observation into its latent representation (h)."""
		return self._encoder(obs) # 输入：obs.shape = (batch_size, obs_dim) 输出： (batch_size, latent_dim)

	def next(self, z, a):
		"""Predicts next latent state (d) and single-step reward (R)."""
		# z.shape =  (num_envs*num_pi_trajs, latent_dim) ; action.shap = [num_envs*num_pi_trajs,action_dim]
		# x.shape = (num_envs*num_pi_trajs, latent_dim + action_dim)
		x = torch.cat([z, a], dim=-1)
		# self._dynamic.shape = (num_envs*num_pi_trajs, latent_dim)；_reward.shape = (num_envs*num_pi_trajs,1)
		return self._dynamics(x), self._reward(x) 

	def pi(self, z, std=0):
		"""Samples an action from the learned policy (pi)."""
		# mu.shape = (batch_size, action_dim); z.shape=(num_envx*Num, latent)这里网络的input是z
		# mlp网络的输入是latent_dim,输出是action_dim
		mu = torch.tanh(self._pi(z))
		if std > 0:
			std = torch.ones_like(mu) * std
			# 返回值: 一个张量，包含服从截断正态分布的样本，值在 [-1.0 + eps, 1.0 - eps] 范围内，且噪声被裁剪；限制噪声范围为 [-0.3, 0.3]
			return h.TruncatedNormal(mu, std).sample(clip=0.3)
		# 直接返回不带噪声的均值
		return mu

	def Q(self, z, a): 
		"""Predict state-action value (Q)."""
		# z.shape=(batch_size, latent_dim), a.shape=(batch_size, action_dim)
		# x.shape(batch_size,latent_dim+action_dim)
		x = torch.cat([z, a], dim=-1)
		# Q1.shape = (B,1) ; Q2.shape = (B,1)
		return self._Q1(x), self._Q2(x)


class TDMPC():
	"""Implementation of TD-MPC learning + inference."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device('cuda')
		self.std = h.linear_schedule(cfg.std_schedule, 0)
		self.model = TOLD(cfg).cuda()
		self.model_target = deepcopy(self.model)
		self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)
		self.model.eval()
		self.model_target.eval()

	def state_dict(self):
		"""Retrieve state dict of TOLD model, including slow-moving target network."""
		return {'model': self.model.state_dict(),
				'model_target': self.model_target.state_dict()}

	def save(self, fp):
		"""Save state dict of TOLD model to filepath."""
		torch.save(self.state_dict(), fp)
	
	def load(self, fp):
		"""Load a saved state dict from filepath into current agent."""
		d = torch.load(fp)
		self.model.load_state_dict(d['model'])
		self.model_target.load_state_dict(d['model_target'])

	@torch.no_grad()
	def estimate_value(self, z, actions, horizon): 
		"""Estimate value of a trajectory starting at latent state z and executing given actions.
  		z_flat.shape=(num_envs*(num_samples+num_pi_trajs), latent_dim)
		actions_flat.shape=(H, B*N, A) 
    	"""
		G, discount = 0, 1
		# TD MPC step 6， actions shape: (H, N, A)
		for t in range(horizon):
			# z.shape = (num_envs*num_pi_trajs,latent_dim); reward.shape = (num_envs*num_pi_trajs, 1)
			z, reward = self.model.next(z, actions[t]) # TD MPC step 7 and 8
			# G = (num_envs*num_pi_trajs, 1)
			G += discount * reward
			discount *= self.cfg.discount
		# TD MPC step9
		# G += discount * torch.min(*self.model.Q(z, actions[-1])); G = (num_envs*num_pi_trajs, 1)
		G += discount * torch.min(*self.model.Q(z, self.model.pi(z, self.cfg.min_std)))
		# G = (num_envs*num_pi_trajs, 1)
		return G

	@torch.no_grad()
	def plan(self, obs, eval_mode=False, step=None, t0=True):
		"""
		Plan next action using TD-MPC inference.
		obs: raw input observation.
		eval_mode: uniform sampling and action noise is disabled during evaluation.
		step: current time step. determines e.g. planning horizon.
		t0: whether current step is the first step of an episode.
		在191行 从num_samples+num_pi_trajs 这么多轨迹中根据value值 给每个环境选出了最优的K条轨迹对应的elite_value 和elite_action；
		在220算出K条轨迹各自的得分即被选中的概率P
		在228行 根据概率P给每个环境选出最优轨迹得到它的索引值 即
		在233行 根据索引值 选出轨迹对应的action轨迹
		在237行 从action轨迹中取第一个时间点的action 作为最终的action
		"""
		# Seed steps
		# seed_steps 步里，干脆不依赖策略网络，直接随机选动作，用来快速填充 replay buffer，增加数据多样性,把数据放到GPU
		if step < self.cfg.seed_steps and not eval_mode:
			return torch.empty((self.cfg.num_envs, self.cfg.action_dim), dtype=torch.float32, device=self.device).uniform_(-1, 1)

		# Sample policy trajectories
		# obs = (num_envs, obs_dim)
		# obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
  		
		# horizon是变化的，训练初期 horizon = 1（规划只看一步，探索为主），训练到 1e5 步之后 horizon = 5（规划更长远，利用为主）
		# horizon = init + [(final - init)/duration] * step 这是init 到final 的线性插值，step控制进度
		horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
		# self.cfg.mixture_coef=0.3 假设第三步需要100(self.cfg.num_samples)条轨迹，那么第四步步可能采样70条用于explore，第三步100条是exploite
		num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)

		 # (TD-MPC step4)
		if num_pi_trajs > 0:
			# dim = (num_envs,H,N,D)
			pi_actions = torch.empty(self.cfg.num_envs, horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)

			# obs = (num_envs, obs_dim)， z.shape = (num_envs,latent_dim)
			z = self.model.h(obs)
			# 给每个环境复制了 num_pi_trajs 条初始 latent state; (num_envs,1,L)-->(num_envs, N, L); -1表示维度不变
			z = z.unsqueeze(1).expand(-1, num_pi_trajs, -1)  
			for t in range(horizon):
				# min_std用来在采样动作时控制高斯策略的最小方差，避免策略在训练初期或某些状态下变得过于确定性（std 接近 0）而丧失探索能力
				# z.reshape(-1, z.shape[-1]).shape = (num_envx*Num, latent); action_t.shape = (num_envs * num_pi_trajs, action_dim)
				action_t = self.model.pi(z.reshape(-1, z.shape[-1]), self.cfg.min_std)
				# pi_actions.shape = (num_envs, horizon, num_pi_trajs, action_dim)；pi_actions[:, t].shape=(num_envs, num_pi_trajs, action_dim)
				pi_actions[:, t] = action_t.view(self.cfg.num_envs, num_pi_trajs, -1)
				# z.reshape(-1, z.shape[-1])=(num_envs*num_pi_trajs, L); pi_actions[:, t].reshape=(num_envs*num_pi_trajs, action_dim)
				# z.shape = (num_envs*num_pi_trajs,latent_dim)
				z, _ = self.model.next(z.reshape(-1, z.shape[-1]), pi_actions[:, t].reshape(-1, action_t.shape[-1]))
				# z.shape = (num_envs, num_pi_trajs,latent_dim)
				z = z.view(self.cfg.num_envs, num_pi_trajs, -1)
    
		# Initialize state and parameters (TD-MPC step 3)
		# z.shape = (num_envs,latent_dim)
		z = self.model.h(obs)
		# z.shape = (num_envs, num_samples+num_pi_trajs, latent_dim)
		z = z.unsqueeze(1).expand(-1, self.cfg.num_samples+num_pi_trajs, -1)
		# mean.shape == (env_nums, H, A)
		mean = torch.zeros(self.cfg.num_envs, horizon, self.cfg.action_dim, device=self.device)
		std = 2*torch.ones(self.cfg.num_envs, horizon, self.cfg.action_dim, device=self.device)
		if not t0 and hasattr(self, '_prev_mean'):
			mean[:, :-1] = self._prev_mean[:, 1:]

		# Iterate CEM (TD MPC step2)
		for i in range(self.cfg.iterations):
      		# “均值 + 标准差 * 随机噪声”（即 μ + σ * ε，其中 ε ~ N(0, 1)）高斯分布采样的标准方式 TD MPC(step3)
			# mean: (env_nums, H, A) ---> std.unsqueeze(2) (env_nums,  H, N, A), dim(actions) = 4 
			# std.unsqueeze(2) 因为我们想要的action(B,H,N,A)所以增加一个维度--> (env_nums, H,1,A) 会在第 2 维自动广播到 (env_nums,H,N,A)，与 randn 做逐元素乘法
			# mean.unsqueeze(2).shape = (env_nums, H, 1, A); 这里应该repeat(2,num_samples)嘛？
   			# actions.shape == (env_nums, H, num_samples, A)
			actions = torch.clamp(mean.unsqueeze(2) + std.unsqueeze(2) * \
				torch.randn(self.cfg.num_envs, horizon, self.cfg.num_samples, self.cfg.action_dim, device=std.device), -1, 1)
			if num_pi_trajs > 0:
				# actions shape: (num_envs, H, num_samples, A). pi_actions shape: (num_envs, H, N_pi, A). cat(dim=1) -> (H, N+N_pi, A)
				# (B, H, num_samples+num_pi_trajs, A)刚好于z匹配 z.shape = (num_envs, num_samples+num_pi_trajs, latent_dim)
				actions = torch.cat([actions, pi_actions], dim=2)

			# Compute elite actions (TDMPC step 9)
			# z.shape = (num_envs, num_samples+num_pi_trajs, latent_dim)； z_flat.shape=(num_envs*(num_samples+num_pi_trajs), latent_dim)
			z_flat = z.reshape(-1, z.shape[-1]) # 保持列数 ，其他计算填充
			# permute作用是把H从第一维挪到第0维 (B, H, N, A) → actions_flat.shape=(H, B*N, A) 
			actions_flat = actions.permute(1, 0, 2, 3).reshape(horizon, -1, actions.shape[-1]) 
			# value.shape = (num_envs*num_samples+num_pi_trajs, 1)
			value = self.estimate_value(z_flat, actions_flat, horizon).nan_to_num_(0)
			# (num_envs, num_samples+num_pi_trajs, 1) 每个轨迹对应一个Value
			value = value.view(self.cfg.num_envs, -1, 1) 
   			# top-k值，indices 是它们在原张量中的位置 N_total = num_samples+num_pi_trajs
			# value.shape = (B, N_total, 1) --> value.squeeze(2) = (B, N_total) 在N_total这个维度上选出k个最大值返回他的索引
			# elite_idxs.shape = (B, K)
			elite_idxs = torch.topk(value.squeeze(2), self.cfg.num_elites, dim=1).indices
			# elite_idxs.unsqueeze(-1) =  (B, K, 1) ; value.shape = (B, N_total, 1)
			# 表示沿着dim = 1 即N_total方向 按照索引elite_indx 的k 取出对应位置的值 所以结构elite_value.shape = (B,K,1)
			elite_value = torch.gather(value, 1, elite_idxs.unsqueeze(-1))   
			# elite_idxs.unsqueeze(1).unsqueeze(-1).shape = (B, 1, K, 1) → 
   			# expand(-1, actions.shape[1], -1, actions.shape[-1])表示把dim=1和dim=3分别复制H遍和A遍 到 (B, H, K, A)
			# actions.shape = (B, H, N_total, A) 沿着dim=2 即N_total方向取值， 索引是 (B, H, K, A) 中的K 
   			# 所以结果 elite_actions.shape = (B, H, top_K, A)
			elite_actions = torch.gather(actions, 2, elite_idxs.unsqueeze(1).unsqueeze(-1).expand(-1, actions.shape[1], -1, actions.shape[-1]))
   
			# Update parameters TD MPC step 9
			# shape = (num_envs, top_k, 1) # 在top_k这个维度上取最大值 shape=[num_envs, 1, 1]
			max_value, _ = elite_value.max(dim=1, keepdim=True)
			# Ω  score.shape=(num_envs, top_k, 1); elite_value = (B, K, 1) ; max_value = [num_envs, 1, 1]
			score = torch.softmax(self.cfg.temperature * (elite_value - max_value), dim=1) 
			# 沿着top_k方向求和 score.shape=(num_envs, top_k, 1)
			score_sum = score.sum(1, keepdim=True) + 1e-9
			# μ [num_envs,H,k,A]; score.unsqueeze(1).shape = (num_envs, 1, top_k, 1) 对齐 elite_actions.shape = (B, H, K, A)
			_mean = torch.sum(score.unsqueeze(1) * elite_actions, dim=2) / score_sum 
			# σ [num_envs,H,k,A]
			_std = torch.sqrt(torch.sum(score.unsqueeze(1) * (elite_actions - _mean.unsqueeze(2)) ** 2, dim=2) / score_sum)
			# clamp 裁掉 A 维度上的值
			_std = _std.clamp_(self.std, 2)
			mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

		# Outputs TD MPC step 10
  		# score.shape=(num_envs, top_k, 1)-->(num_envs, top_k),top_k表示k条中每一条被选中的概率，sum(top_k)=1，所以只会选一条最终
		score = score.squeeze(-1).cpu().numpy()
		# elite_actions.shape = (B, H, top_K, A) 每个env 只在top_k中选取1条，所以selected_idxs.shape = (num_envs,), sum(top_k)=1
		selected_idxs = np.zeros(self.cfg.num_envs, dtype=int)
		for env in range(self.cfg.num_envs):
			 # score.shape=(num_envs, top_k); probability.shape=(1,top_k), sum(top_k)=1; score 表示top_k中每一条被选中的概率
			probablity = score[env]
			# k条 所以先np.arange(k)=[0,1,2...,k], p=[0.1,0.2...] sum(p)=1 --> selected_idxs.shape = (num_envs,)
			selected_idxs[env] = np.random.choice(np.arange(score.shape[1]), p=probablity)
		# selected_idxs.shape = (num_envs,)-->(num_envs,1,1,1)-->(num_envs, H, 1, A)
		selected_idxs = torch.tensor(selected_idxs, device=elite_actions.device).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, elite_actions.shape[1], -1, elite_actions.shape[-1])
		# elite_actions.shape = (B, H, K, A) 在dim=2的维度上根据 selected_idxs.shape=(num_envs, H, 1, A)索引提取对应位置的action
		# actions.shape=(B, H, A) K条轨迹的K个action只取对应那条轨迹的action所以结果第k个维度没了
		actions = torch.gather(elite_actions, 2, selected_idxs).squeeze(2) 
		self._prev_mean = mean
		# mean.shape = (B,H,A) 这里是要在Horizon维度上取第一个时间步的值对应MPC；
		mean, std = actions[:, :1, :], _std[:, :1, :] # (B,1,A)
		self._std = _std
		# a.shape = (B,1,A)
		a = mean
		if not eval_mode:
			a += std * torch.randn(self.cfg.action_dim, device=std.device)
		a = a.squeeze(1) # (num_envs, action_dim)
		return a # (num_envs, action_dim)

	def update_pi(self, zs):
		"""Update policy using a sequence of latent states."""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)

		# Loss is a weighted sum of Q-values
		# zs = [z_0, z_1]; zs.shape = (2,batch_size, latent_dim)
		pi_loss = 0
		for t,z in enumerate(zs):
			# z.shape = (batch_size, latent_dim); a.shape = (batch_size, action_dim)
			a = self.model.pi(z, self.cfg.min_std)
			# self.model.Q(z, a) 会 return Q1和Q2； Q1.shape = (B,1) ; Q2.shape = (B,1)
			# Q.shape = (B,1)
			Q = torch.min(*self.model.Q(z, a))
			# pi_loss = (1,) scalar
			pi_loss += -Q.mean() * (self.cfg.rho ** t)

		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.pi_optim.step()
		self.model.track_q_grad(True)
		return pi_loss.item()

	@torch.no_grad()
	def _td_target(self, next_obs, reward):  # next_obs.shape = (Batch_size,obs_dim)； reward.shape = (Batch_size,1)
		"""Compute the TD-target from a reward and the observation at the following time step."""
		# next_z.shape = (batch_size, latent_dim) ; next_obs.shape = (H+1, Batch_size,obs_dim)
		next_z = self.model.h(next_obs)
		# r_i + gamma*min(Q(z,pi_theta)), Q = Q_1, Q_2; Q1.shape = (B,1) ; Q2.shape = (B,1)
		# td_target.shape = (Batch_size,1)
		td_target = reward + self.cfg.discount * \
			torch.min(*self.model_target.Q(next_z, self.model.pi(next_z, self.cfg.min_std)))
		return td_target

	def update(self, replay_buffer, step):
		"""Main update function. Corresponds to one iteration of the TOLD model learning."""
		# TOLD step 9 
		# obs.shape = (batch_size, obs_dim);  next_obs.shape = (H+1, Batch_size,obs_dim); 
  		# action.shape = (H, Batch_size, action_dim) ;reward.shape = (H, Batch_size,1);weight.shape(batch_size,)
		obs, next_obses, action, reward, idxs, weights = replay_buffer.sample()
		self.optim.zero_grad(set_to_none=True)
		self.std = h.linear_schedule(self.cfg.std_schedule, step)
		# PyTorch nn.Module 的内置方法，用于将模型设置为“训练模式”（self.training = True）
		self.model.train()

		# Representation
		# obs.shape = (batch_size, obs_dim) ；z.shape =  (batch_size, latent_dim)
		z = self.model.h(obs) 
		# detach() 返回一个新的张量，与原 z 共享相同的数据（内存），但断开了梯度计算链（requires_grad = False）
		# zs 是一个列表（list），初始只包含一个元素：detach 后的 z; zs[0] 的维度是 (batch_size, latent_dim)
		zs = [z.detach()]

		consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
		# TOLD step 12
		for t in range(self.cfg.horizon):

			# Predictions
			# action.shape = (H, Batch_size, action_dim) 取第t个[batch_size,action_dim]的矩阵
			# Q1.shape = (B,1) ; Q2.shape = (B,1)
			Q1, Q2 = self.model.Q(z, action[t]) 
			# TOLD step 15, step 13 
   			#  z.shape =  (batch_size, latent_dim) ; action.shap = [batch_size,action_dim]
			# z.shape = (batch_size, cfg.latent_dim)； reward_pred.shape = (batch_size, 1)
			z, reward_pred = self.model.next(z, action[t])
			with torch.no_grad():
				# next_obs.shape = (H+1, Batch_size,obs_dim)
				next_obs = next_obses[t]
				# z_t+1 = h_theta(s_t+1)
				# 输入：next_obs.shape = (batch_size, obs_dim) 输出： next_z.shape = (batch_size, latent_dim)
				next_z = self.model_target.h(next_obs)
				# TOLD step 14
				# next_obs.shape = (H+1, Batch_size,obs_dim)； reward.shape = (H, Batch_size,1)
				# td_target.shape = (Batch_size, 1)
				td_target = self._td_target(next_obs, reward[t])
			# zs = [z_0, z_1]; zs.shape = (2,batch_size, latent_dim); z.shape =  (batch_size, latent_dim)
			zs.append(z.detach())

			# Losses
			rho = (self.cfg.rho ** t)
			# TOLD step 15
			# z.shape = (batch_size, cfg.latent_dim); next_z.shape = (batch_size, latent_dim)
			# h.mse(z, next_z) → (B, latent_dim) （逐元素平方差); torch.mean(..., dim=1, keepdim=True) → (B, 1)
			# consistency_loss.shape = (B,1)
			consistency_loss += rho * torch.mean(h.mse(z, next_z), dim=1, keepdim=True)
			# TOLD step 13
			# reward_pred.shape = (batch_size, 1) ;  reward.shape = (Batch_size,1)
			# reward_loss.shape = (batch_size, 1)
			reward_loss += rho * h.mse(reward_pred, reward[t])
			# TOLD step 14
			# 用 MSE (均方误差) 衡量 Q1/Q2 与 TD target 的差距, critic loss
			# td_target.shape = (Batch_size, 1) ; Q1.shape = (B,1) ; Q2.shape = (B,1)
			# value_loss.shape = (Batch_size, 1)
			value_loss += rho * (h.mse(Q1, td_target) + h.mse(Q2, td_target))
			# 用 L1 (绝对值误差) 计算 TD 误差
			# 这个值不会反传梯度，而是用于 更新优先经验回放的 priority
			# td_target.shape = (Batch_size, 1) ; Q1.shape = (B,1) ; Q2.shape = (B,1)
			# priority_loss.shape = (Batch_size, 1)
			priority_loss += rho * (h.l1(Q1, td_target) + h.l1(Q2, td_target))

		# Optimize model
		# total_loss.shape = (B,1)；onsistency_loss.shape = (B,1)；reward_loss.shape = (B, 1)；value_loss.shape = (Batch_size, 1)
		total_loss = self.cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
					 self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
					 self.cfg.value_coef * value_loss.clamp(max=1e4)
		# weights → 来自优先经验回放 (PER) 的采样权重，用于修正采样 bias
		# weight.shape(batch_size,); total_loss.squeeze(1) = (batch_size,); weight_loss.shape = (1,)
		weighted_loss = (total_loss.squeeze(1) * weights).mean()
		# 因为这里 loss 是累加了整个 horizon 的误差，所以要除以 horizon，防止梯度爆炸
		weighted_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon))
		weighted_loss.backward()
		# clip_grad_norm_() → 限制梯度范数，防止梯度过大导致不稳定（梯度爆炸）
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		# 参数更新
		self.optim.step()
		# update_priorities() → 用刚才的 priority_loss 更新 replay buffer 中每条样本的优先级
		# priority_loss.shape = (Batch_size, 1)
		replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())

  		# TOLD step 18 
		# Update policy + target network ; zs = [z_0, z_1]
		pi_loss = self.update_pi(zs)
		# TOLD step 19
		if step % self.cfg.update_freq == 0:
			h.ema(self.model, self.model_target, self.cfg.tau)

		self.model.eval()
		return {
				'mean_action_noise_std': self._std.mean().item(),
				'mean_value_loss': value_loss.mean().item(),
				'mean_pi_loss': pi_loss,
				'mean_reward_loss': reward_loss.mean().item(),
				'total_loss': total_loss.mean().item(),
				'weighted_loss': weighted_loss.mean().item(),
				'grad_norm': grad_norm,
			}

			

