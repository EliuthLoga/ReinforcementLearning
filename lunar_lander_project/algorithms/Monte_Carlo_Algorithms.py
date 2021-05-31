import numpy as np
from collections import defaultdict
from gym.envs.box2d.lunar_lander_project.algorithms.Base_RL_Algorithms import RLBaseAlgorithmClass
from gym.envs.box2d.lunar_lander_project.util.Utility import load_dict_from_JSON


class MonteCarlo(RLBaseAlgorithmClass):
	"""
	This class defines Monte Carlo algorithms.
	"""
	@staticmethod
	def load_initial_policy(json_file):
		"""
		This method loads the given JSON file to retrieve the Q, sum_returns and  states_count.
		:param json_file: The JSON file name.
		:return: A tuple (Q, sum_returns, states_count).
		"""
		init_p = load_dict_from_JSON(json_file)
		
		return init_p['Q'], init_p['sum_returns'], init_p['states_count']
	
	def initialize_Q(self, actions_number):
		"""
		This method loads and initializes the Q, when the initial policy is defined.
		Otherwise, it creates a new Q.
		:param actions_number: Number of actions in each state.
		:return: A tuple (Q, sum_returns, states_count).
		"""
		if self.policy_json_file is not None:
			Q, sum_returns, states_count = MonteCarlo.load_initial_policy(
					self.policy_json_file)
			print('{} loaded states.'.format(len(Q.keys())))
			return Q, sum_returns, states_count
		else:
			print('Initializing policy.')
			return self.create_initial_policy(actions_number)
	
	@staticmethod
	def wrap_data(Q, sum_returns, states_count):
		"""
		This method wraps the Monte Carlo required data to store the policy.
		:param Q: A dict policy data.
		:param sum_returns: A dict with the sum of returns.
		:param states_count:
		:return: A dict with these tree elements. The keys are: Q and returns.
		"""
		return {'Q': Q, 'sum_returns': sum_returns, 'states_count': states_count}
	
	@staticmethod
	def create_initial_policy(actions_number):
		"""
		This method creates the initial policy for On-Policy Monte Carlo algorithm:
		* Q - A dict policy data.
		* sum_returns - A dict with the sum of returns in each state.
		* states_count - A dict with the value of times visited each states.
		
		:param actions_number: Number of actions.
		:return: A tuple (Q, sum_returns, states_count).
		"""
		sum_returns = defaultdict(lambda: {'0': 0, '1': 0, '2': 0, '3': 0})
		states_count = defaultdict(lambda: {'0': 0, '1': 0, '2': 0, '3': 0})
		Q = defaultdict(lambda: [0] * actions_number)
		
		return Q, sum_returns, states_count
	
	def ensure_state_in_returns(self, sum_returns, states_count, state):
		"""
		This method ensures the given state is in the sum_returns and states_count dict.
		:param sum_returns: The sum_returns dict.
		:param states_count: The states_count dict.
		:param state: The state to make sure is in the dicts.
		"""
		# When the policy is loaded then there could be
		# new states for returns.
		if self.policy_json_file is not None and \
			state not in sum_returns.keys():
			# String are used due to dict is stored.
			sum_returns[state] = {'0': 0, '1': 0, '2': 0, '3': 0}
			states_count[state] = {'0': 0, '1': 0, '2': 0, '3': 0}

	def generate_episode(self, env, env_states, Q, epsilon=.1, max_steps=2000, render=False):
		"""
		This method generates an episode using the given environment.
		:param env: The environment used to generate the episode.
		:param env_states:
		:param Q: The policy data used to generate the episode.
		:param epsilon: The epsilon for e-greedy action probability.
		:param max_steps: The maximum number of steps an episode must have.
		:param render: Whether the episode is render.
		:return: A tuple (episode, episode_reward).
		"""
		episode = []
		current_state = env_states.get_state_key(env_states.discrete_state(env.reset()))
		episode_reward = 0
		
		# For each step of episode (or until episode <= max_step).
		while len(episode) <= max_steps:
			
			# If the policy was loaded, then there could be new states.
			self.ensure_state_exists(Q, current_state, env.action_space.n)
			
			# Get probability distribution for state's action.
			prob_scores = self.e_greedy_action_probabilities(
					Q,
					current_state,
					env.action_space.n,
					epsilon)
			
			# Select the action with the given probability distribution
			# (e-greedy in this case).
			action = np.random.choice(np.arange(len(prob_scores)), p=prob_scores)

			# Take action and get new_state and reward.
			next_state, reward, done, info = env.step(action)

			# Update episode reward.
			episode_reward += reward
			
			# Render if required.
			if render:
				env.render()
			
			# Add (state, action reward) tuple to episode list.
			episode.append((current_state, action, reward))
			
			# Until S is terminal.
			if done:
				break
			
			# S <- S'.
			current_state = env_states.get_state_key(env_states.discrete_state(next_state))
		
		return episode, episode_reward
	
	def on_policy_fv_mc_control(
			self,
			env,
			env_states,
			total_episodes,
			gamma=.9,
			epsilon=.1,
			batch_size=10,
			max_steps=2000,
			render=False,
			overwrite_policy_file=False):
		"""
		This method performs the On-Policy First-Visit Monte Carlo control algorithm.
		:param env: The environment in which the algorithm will be executed.
		:param env_states: This class defines how states will be discretized for the given state.
		:param total_episodes: Number of episodes that will be executed in the algorithm.
		:param gamma: The gamma value.
		:param epsilon: The epsilon value.
		:param batch_size: The processed batch size to be deploy in console.
		:param max_steps: The maximum number of steps per episode.
		:param render: Whether the environment will be render or not.
		:param overwrite_policy_file: Whether the given policy will be overwritten or not.
		:return: A tuple (Q, sum_returns, states_count).
		"""
		print('==== First-Visit Monte Carlo ====')
		print('Running {} iterations.'.format(total_episodes))
		print('Gamma = {}. Epsilon={}.'.format(gamma, epsilon))
		
		# pi <- an arbitrary e-soft policy.
		# Initialize Q(s,a) in Real (arbitrarily), for all s in +, a in A(s).
		# Returns <- empty list, for all s in S, a in A(s)
		# The result is compounded by two dicts (sum_returns and states_count)
		# to improve computing performance.
		Q, sum_returns, states_count = self.initialize_Q(env.action_space.n)
		
		batch_number = 0
		
		# Repeat forever (for each episode):
		for e in range(1, total_episodes + 1):
			
			# Generate an episode following policy pi: S0,A0,R1,...,ST-1,AT-1, RT.
			episode, episode_reward = self.generate_episode(
					env,
					env_states,
					Q,
					epsilon=epsilon,
					max_steps=max_steps,
					render=render)
			
			# print(episode)
			if render:
				print('Episode {} reward: {}'.format(str(e), str(episode_reward)))
			
			# G <- 0
			G = 0
			
			# Loop for each step of episode, t=T-1,T-2,...,0:
			# Select only states and actions.
			state_actions_in_episode = list((sar[0], sar[1]) for sar in episode)
			
			# states_list = list((sar[0]) for sar in episode)
			# Index to iterate the actions in reverse order, t = T-1, T-2,..., 0
			t = len(state_actions_in_episode) - 2
			
			# For each step t:
			while t >= 0:
				# Get (St, At).
				state_t, action_t = state_actions_in_episode[t]
				
				# When the policy is loaded then there could be new states.
				# Make sure they are available for sum_returns and states_count.
				self.ensure_state_in_returns(sum_returns, states_count, state_t)

				# G <- gamma * G + Rt+1
				G = gamma * G + episode[int(t + 1)][2]
				
				# Unless the pair (St, At) appear in S0, A0, S1, A1,...,St-1, At-1:
				if (state_t, action_t) in state_actions_in_episode[:t]:
					# Append G to Returns(St,At)
					sum_returns[state_t][str(action_t)] += G
					states_count[state_t][str(action_t)] += 1
					
				else:
					# Update Q(St,At):
					# Q(St,At) <- average(Returns(St,At))
					# Q[state_t][action_t] = sum_returns[state_t][str(action_t)] / states_count[state_t][str(action_t)]
					
					Q[state_t][action_t] = Q[state_t][action_t] + gamma * (episode[int(t + 1)][2] - Q[state_t][action_t])
					
					# The following statements are assigned during the episode generation.
					# See: gym.envs.box2d.Utility.e_greedy_action_probabilities
					
					# A* <- argmax_a Q(St,a)
					# For all a in A(St):
					#       * pi(a|s) = 1 - e + (e/|A(S)|) if a = A*.
					# 	    * pi(a|s) = e/|A(S)| if a != A*.
				
				t -= 1
			
			# Print batch and save Q if required.
			if e % batch_size == 0:
				batch_number += 1
				print('* Batch with {} episodes.'.format(str(batch_size * batch_number)))
				
				if self.saving_batch:
					self.save_policy_data(
							self.wrap_data(Q, sum_returns, states_count),
							batch_size * batch_number,
							overwrite_policy_file=overwrite_policy_file)
		
		print('Training was completed!!!')
		
		return Q, sum_returns, states_count
	
	def on_policy_ev_mc_control(
			self,
			env,
			env_states,
			total_episodes,
			gamma=.9,
			epsilon=.1,
			batch_size=10,
			max_steps=2000,
			render=False,
			overwrite_policy_file=False):
		"""
		This method performs the On-Policy Every-Visit Monte Carlo control algorithm.
		:param env: The environment in which the algorithm will be executed.
		:param env_states: This class defines how states will be discretized for the given state.
		:param total_episodes: Number of episodes that will be executed in the algorithm.
		:param gamma: The gamma value.
		:param epsilon: The epsilon value.
		:param batch_size: The processed batch size to be deploy in console.
		:param max_steps: The maximum number of steps per episode.
		:param render: Whether the environment will be render or not.
		:param overwrite_policy_file: Whether the given policy will be overwrited or not.
		:return: A tuple (Q, sum_returns, states_count).
		"""
		print('==== Every-Visit Monte Carlo ====')
		print('Running {} iterations.'.format(total_episodes))
		print('Gamma = {}. Epsilon={}.'.format(gamma, epsilon))
		
		# pi <- an arbitrary e-soft policy.
		# Initialize Q(s,a) in Real (arbitrarily), for all s in +, a in A(s).
		# Returns <- empty list, for all s in S, a in A(s)
		# Returns is compounded by two dicts (sum_returns and states_count)
		# to improve computing performance.
		Q, sum_returns, states_count = self.initialize_Q(env.action_space.n)
		
		batch_number = 0
		
		# Repeat forever (for each episode):
		for e in range(1, total_episodes + 1):
			
			# Generate an episode following policy pi: S0,A0,R1,...,ST-1,AT-1, RT.
			episode, episode_reward = self.generate_episode(
					env,
					env_states,
					Q,
					epsilon=epsilon,
					max_steps=max_steps,
					render=render)
			
			if render:
				print('Episode {} reward: {}'.format(str(e), str(episode_reward)))
			
			# G <- 0
			G = 0
			
			# Loop for each step of episode, t=T-1,T-2,...,0:
			# Select only states and actions.
			state_actions_in_episode = list((sar[0], sar[1]) for sar in episode)
			# Index to iterate the actions in reverse order, t = T-1, T-2,..., 0
			t = len(state_actions_in_episode) - 2
			
			# For each step t:
			while t >= 0:
				# Get (St, At).
				state_t, action_t = state_actions_in_episode[t]
				
				# When the policy is loaded then there could be new states.
				# Make sure they are available for sum_returns and states_count.
				self.ensure_state_in_returns(sum_returns, states_count, state_t)
				
				# G <- gamma * G + Rt+1
				G = gamma * G + episode[int(t + 1)][2]
				
				# Unless the pair (St, At) appear in S0, A0, S1, A1,...,St-1, At-1:
				# Append G to Returns(St,At)
				sum_returns[state_t][str(action_t)] += G
				states_count[state_t][str(action_t)] += 1
				
				# Update Q(St,At):
				# Q(St,At) <- average(Returns(St,At))
				Q[state_t][action_t] = sum_returns[state_t][str(action_t)] / states_count[state_t][str(action_t)]
				
				# The following statements are assigned during the episode generation.
				# See: gym.envs.box2d.Utility.e_greedy_action_probabilities
				
				# A* <- argmax_a Q(St,a)
				# For all a in A(St):
				#       * pi(a|s) = 1 - e + (e/|A(S)|) if a = A*.
				# 	    * pi(a|s) = e/|A(S)| if a != A*.
				
				t -= 1
			
			# Print batch and save Q if required.
			if e % batch_size == 0:
				batch_number += 1
				print('* Batch with {} episodes.'.format(str(batch_size * batch_number)))
				
				if self.saving_batch:
					self.save_policy_data(
							self.wrap_data(Q, sum_returns, states_count),
							batch_size * batch_number,
							overwrite_policy_file=overwrite_policy_file)
		
		print('Training was completed!!!')
		
		return Q, sum_returns, states_count
