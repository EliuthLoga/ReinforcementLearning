import numpy as np
from collections import defaultdict
from gym.envs.box2d.lunar_lander_project.algorithms.Base_RL_Algorithms import RLBaseAlgorithmClass
from gym.envs.box2d.lunar_lander_project.util.Utility import load_dict_from_JSON


class TemporalDifference(RLBaseAlgorithmClass):
	"""
	This class defines the temporal difference RL algorithms:
	SARSA, Q-Learning and Expected SARSA.
	"""
	
	@staticmethod
	def load_initial_policy(initial_policy_file):
		"""
		This method loads the given JSON file to retrieve the Q.
		:param initial_policy_file: The JSON file name.
		:return: The Q defined in the file.
		"""
		return load_dict_from_JSON(initial_policy_file)

	def initialize_Q(self, actions_number):
		"""
		This method loads and initializes the Q, when the initial policy is defined.
		Otherwise, it creates a new Q.
		:param actions_number: Number of actions in each state.
		:return: The initialized Q.
		"""
		if self.policy_json_file is not None:
			Q = TemporalDifference.load_initial_policy(self.policy_json_file)
			print('{} loaded states.'.format(len(Q.keys())))
		else:
			print('Initializing policy.')
			Q = defaultdict(lambda: [0] * actions_number)
			
		return Q

	def sarsa_control(
			self,
			env,
			env_states,
			total_episodes,
			gamma=.9,
			epsilon=.1,
			step_size=.9,
			batch_size=10,
			max_steps=2000,
			render=False,
			overwrite_policy_file=False):
		"""
		This method performs the SARSA Temporal Difference algorithm.
		:param env: The environment in which the algorithm will be executed.
		:param env_states: This class defines how states will be discretized for the given state.
		:param total_episodes: Number of episodes that will be executed in the algorithm.
		:param gamma: The gamma value.
		:param epsilon: The epsilon value.
		:param step_size: The alpha or gamma value.
		:param batch_size: The processed batch size to be deploy in console.
		:param max_steps: The maximum number of steps per episode.
		:param render: Whether the environment will be render or not.
		:param overwrite_policy_file: Whether the given policy will be overwritten or not.
		:return: The generated Q.
		"""
		print('==== SARSA ====')
		print('Running {} iterations.'.format(total_episodes))
		print('Gamma = {}. Epsilon={}. Step_Size = {}.'.format(gamma, epsilon, step_size))
		
		# Initialize Q q(S,A), for all s in S+, a in A(s), arbitrarily
		# except that Q(terminal, .) = 0.
		Q = self.initialize_Q(env.action_space.n)
		
		batch_number = 0
		
		# Loop for each episode:
		for episode in range(1, total_episodes + 1):
			
			# Initialize S (current_state).
			current_state = env_states.get_state_key(env_states.discrete_state(env.reset()))
			
			# If the policy was loaded, then there could be new actions.
			# Make sure current_state is in Q dict.
			self.ensure_state_exists(Q, current_state, env.action_space.n)
			
			# Choose A (current_action) from S
			# using policy derived from Q (e.g., e-greedy)
			current_action = self.get_e_greedy_action(epsilon, Q[current_state])
			
			# Track number of steps and episode reward.
			step_number = 0
			episode_reward = 0

			# Loop for each step of episode, or until episode <= max_step:
			while step_number <= max_steps:
				
				# Take action A, observe R (reward), S' (next_state).
				next_state, reward, done, info = env.step(current_action)
				
				# Update episode reward.
				episode_reward += reward
				
				# Render if required.
				if render:
					env.render()
				
				# Transform next_state to a discrete value and make sure it's in Q dict.
				next_state = env_states.get_state_key(env_states.discrete_state(next_state))
				self.ensure_state_exists(Q, next_state, env.action_space.n)
				
				# Choose A' from S' using policy derived from Q (e.g., e-greedy).
				next_action = self.get_e_greedy_action(epsilon, Q[next_state])
				
				# Bootstrapping:
				# Q(S, A) <- Q(S,A) + alpha[R + gamma*Q(S',A') - Q(S,A)]
				Q[current_state][current_action] += step_size * (reward + (
						gamma * Q[next_state][next_action]) - Q[current_state][current_action])
				
				# S <- S'
				current_state = next_state
				# A <- A'
				current_action = next_action
				
				# Increment step track in 1.
				step_number += 1
				
				# Until S is terminal.
				if done:
					print("Done with " + str(step_number) + " steps. ")
					break
			
			if render:
				print('Episode reward: {}'.format(str(episode_reward)))
			
			# Print batch and save if required.
			if episode % batch_size == 0:
				print('* Batch with {} episodes.'.format(str(batch_size * batch_number)))
				batch_number += 1
				
				if self.saving_batch:
					self.save_policy_data(
							Q,
							batch_size * batch_number,
							overwrite_policy_file=overwrite_policy_file)
		print('Training was completed!!!')
		
		return Q

	def q_learning_control(
			self,
			env,
			env_states,
			total_episodes,
			gamma=.9,
			epsilon=.1,
			step_size=.9,
			batch_size=10,
			max_steps=2000,
			render=False,
			overwrite_policy_file=False,
			expected_sarsa=False):
		"""
		This method performs the Q-Learning or Expected SARSA algorithm, deppending on the expected_sarsa flag value.
		If True the method will execute the Expected SARSA algorithm.
		If False the method will execute Q-Learning algorithm.
		:param env: The environment in which the algorithm will be executed.
		:param env_states: This class defines how states will be discretized for the given state.
		:param total_episodes: Number of episodes that will be executed in the algorithm.
		:param gamma: The gamma value.
		:param epsilon: The epsilon value.
		:param step_size: The alpha or gamma value.
		:param batch_size: The processed batch size to be deploy in console.
		:param max_steps: The maximum number of steps per episode.
		:param render: Whether the environment will be render or not.
		:param overwrite_policy_file: Whether the given policy will be overwritten or not.
		:param expected_sarsa: If True the algorithm will execute the Expected SARSA algorithm.
		Otherwise, it will execute Q-Learning algorithm.
		:return: The generated Q.
		"""
		if expected_sarsa:
			print('==== Expected SARSA ====')
		else:
			print('==== Q-Learning ====')
		
		print('Running {} iterations.'.format(total_episodes))
		print('Gamma = {}. Epsilon={}. Step_Size = {}.'.format(gamma, epsilon, step_size))
		
		# Initialize Q q(S,A), for all s in S+, a in A(s), arbitrarily
		# except that Q(terminal, .) = 0.
		Q = self.initialize_Q(env.action_space.n)
		
		batch_number = 0
		
		# Loop for each episode:
		for episode in range(1, total_episodes + 1):
			
			# Initialize S (current_state).
			current_state = env_states.get_state_key(env_states.discrete_state(env.reset()))
			
			# If the policy was loaded, then there could be new actions.
			# Make sure current_state is in Q dict.
			self.ensure_state_exists(Q, current_state, env.action_space.n)
			
			# Track number of steps and episode reward.
			step_number = 0
			episode_reward = 0
			
			# Loop for each step of episode, or until episode <= max_step:
			while step_number <= max_steps:
				
				# Choose A (current_action) from S
				# using policy derived from Q (e.g., e-greedy)
				current_action = self.get_e_greedy_action(epsilon, Q[current_state])
				
				# Take action A, observe R (reward), S' (next_state).
				next_state, reward, done, info = env.step(current_action)
				
				# Update episode reward.
				episode_reward += reward
				
				# Render if required.
				if render:
					env.render()
				
				# Transform next_state to a discrete value and make sure it's in Q dict.
				next_state = env_states.get_state_key(env_states.discrete_state(next_state))
				self.ensure_state_exists(Q, next_state, env.action_space.n)
				
				# Bootstrapping:
				# self.get_q_term() will return the proper value,
				# depending on whether is expected SARSA or QL.
				
				Q[current_state][current_action] += step_size * (reward + (gamma * self.get_q_term(
						Q, next_state, env.action_space.n, expected_sarsa)) - Q[current_state][current_action])
				
				# S <- S'
				current_state = next_state
				
				# Increment step track in 1.
				step_number += 1
				
				# Until S is terminal.
				if done:
					#print("Done with " + str(step_number) + " steps.")
					break
					
			if render:
				print('Episode reward: {}'.format(str(episode_reward)))

			# Print batch and save if required.
			if episode % batch_size == 0:
				batch_number += 1
				print('* Batch with {} episodes.'.format(str(batch_size * batch_number)))
				
				if self.saving_batch:
					self.save_policy_data(
							Q,
							batch_size * batch_number,
							overwrite_policy_file=overwrite_policy_file)
		print('Training was completed!!!')
		
		return Q

	def get_q_term(self, Q, state, actions_number, expected_sarsa):
		"""
		The difference between Q-Learning and Expected SARSA is the calculus of one term.
		So, this method calculates that term depending on the expected_sarsa flag value.
		If True the method will execute the Expected SARSA algorithm.
		If False the method will execute Q-Learning algorithm.
		:param Q: The Q dict.
		:param state: The state.
		:param actions_number: Number of action the state has.
		:param expected_sarsa: If True it returns Q for Expected SARSA. Otherwise, it returns Q for Q-Learning.
		:return: The term value.
		"""
		# Select the Q(S',a) term to be used.
		if expected_sarsa:
			
			# Calculate the expected Q(S',a).
			# Q(S, A) <- Q(S,A) + alpha[R + gamma * sum(pi(a|St+1) * Q(St+1,a)) - Q(S,A)]
			next_state_action_value = self.sum_expected_state_action(
					Q,
					state,
					actions_number)
		else:
			# Select a such as max_a Q(S',a).
			# Q(S, A) <- Q(S,A) + alpha[R + gamma * max_a Q(S',a) - Q(S,A)]
			next_state_action_value = Q[state][int(np.argmax(Q[state]))]
		
		return next_state_action_value

	@staticmethod
	def sum_expected_state_action(Q, state, actions_number):
		"""
		This method calculates the expected values sum that is required by Expected SARSA algorithm.
		:param Q: The Q dict.
		:param state: The state.
		:param actions_number: Number of actions the state has.
		:return: The sum of expectations.
		"""
		result = 0
		
		pi = RLBaseAlgorithmClass.e_greedy_action_probabilities(
				Q,
				state,
				number_actions=actions_number)
		
		for action in range(0, actions_number):
			result += pi[action] * Q[state][action]
			
		return result
