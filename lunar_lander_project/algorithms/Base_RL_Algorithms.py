from abc import abstractmethod

import numpy as np
from gym.envs.box2d.lunar_lander_project.util.Utility import save_dict_as_JSON, ensure_directory_exists


class RLBaseAlgorithmClass:
	"""
	This class defines the main method used by the RL algorithms to store the results,
	load and initialized policies.
	"""
	# The policy file.
	policy_json_file = None
	
	def __init__(self, result_path=None, result_file_name=None, saving_batch=False):
		"""
		This is the constructor for the RL algorithms.
		:param result_path: The path where results are stored.
		:param result_file_name: The file name of the results.
		:param saving_batch: Whether batch data must be stored.
		"""
		self.result_path = result_path
		self.result_file_name = result_file_name
		self.saving_batch = saving_batch

	def set_policy_json_file(self, policy_json_file):
		"""
		This method sets the JSON file for the initial policy.
		:param policy_json_file: The JSON file name.
		:return: A Python dict with the policy data.
		"""
		self.policy_json_file = policy_json_file

	def save_policy_data(self, data_dict, batch_number=None, overwrite_policy_file=False):
		"""
		This method saves the policy data as a JSON file.
		:param data_dict: The data to be saved.
		:param batch_number: The batch number to be added as name.
		:param overwrite_policy_file: Whether the original policy file will be overwritten.
		"""
		if overwrite_policy_file:
			save_dict_as_JSON(self.policy_json_file, data_dict)
		else:
			file_name = self.get_file_name()
			file_name = self.add_path(file_name)
			file_name = self.add_batch_number(file_name, batch_number)
		
			save_dict_as_JSON(file_name, data_dict)

	def get_file_name(self):
		"""
		This method creates the proper JSON file name to be saved.
		:return: The JSON file name.
		"""
		if self.policy_json_file is not None:
			if '/' in self.policy_json_file:
				file_name = self.policy_json_file[self.policy_json_file.rindex('/') + 1:]
			else:
				file_name = self.policy_json_file
			
			file_name = file_name[:file_name.rindex('.')]
		else:
			file_name = self.result_file_name
			
		return file_name
	
	def add_path(self, file_name):
		"""
		This method adds the path to the JSON file name.
		:param file_name: The file name.
		:return: The path + file name.
		"""
		if self.result_path is not None:
			ensure_directory_exists(self.result_path)

			return self.result_path + file_name
		else:
			return file_name
	
	# def add_param(self, file_name):
	# 	return file_name + '_epsilon-' + str(self.epsilon) + '_gamma-'+str(self.gamma)
	
	def ensure_state_exists(self, Q, state, actions_number):
		"""
		This method ensures that the given state is in Q.
		:param Q: The Q data.
		:param state: The state to be tested.
		:param actions_number: Number of actions the state has.
		"""
		# If the policy was loaded, then there could be new states.
		if self.policy_json_file is not None:
			# This method helps to make sure the states is in Q.
			self.actions_at_state(
					state,
					Q,
					actions_number)
	
	@abstractmethod
	def initialize_Q(self, actions_number):
		"""
		This method loads and initializes the Q, when the initial policy is defined.
		:param actions_number: Number of actions.
		:return: A dict with Q data.
		"""
		pass
	
	@staticmethod
	def add_batch_number(file_name, batch_number):
		"""
		This method adds the batch size to the file name.
		:param file_name: The file name.
		:param batch_number: The batch size.
		"""
		return file_name + '_' + str(batch_number)
	
	@staticmethod
	def actions_at_state(state, data, number_states):
		"""
		This method look for the actions list for a given sate.
		If the state it isn't in the data, then it creates the vector
		using the given number_states.
		:param state: The state value.
		:param data: The data where the method searches.
		:param number_states: Number of states to create the list.
		:return: A list of actions.
		"""
		if state in data:  # TODO: Make sure works for string and int keys.
			actions = data[state]
		else:
			data[state] = [0] * number_states
			actions = data[state]
		
		return actions
	
	@staticmethod
	def e_greedy_action_probabilities(Q, state, number_actions=2, epsilon=.1):
		"""
		This method returns the pi(a|S) for all actions.
		It assigns the probabilities as follows:

		* pi(a|s) = 1 - e + (e/|A(S)|) if a = A*.
		* pi(a|s) = e/|A(S)| if a != A*.

		:param Q: The state-action values.
		:param state: The state S to be evaluated.
		:param number_actions: The number of actions.
		:param epsilon: Epsilon value.
		:return: A list of probabilities, for each action.
		"""
		A = np.ones(number_actions, dtype=float) * epsilon / number_actions
		best_action = np.argmax(Q[state])
		A[best_action] += (1.0 - epsilon)
		
		return A
	
	@staticmethod
	def get_e_greedy_action(epsilon, actions):
		"""
		This method returns an action using e-greedy.
		:param epsilon: Probability to make the desition.
		:param actions: The set of actions.
		:return: An action from the given set.
		"""
		# 1 = Non-Greedy (Explore).
		if np.random.choice(
				np.arange(0, 2),
				p=[1 - epsilon, epsilon]):
			
			a = np.random.choice(
					np.arange(len(actions)),
					p=np.ones(len(actions)) * (1 / len(actions)))
		# 0 = Greedy (Exploit).
		else:
			a = np.argmax(actions)
		
		return a
