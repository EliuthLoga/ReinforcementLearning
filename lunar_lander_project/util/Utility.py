import os
import json
from gym.envs.box2d.lunar_lander_project.algorithms.Base_RL_Algorithms import RLBaseAlgorithmClass

"""
This Python file contain useful functions for Reinforcement Algorithms.
"""


def ensure_directory_exists(path):
	"""
	
	:param path:
	:return:
	"""
	if not os.path.exists(path):
		os.makedirs(path)


def save_dict_as_JSON(file_name, dict1):
	"""
	This method saves a Python dict as JSON file.
	:param file_name: The JSON file name.
	:param dict1: The dict to be saved.
	:return: None
	"""
	if not file_name.endswith('.json'):
		file_name += '.json'
	
	print('Saving file... ' + str(file_name))

	with open(file_name, 'w') as f:
		json.dump(dict1, f)
	
	print('Data saved successfully...')


def load_dict_from_JSON(file_name):
	"""
	This method loads the Python dict from the given JSON file.
	:param file_name: The JSON file name.
	:return: A Python dict with the file data.
	"""
	print('Loading file... ' + str(file_name))
	
	try:
		with open(file_name) as f:
			return json.load(f)
	except IOError:
		print("An error occured")
		return None


def run_environment(env, env_states, Q, episodes, epsilon=.1, render=False, max_steps=2000):
	"""
	This method renders the given environment as an application.
	:param env: The environment to be render.
	:param Q: The policy used to render de environment.
	:param epsilon: The epsilon value used to choose e-greedy actions.
	:return: The total reward.
	"""
	episode = 0
	total_reward = 0
	
	while episode <= episodes:
		episode_reward = 0
		s = env.reset()
		step = 0
		
		while step <= max_steps:
			# Get actions value at current state.
			actions = RLBaseAlgorithmClass.actions_at_state(
					env_states.get_state_key(env_states.discrete_state(s)),
					Q,
					env.action_space.n)
			
			# Take action using e-greedy.
			# a = {0: nothing, 1: left, 2: up, 3: right}
			s, r, done, info = env.step(
					RLBaseAlgorithmClass.get_e_greedy_action(epsilon, actions))
			
			# Update total reward.
			episode_reward += r
			
			if render and episode == episodes:
				env.render()
			
			if done:
				break
			
			step += 1
		
		episode += 1
		# print('Episode reward ' + str(episode_reward))
		total_reward += episode_reward
	
	return total_reward / episodes






