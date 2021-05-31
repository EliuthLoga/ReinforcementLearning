
from gym.envs.box2d.lunar_lander import LunarLander
from gym.envs.box2d.lunar_lander_project.states.LunarLander_States import LunarLanderPolarStates
from gym.envs.box2d.lunar_lander_project.algorithms.RL_Algorithms_Utility import run_monte_carlo, run_sarsa, \
	run_expected_sarsa, run_q_learning
		
		
def run_rl_algorithm(
		algorithm,
		episodes,
		batch_size,
		path,
		saving_batch,
		render,
		override,
		policy_json_file,
		gamma=.9,
		epsilon=.1,
		step_size=.9):
	"""
	This method runs the training process for the given RL algorithm.
	:param algorithm: The algorithm option, 0: Monte Carlo, 1: SARSA 2: Q-Learning, 3: Expected SARSA.
	:param episodes: Number of episodes that will be executed in the algorithm.
	:param batch_size: The processed batch size to be deploy in console.
	:param path: The path where results are stored.
	:param saving_batch: Whether batches are saved: True for saving batches.
	:param render: Whether the environment will be render or not.
	:param override: Whether the given policy will be overwritten or not.
	:param policy_json_file: The JSON file with contains the initial policy.
	:param gamma: The gamma value.
	:param epsilon: The epsilon value.
	:param step_size: The alpha or gamma value.
	:return:
	"""
	param_name = '_gamma-' + str(gamma) + '_epsilon-' + str(epsilon)
	
	if algorithm == 0:
		run_monte_carlo(
				LunarLander(),
				LunarLanderPolarStates(),
				'MC_Data' + param_name,
				saving_batch,
				batch_size,
				episodes,
				gamma=gamma,
				epsilon=epsilon,
				path=path,
				render=render,
				policy_json_file=policy_json_file,
				overwrite_policy_file=override)
	
	param_name += '_stepsize-' + str(step_size)
	
	if algorithm == 1:
		run_sarsa(
				LunarLander(),
				LunarLanderPolarStates(),
				'Sarsa_Data' + param_name,
				saving_batch,
				batch_size,
				episodes,
				gamma=gamma,
				epsilon=epsilon,
				step_size=step_size,
				path=path,
				render=render,
				policy_json_file=policy_json_file,
				overwrite_policy_file=override)
	
	if algorithm == 2:
		run_q_learning(
				LunarLander(),
				LunarLanderPolarStates(),
				'QL_Data' + param_name,
				saving_batch,
				batch_size,
				episodes,
				gamma=gamma,
				epsilon=epsilon,
				step_size=step_size,
				path=path,
				render=render,
				policy_json_file=policy_json_file,
				overwrite_policy_file=override)
	
	if algorithm == 3:
		run_expected_sarsa(
				LunarLander(),
				LunarLanderPolarStates(),
				'Exp_Sarsa_Data' + param_name,
				saving_batch,
				batch_size,
				episodes,
				gamma=gamma,
				epsilon=epsilon,
				step_size=step_size,
				path=path,
				render=render,
				policy_json_file=policy_json_file,
				overwrite_policy_file=override)


def run_training(
		algorithm, number_episodes, batch_size, data_path, policy_file,
		gamma=.9, epsilon=.1, step_size=.9, saving_batch=True):
	"""
	This method runs the training lunar lander environment with the given RL algorithm.
	:param algorithm: The algorithm option, 0: Monte Carlo, 1: SARSA 2: Q-Learning, 3: Expected SARSA.
	:param number_episodes: Number of episodes that will be executed in the algorithm.
	:param batch_size: The processed batch size to be deploy in console.
	:param data_path: The path where results are stored.
	:param policy_file: The JSON file with contains the initial policy.
	:param gamma: The gamma value.
	:param epsilon: The epsilon value.
	:param step_size: The alpha or gamma value.
	:param saving_batch: Whether batches are saved: True for saving batches.
	:return:
	"""
	run_rl_algorithm(
			algorithm,
			number_episodes,
			batch_size,
			data_path,
			saving_batch,
			False,
			False,
			policy_file,
			gamma=gamma,
			epsilon=epsilon,
			step_size=step_size)


def run_deploy(algorithm, policy_file):
	"""
	This method renders the training lunar lander environment 10 times using the given RL algorithm.
	:param algorithm: The algorithm option, 0: Monte Carlo, 1: SARSA 2: Q-Learning, 3: Expected SARSA.
	:param policy_file: The JSON file with contains the initial policy.
	:return:
	"""
	run_rl_algorithm(
			algorithm,
			10,
			1,
			None,
			False,
			True,
			True,
			policy_file)


def get_policy_file(algorithm):
	"""
	This algorithm returns the policy file for each RL algorithm.
	:param algorithm: The algorithm option, 0: Monte Carlo, 1: SARSA 2: Q-Learning, 3: Expected SARSA.
	:return: The policy file.
	"""
	result = None
	
	if algorithm == 0:
		result = 'policy/mc/MC_Data_gamma-0.9_epsilon-0.1_100000.json'
	elif algorithm == 1:
		result = 'policy/sarsa/Sarsa_Data_100000.json'
	elif algorithm == 2:
		result = 'policy/ql/QL_Data_100000.json'
	elif algorithm == 3:
		result = 'policy/expected/Exp_Sarsa_Data_100000.json'
	
	return result
