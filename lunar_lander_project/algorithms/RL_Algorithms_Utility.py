
from gym.envs.box2d.lunar_lander_project.algorithms.Monte_Carlo_Algorithms import MonteCarlo
from gym.envs.box2d.lunar_lander_project.algorithms.Temporal_Difference_RLAlgorithms import TemporalDifference


def run_monte_carlo(
		env,
		env_states,
		result_file_name,
		saving_batch,
		batch_size,
		episodes,
		gamma=.9,
		epsilon=.1,
		path=None,
		render=False,
		policy_json_file=None,
		overwrite_policy_file=False):
	"""
	This method performs the On-Policy First-Visit Monte Carlo algorithm.
	:param env: The environment in which the algorithm will be executed.
	:param env_states: This class defines how states will be discretized for the given state.
	:param result_file_name: The JSON file name where Q is stored.
	:param saving_batch: Whether batches are saved: True for saving batches.
	:param episodes: Number of episodes that will be executed in the algorithm.
	:param gamma: The gamma value.
	:param epsilon: The epsilon value.
	:param batch_size: The processed batch size to be deploy in console.
	:param path: The path where results are stored.
	:param render: Whether the environment will be render or not.
	:param policy_json_file: The JSON file with contains the initial policy.
	:param overwrite_policy_file: Whether the given policy will be overwritten or not.
	"""
	rla = MonteCarlo(
			result_path=path,
			result_file_name=result_file_name,
			saving_batch=saving_batch)
	
	if policy_json_file is not None:
		rla.policy_json_file = policy_json_file
	
	Q, sum_returns, states_count = rla.on_policy_fv_mc_control(
			env,
			env_states,
			episodes,
			gamma=gamma,
			epsilon=epsilon,
			batch_size=batch_size,
			render=render,
			overwrite_policy_file=overwrite_policy_file)
	
	if not saving_batch:
		rla.save_policy_data(
				MonteCarlo.wrap_data(Q, sum_returns, states_count),
				episodes,
				overwrite_policy_file=overwrite_policy_file)


def run_sarsa(
		env,
		env_states,
		result_file_name,
		saving_batch,
		batch_size,
		episodes,
		gamma=.9,
		epsilon=.1,
		step_size=.9,
		path=None,
		render=False,
		policy_json_file=None,
		overwrite_policy_file=False):
	"""
	This method performs the SARSA algorithm.
	:param env: The environment in which the algorithm will be executed.
	:param env_states: This class defines how states will be discretized for the given state.
	:param result_file_name: The JSON file name where Q is stored.
	:param saving_batch: Whether batches are saved: True for saving batches.
	:param episodes: Number of episodes that will be executed in the algorithm.
	:param gamma: The gamma value.
	:param epsilon: The epsilon value.
	:param step_size: The alpha or gamma value.
	:param batch_size: The processed batch size to be deploy in console.
	:param path: The path where results are stored.
	:param render: Whether the environment will be render or not.
	:param policy_json_file: The JSON file with contains the initial policy.
	:param overwrite_policy_file: Whether the given policy will be overwritten or not.
	"""
	rla = TemporalDifference(
			result_path=path,
			result_file_name=result_file_name,
			saving_batch=saving_batch)
	
	if policy_json_file is not None:
		rla.policy_json_file = policy_json_file
	
	Q = rla.sarsa_control(
			env,
			env_states,
			episodes,
			gamma=gamma,
			epsilon=epsilon,
			step_size=step_size,
			batch_size=batch_size,
			render=render,
			overwrite_policy_file=overwrite_policy_file)
	
	if not saving_batch:
		rla.save_policy_data(
				Q,
				episodes,
				overwrite_policy_file=overwrite_policy_file)


def run_q_learning(
		env,
		env_states,
		result_file_name,
		saving_batch,
		batch_size,
		episodes,
		gamma=.9,
		epsilon=.1,
		step_size=.9,
		path=None,
		render=False,
		policy_json_file=None,
		overwrite_policy_file=False):
	"""
	This method performs the Q-Learning algorithm.
	:param env: The environment in which the algorithm will be executed.
	:param env_states: This class defines how states will be discretized for the given state.
	:param result_file_name: The JSON file name where Q is stored.
	:param saving_batch: Whether batches are saved: True for saving batches.
	:param episodes: Number of episodes that will be executed in the algorithm.
	:param gamma: The gamma value.
	:param epsilon: The epsilon value.
	:param step_size: The alpha or gamma value.
	:param batch_size: The processed batch size to be deploy in console.
	:param path: The path where results are stored.
	:param render: Whether the environment will be render or not.
	:param policy_json_file: The JSON file with contains the initial policy.
	:param overwrite_policy_file: Whether the given policy will be overwritten or not.
	"""
	rla = TemporalDifference(
			result_path=path,
			result_file_name=result_file_name,
			saving_batch=saving_batch)
	
	if policy_json_file is not None:
		rla.policy_json_file = policy_json_file
	
	Q = rla.q_learning_control(
			env,
			env_states,
			episodes,
			gamma=gamma,
			epsilon=epsilon,
			step_size=step_size,
			batch_size=batch_size,
			render=render,
			overwrite_policy_file=overwrite_policy_file)
	
	if not saving_batch:
		rla.save_policy_data(
				Q,
				episodes,
				overwrite_policy_file=overwrite_policy_file)


def run_expected_sarsa(
		env,
		env_states,
		result_file_name,
		saving_batch,
		batch_size,
		episodes,
		gamma=.9,
		epsilon=.1,
		step_size=.9,
		path=None,
		render=False,
		policy_json_file=None,
		overwrite_policy_file=False):
	"""
	This method performs the Expected SARSA algorithm.
	:param env: The environment in which the algorithm will be executed.
	:param env_states: This class defines how states will be discretized for the given state.
	:param result_file_name: The JSON file name where Q is stored.
	:param saving_batch: Whether batches are saved: True for saving batches.
	:param episodes: Number of episodes that will be executed in the algorithm.
	:param gamma: The gamma value.
	:param epsilon: The epsilon value.
	:param step_size: The alpha or gamma value.
	:param batch_size: The processed batch size to be deploy in console.
	:param path: The path where results are stored.
	:param render: Whether the environment will be render or not.
	:param policy_json_file: The JSON file with contains the initial policy.
	:param overwrite_policy_file: Whether the given policy will be overwritten or not.
	"""
	rla = TemporalDifference(
			result_path=path,
			result_file_name=result_file_name,
			saving_batch=saving_batch)
	
	if policy_json_file is not None:
		rla.policy_json_file = policy_json_file
	
	Q = rla.q_learning_control(
			env,
			env_states,
			episodes,
			gamma=gamma,
			epsilon=epsilon,
			step_size=step_size,
			batch_size=batch_size,
			render=render,
			overwrite_policy_file=overwrite_policy_file,
			expected_sarsa=True)
	
	if not saving_batch:
		rla.save_policy_data(
				Q,
				episodes,
				overwrite_policy_file=overwrite_policy_file)
