from gym.envs.box2d.lunar_lander_project.Running_Methods import run_training

"""
This method runs the training process for the Reinforcement Learning algorithms. It also stores the policy data.
"""

if __name__ == '__main__':
	# 0: Monte Carlo
	# 1: SARSA
	# 2: Q-Learning
	# 3: Expected SARSA
	# It won't run with other value.
	
	algorithm_number = 0

	episodes_number = 50000     # Episodes to train RL algorithm.
	batch = 5000                # Batch size.
	path = 'policy/mc/'         # Path to store policies.
	
	# For each epsilon value.
	for e in range(6, 10):
		# For each gamma value.
		for g in range(1, 10):
			
			epsilon = round(e*.1, 1)
			gamma = round(g*.1, 1)
	
			run_training(
					algorithm_number,
					episodes_number,
					batch,
					path,
					None,
					gamma=gamma,
					epsilon=epsilon,
					step_size=.9,
					saving_batch=False)
			
	# Example of how to run the training with initial policy Q.
	# The algorithm will load the initial policy and continue with training.

	# run_training(algorithm_number,
	# 		episodes_number,
	# 		batch,
	# 		path,
	# 		"data/mc/1/MC_Data_gamma-0.9_epsilon-0.1_50000.json",
	# 		gamma=.9,
	# 		eplsion=.1,
	# 		step_size=.9,
	# 		saving_batch=True)