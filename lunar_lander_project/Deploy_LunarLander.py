from gym.envs.box2d.lunar_lander_project.Running_Methods import run_deploy, get_policy_file

"""
This method renders the Lunar Lander using the policy data in policy/ path.
This is used to observe the result of the Reinforcement Learning algorithms.
Render executes the Lunar Lander 10 times and finally it saves the policy data after the 10 iterations.
"""

if __name__ == '__main__':
	# 0: Monte Carlo
	# 1: SARSA
	# 2: Q-Learning
	# 3: Expected SARSA
	# It won't run with other value.
	
	algorithm_number = 2
	run_deploy(algorithm_number, get_policy_file(algorithm_number))