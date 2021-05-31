# Reinforcement Learning

This repository contains a scholar project of Reinforcement Learning subject. The practice is based on Lunar Lander, a rocket trajectory optimization. This is a classic topic in Optimal Control which is addressed using varios RL algorithms. 

Luna lander is an environment proved by Open AI Gym. The source code of this environment can be found here: https://gym.openai.com/envs/LunarLander-v2/

![Lunar Lander](https://github.com/EliuthLoga/RL/blob/master/lunar_lander_project/img/LunarLander.png)


More information about how to install Open AI Gym environments can be found here: https://gym.openai.com

## Reinforcement Learning Algorithms
This project implements the following RL algorithms: 
* Monte Carlo
* SARSA 
* Q-Learning 
* Expected SARSA.

## Running and Training Algorithms

Once Lunar Lander environment is installed based on the instructions here: https://gym.openai.com. The next step is to clone this branch and take a look at

* Deploy_LunarLander.py which deploys Lunar Lander using the selected algorithm’s previous trained policy.
* Train_LunarLander.py to keep training the selected algorithm’s previous trained policy or start a new trainning from scratch using the selected algorithm.
