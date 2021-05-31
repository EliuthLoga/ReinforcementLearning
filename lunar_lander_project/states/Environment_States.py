from abc import abstractmethod


class EnvironmentStates:
	"""
	This class defines the methods the environment states must have to interact with the RL algorithms.
	"""
	
	@abstractmethod
	def get_state_key(self, state):
		"""
		This method returns a string version of the state list.
		The string value is used as dict key value.
		:param state: The state as list.
		:return: A string representation of the state.
		"""
		
		return str(int(state))
	
	@abstractmethod
	def discrete_state(self, state):
		"""
		This method the discrete form a given state.
		:param state: The state value to be discretized.
		:return: A discrete value of the state.
		"""
		return state