import math
import numpy as np

from gym.envs.box2d.lunar_lander_project.states.Environment_States import EnvironmentStates

"""
This Python file contains useful methods to discrete Lunar Lander states.

Here's a description of the states and possible values:

	s = [horizontal coordinate (-1, 1),
		vertical coordinate (1.38,-.38),
		horizontal speed (- 1 left, +1 right),
		vertical speed (+ up, - down),
		angle (-3.5  CW, +3.5 CCW),
		angularSpeed (-infty CW, +infty CCW),
		leg-right lands,
		leg-left lands]

The main methods are:

	* LunarLanderPolarStates.discrete_state(state)
	* LunarLanderPolarStates.get_state_key(state)

"""


class LunarLanderPolarStates(EnvironmentStates):
	
	"""
	This class defines the Lunar Lander states in Polar coordinates.
	"""
	
	def get_state_key(self, state):
		"""
		This method returns a string version of the state list.
		The string value is used as dict key value.
		:param state: The state as list.
		:return: A string representation of the state.
		"""

		return ','.join(str(x) for x in state)
	
	def discrete_state(self, state):
		"""
		This method returns the discrete value of the state.
		:param state: A list of values that describes the lunar lander state.
		The values are continues.
		:return: A list of discrete values.
		"""
		spatial = LunarLanderPolarStates.spatial_state(state[0], state[1])
		speed = LunarLanderPolarStates.speed_state(state[2], state[3])
		angles = LunarLanderPolarStates.angles_state(state[4], state[5])
		
		return (spatial[0], spatial[1],
				speed[0], speed[1],
				angles[0], angles[1],
				state[6], state[7])
	
	@staticmethod
	def angles_state(angle, speed_angle):
		"""
		This method returns the discrete values for the ship angle and speed angle.
		:param angle: A float number that represents the angle value.
		:param speed_angle: A float value that represents the speed angle value.
		:return: A tuple of discrete values (angle, speed_angle).
		"""
		return LunarLanderPolarStates.angle_state(angle), LunarLanderPolarStates.speed_angle_state(speed_angle)
	
	@staticmethod
	def speed_angle_state(val):
		"""
		This method defines the discrete states for the speed angle.
		:param val: A continues value of the speed angle.
		:return: An int value [1, 10] for counterclockwise direction,
				or [-1 - -10] for clockwise direction.
		"""
		if val >= 0:
			if val <= .05:
				state = 1
			elif val <= .1:
				state = 2
			elif val <= .15:
				state = 3
			else:
				state = 4
		else:
			if val >= -.05:
				state = -1
			elif val >= -.1:
				state = -2
			elif val >= -.15:
				state = -3
			else:
				state = -4
		
		return state
	
	@staticmethod
	def angle_state(val):
		"""
		This method defines the discrete states for the angle.
		:param val: A continues value of the angle.
		:return: An int value [1, 10] for counterclockwise direction,
				or [-1 - -10] for clockwise direction.
		"""
		if val >= 0:
			if val <= .05:
				state = 1
			elif val <= .1:
				state = 2
			elif val <= .15:
				state = 3
			elif val <= .2:
				state = 4
			elif val <= .3:
				state = 5
			elif val <= .5:
				state = 6
			elif val <= .75:
				state = 7
			elif val <= 1.5:
				state = 8
			elif val <= 2.4:
				state = 9
			else:
				state = 10
		else:
			if val >= -.05:
				state = -1
			elif val >= -.1:
				state = -2
			elif val >= -.15:
				state = -3
			elif val >= -.2:
				state = -4
			elif val >= -.3:
				state = -5
			elif val >= -.5:
				state = -6
			elif val >= -.75:
				state = -7
			elif val >= -1.5:
				state = -8
			elif val >= -2.4:
				state = -9
			else:
				state = -10
		
		return state
	
	@staticmethod
	def spatial_state(x, y):
		"""
		This method defines the discrete states for the x and y coordinates.
		It converts cartesian coordinates to polar coordinates.
		:param x: The continues value for the x cartesian coordinate.
		:param y: The continues value for the y cartesian coordinate.
		:return: A tuple of polar coordinates: (radius, angle).
				radius: int [1, 12]
				angle: int [1, 6] or [-1, -6]
		"""
		# Convert x and y values to radius and angle values.
		r, angle = LunarLanderPolarStates.to_polar(x, y)
		
		return LunarLanderPolarStates.radius_state(r), LunarLanderPolarStates.region_state(angle)
	
	@staticmethod
	def to_polar(x, y):
		"""
		This method converts the x and y cartesian coordinates to polar coordinates.
		:param x: The continues value for the x cartesian coordinate.
		:param y: The continues value for the y cartesian coordinate.
		:return: A tuple of continues values (radius, angle).
		"""
		return np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)
	
	@staticmethod
	def region_state(angle):
		"""
		This method defines the discrete states (regions) of the polar-angle coordinates.
		:param angle: The continues value of the polar-angle coordinate.
		:return: An int value [1, 6] for top regions from right-left,
		or [-1, -6] for down regions from right-left.
		"""
		if angle >= 0:
			if angle < round(math.pi/6, 3):
				state = 1
			elif angle < round((2 * math.pi)/6, 3):
				state = 2
			elif angle < round((3 * math.pi)/6, 3):
				state = 3
			elif angle < round((4 * math.pi)/6, 3):
				state = 4
			elif angle < round((5 * math.pi)/6, 3):
				state = 5
			else:
				state = 6
		else:
			if angle >= -round(math.pi/6, 3):
				state = -1
			elif angle >= -round((2 * math.pi)/6, 3):
				state = -2
			elif angle >= -round((3 * math.pi)/6, 3):
				state = -3
			elif angle >= -round((4 * math.pi)/6, 3):
				state = -4
			elif angle >= -round((5 * math.pi)/6, 3):
				state = -5
			else:
				state = -6
		
		return state
	
	@staticmethod
	def radius_state(r):
		"""
		This method defines the discrete states of the
		polar-radius coordinate value.
		:param r: The continues value of the polar-radius coordinate.
		:return: An int value [1, 6].
		"""
		if r <= .1:
			state = 1
		elif r <= .2:
			state = 2
		elif r <= .3:
			state = 3
		elif r <= .4:
			state = 4
		elif r <= .5:
			state = 5
		elif r <= .6:
			state = 6
		elif r <= .7:
			state = 7
		elif r <= .8:
			state = 8
		elif r <= .9:
			state = 9
		elif r <= 1:
			state = 10
		elif r <= 1.2:
			state = 11
		else:
			state = 12
		
		return state
	
	@staticmethod
	def speed_state(x, y):
		"""
		This method returns the discrete states of the speed in x and y axes.
		:param x: The continues value of the speed in the x-axis.
		:param y: The continues value of the speed in the y-axis.
		:return: A tuple of discrete values (speed_x, speed_y).
		"""
		return LunarLanderPolarStates.speed(x), LunarLanderPolarStates.speed(y)
	
	@staticmethod
	def speed(val):
		"""
		This method defines the discrete states of the speed in x or y-axis.
		:param val: The continues value of speed in x or y-axis.
		:return: An int value [1, 10] for up and right or [-1, -10] for down and left.
		"""
		if val >= 0:
			if val <= .1:
				state = 1
			elif val <= .2:
				state = 2
			elif val <= .3:
				state = 3
			elif val <= .4:
				state = 4
			elif val <= .5:
				state = 5
			elif val <= .6:
				state = 6
			elif val <= .7:
				state = 7
			elif val <= .8:
				state = 8
			elif val <= .9:
				state = 9
			else:
				state = 10
		else:
			if val >= -.1:
				state = -1
			elif val >= -.2:
				state = -2
			elif val >= -.3:
				state = -3
			elif val >= -.5:
				state = -5
			elif val >= -.6:
				state = -6
			elif val >= -.7:
				state = -7
			elif val >= -.8:
				state = -8
			elif val >= -.9:
				state = -9
			else:
				state = -10
		
		return state
