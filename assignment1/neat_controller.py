# implements controller structure for player
from evoman.controller import Controller
import numpy as np


class player_controller(Controller):
	def __init__(self, _n_hidden):
		# Number of hidden neurons
		self.n_hidden = [_n_hidden]

	def control(self, inputs: np.ndarray, controller=None):
		"""
		Here what action is going to do is decided. This is called by the function update in the class Player
		which is called automatically when updating the tilemaps at each tick of the game by the function run_single
		of the class Environment.
		:param inputs: A numpy list containing the values of the sensors available.
		Default array contains (in this order):
		 - horizontal distance between player and enemy
		 - vertical distance between player and enemy
		 - direction of player (1 -> right, -1 -> left)
		 - direction of enemy (1 -> right, -1 -> left)
		 - FOR EACH OF THE BULLETS (default there are 8 bullets total)
		 	- horizontal distance between player and bullet
		 	- vertical distance between player and bullet
		 	(NOTE: As a default, if there were fired less than 8 bullets, the remaining bullets will be added
		 	here with horizontal and vertical distance of 0)
		:param controller: What is going to be used here as controller. It can be whatever, and outside of the environment
		this will be used for updating its inner values (e.g. with evolutionary algorithms), and here it is used for
		extracting the next move. E.g. it can be a dictionary containing the weights and biases of a neural network,
		or a pytorch object, and so on.
		:return: a list of 5 values as one-hot vector (all 0s except one 1): [left, right, jump, shoot, release]
		"""
		pass