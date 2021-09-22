# implements controller structure for player
from evoman.controller import Controller


class player_controller(Controller):
	def __init__(self, _n_hidden):
		# Number of hidden neurons
		self.n_hidden = [_n_hidden]

	def control(self, inputs, controller=None):
		pass