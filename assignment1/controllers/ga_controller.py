import pygad.gann as gann
from demo_controller import player_controller

class ga_controller(player_controller):
    def __init__(self, population: int):
        super().__init__(0)
        self.evolving_networks = gann.GANN(
            num_solutions = population,
            num_neurons_input = 20,
            num_neurons_output = 5,
            output_activation = 'sigmoid')