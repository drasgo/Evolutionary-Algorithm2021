from evoman.environment import Environment
import numpy as np


class New_Environment(Environment):
    def fitness_single(self):
        """
        Function representing the fitness formula:
        (100 - enemy_life)^alpha - (100 - player_life)^beta -
        (( Σ(100 - player life throughout time)) / time)^gamma
        """
        return (100 - self.get_enemylife()) - np.exp((100 - self.get_playerlife()), 2) - \
               np.exp(np.sum(100 - np.ndarray(self.player_life_timeseries())) / self.get_time(), 2)

    def run_single(self, enemyn: int, pcont, econt):
        """
        This function is run for a single instance of a match between a player and an enemy.
        :param enemyn: The number of the enemy, meaning that, depending on the number, a different file "enemy#.py" from
        the package evoman will be loaded.
        :param pcont: This will be used by the class Player, when there is the tile update at each game tick, for defining
        the player's controller. If "None", then manual inputs are expected.
        :param econt: This will be used by the class Enemy (which has a different implementation for each of the
        "enemy#.py" files) for defining the controller of that enemy. Note that each enemy type has a different type of
        possible actions, so be careful with that when implementing your own enemy controller. If "None", then it will
        default to the deterministic hardcoded behaviours of that enemy.
        :return: If the game is suddenly quit, it returns None. Otherwise, it returns 5 values:
        - fitness: a float value of the fitness at the end of that instance of the game.
        - player life: an integer value corresponding to the player's life at the end of the game.
        - enemy life: an integer value corresponding to the enemy's life at the end of the game.
        - time: how much time elapsed to finish the game.
        - player life timeseries: a list containing the player's life for each tick of the game
        """
        return super().run_single(enemyn, pcont, econt)

    # checks objective mode
    def play(self,pcont="None",econt="None"):
        """
        If multiplemode is "yes", then it will execute the function self.multiple, which executes the self.run_single
        function for each enemy added in the enemies list, and averages all the values returned by each instance of
        self.run_single by computing (mean - std).
        E.g. all fitnesses (1 per enemy) -> avg_fitness = mean_fitness - std_fitness
        If multiplemode is something else, then it will execute the function self.run_single with the first enemy in the
        given list of enemies.
        :param pcont: This will be used by the class Player, when there is the tile update at each game tick, for defining
        the player's controller. If "None", then manual inputs are expected.
        :param econt: This will be used by the class Enemy (which has a different implementation for each of the
        "enemy#.py" files) for defining the controller of that enemy. Note that each enemy type has a different type of
        possible actions, so be careful with that when implementing your own enemy controller. If "None", then it will
        default to the deterministic hardcoded behaviours of that enemy.
        :return: If the game is suddenly quit, it returns None. Otherwise, it returns 5 values:
        - fitness: a float value of the fitness at the end of that instance of the game.
        - player life: an integer value corresponding to the player's life at the end of the game.
        - enemy life: an integer value corresponding to the enemy's life at the end of the game.
        - time: how much time elapsed to finish the game.

        """
        return super().play(pcont, econt)