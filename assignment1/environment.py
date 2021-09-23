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

    def run_single(self, enemyn, pcont, econt):
        """
        This function is run for a single instance of a match between a player and an enemy.
        :param enemyn:
        :param pcont:
        :param econt:
        :return: If the game is suddenly quit, it returns None. Otherwise, it returns
        """
        return super().run_single(enemyn, pcont, econt)

    # checks objective mode
    def play(self,pcont="None",econt="None"):
        """

        :param pcont:
        :param econt:
        :return:
        """
        return super().play(pcont, econt)