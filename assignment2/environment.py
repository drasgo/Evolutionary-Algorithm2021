import importlib

import pygame
import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)

from evoman.environment import Environment
import numpy as np


class New_Environment(Environment):
    def __init__(self,
                 experiment_name='test',
                 multiplemode="no",  # yes or no
                 enemies=[1],  # array with 1 to 8 items, values from 1 to 8
                 loadplayer="yes",  # yes or no
                 loadenemy="yes",  # yes or no
                 level=2,  # integer
                 playermode="ai",  # ai or human
                 enemymode="static",  # ai or static
                 speed="fastest",  # normal or fastest
                 inputscoded="no",  # yes or no
                 randomini="no",  # yes or no
                 sound="off",  # on or off
                 contacthurt="player",  # player or enemy
                 logs="on",  # on or off
                 savelogs="yes",  # yes or no
                 clockprec="low",
                 timeexpire=3000,  # integer
                 overturetime=100,  # integer
                 solutions=None,  # any
                 fullscreen=False,  # True or False
                 player_controller=None,  # controller object
                 enemy_controller=None,  # controller object
                 use_joystick=False,
                 graphics: bool=False):
        super().__init__(experiment_name,
                         multiplemode,
                         enemies,
                         loadplayer,
                         loadenemy,
                         level,
                         playermode,
                         enemymode,
                         speed,
                         inputscoded,
                         randomini,
                         sound,
                         contacthurt,
                         logs,
                         savelogs,
                         clockprec,
                         timeexpire,
                         overturetime,
                         solutions,
                         fullscreen,
                         player_controller,
                         enemy_controller,
                         use_joystick)
        self.player_life_timeseries = []
        self.graphics = graphics

    def fitness_single(self) -> float:
        """
        Function representing the fitness formula:
        (100 - enemy_life)^alpha - (100 - player_life)^beta -
        (( Σ(100 - player life throughout time)) / time)^gamma
        and then rescaled between 0 and 100
        """
        # old_max = 100
        # old_min = -20000
        # new_max = 100
        # new_min = 0
        # fitness_value = (100 - self.get_enemylife()) - (100 - self.get_playerlife()) ** 2 - \
        #           (np.sum(100 - np.array(self.player_life_timeseries)) / self.get_time()) ** 2
        # rescaled_value = (new_max - new_min) / (old_max - old_min) * (fitness_value - old_max) + new_max
        if self.get_playerlife() == self.get_enemylife(): #The two characters can trade kill, which leads to zero division error
            return -(self.get_time()**0.5)
        else:
            return (self.get_playerlife() / (self.get_playerlife() + self.get_enemylife())) * 100 - self.get_time()**0.5

    def set_graphics(self, graphics: bool):
        self.graphics = graphics

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
        # sets controllers
        self.pcont = pcont
        self.econt = econt
        self.player_life_timeseries = []

        self.checks_params()

        self.enemyn = enemyn  # sets the current enemy
        ends = 0
        self.time = 0
        self.freeze_p = False
        self.freeze_e = False
        self.start = False

        enemy = importlib.import_module(name='evoman.enemy' + str(self.enemyn))

        self.load_sprites()

        # game main loop

        while 1:

            # adjusts frames rate for defining game speed

            if self.clockprec == "medium":  # medium clock precision
                if self.speed == 'normal':
                    self.clock.tick_busy_loop(30)
                elif self.speed == 'fastest':
                    self.clock.tick_busy_loop()

            else:  # low clock precision

                if self.speed == 'normal':
                    self.clock.tick(30)
                elif self.speed == 'fastest':
                    self.clock.tick()

            # game timer
            self.time += 1
            self.player_life_timeseries.append(self.get_playerlife())
            if self.playermode == "human" or self.sound == "on":
                # sound effects
                if self.sound == "on" and self.time == 1:
                    sound = pygame.mixer.Sound('evoman/sounds/open.wav')
                    c = pygame.mixer.Channel(1)
                    c.set_volume(1)
                    c.play(sound, loops=10)

                if self.time > self.overturetime:  # delays game start a little bit for human mode
                    self.start = True
            else:
                self.start = True

            # checks screen closing button
            self.event = pygame.event.get()
            for event in self.event:
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

            if self.graphics:
                self.screen.fill((250, 250, 250))

            # updates objects and draws its itens on screen
            self.tilemap.update(33 / 1000., self)

            if self.graphics:
                self.tilemap.draw(self.screen)
                # # player life bar
                vbar = int(100 * (1 - (self.player.life / float(self.player.max_life))))
                pygame.draw.line(self.screen, (0, 0, 0), [40, 40], [140, 40], 2)
                pygame.draw.line(self.screen, (0, 0, 0), [40, 45], [140, 45], 5)
                pygame.draw.line(self.screen, (150, 24, 25), [40, 45], [140 - vbar, 45], 5)
                pygame.draw.line(self.screen, (0, 0, 0), [40, 49], [140, 49], 2)
                #
                # # enemy life bar
                vbar = int(100 * (1 - (self.enemy.life / float(self.enemy.max_life))))
                pygame.draw.line(self.screen, (0, 0, 0), [590, 40], [695, 40], 2)
                pygame.draw.line(self.screen, (0, 0, 0), [590, 45], [695, 45], 5)
                pygame.draw.line(self.screen, (194, 118, 55), [590, 45], [695 - vbar, 45], 5)
                pygame.draw.line(self.screen, (0, 0, 0), [590, 49], [695, 49], 2)
            #
            # if self.start == False and self.playermode == "human":
            #     myfont = pygame.font.SysFont("Comic sams", 100)
            #     pygame.font.Font.set_bold
            #     self.screen.blit(myfont.render("Player", 1, (150, 24, 25)), (50, 180))
            #     self.screen.blit(myfont.render("  VS  ", 1, (50, 24, 25)), (250, 180))
            #     self.screen.blit(myfont.render("Enemy " + str(self.enemyn), 1, (194, 118, 55)), (400, 180))

            # checks player life status
            if self.player.life == 0:
                ends -= 1

                # tells user that player has lost
                # if self.playermode == "human":
                #     myfont = pygame.font.SysFont("Comic sams", 100)
                #     pygame.font.Font.set_bold
                #     self.screen.blit(myfont.render(" Enemy wins", 1, (194, 118, 55)), (150, 180))

                self.player.kill()  # removes player sprite
                self.enemy.kill()  # removes enemy sprite

                if self.playermode == "human":
                    # delays run finalization for human mode
                    if ends == -self.overturetime:
                        return self.return_run()
                else:
                    return self.return_run()

            # checks enemy life status
            if self.enemy.life == 0:
                ends -= 1

                self.screen.fill((250, 250, 250))
                self.tilemap.draw(self.screen)

                # tells user that player has won
                # if self.playermode == "human":
                #     myfont = pygame.font.SysFont("Comic sams", 100)
                #     pygame.font.Font.set_bold
                #     self.screen.blit(myfont.render(" Player wins ", 1, (150, 24, 25)), (170, 180))

                self.enemy.kill()  # removes enemy sprite
                self.player.kill()  # removes player sprite

                if self.playermode == "human":
                    if ends == -self.overturetime:
                        return self.return_run()
                else:
                    return self.return_run()

            if self.loadplayer == "no":  # removes player sprite from game
                self.player.kill()

            if self.loadenemy == "no":  # removes enemy sprite from game
                self.enemy.kill()

            if self.graphics:
                # updates screen
                pygame.display.flip()

            # game runtime limit
            if self.playermode == 'ai':
                if self.time >= enemy.timeexpire:
                    return self.return_run()

            else:
                if self.time >= self.timeexpire:
                    return self.return_run()

    # returns results of the run
    def return_run(self):
        # gets fitness for training agents
        fitness = self.fitness_single()
        # self.print_logs(
        #     "RUN: run status: enemy: " + str(self.enemyn) + "; fitness: " + str(fitness) + "; player life: " + str(
        #        self.player.life) + "; enemy life: " + str(self.enemy.life) + "; time: " + str(self.time))
        return fitness, self.player.life, self.enemy.life, self.time

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
        :return: If the game is suddenly quit, it returns None. Otherwise, it returns 4 values:
        - fitness: a float value of the fitness at the end of that instance of the game.
        - player life: an integer value corresponding to the player's life at the end of the game.
        - enemy life: an integer value corresponding to the enemy's life at the end of the game.
        - time: how much time elapsed to finish the game.

        """
        return super().play(pcont, econt)