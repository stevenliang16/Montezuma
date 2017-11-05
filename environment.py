import sys
import os
import logging
import cv2
from ale_python_interface import ALEInterface

logger = logging.getLogger(__name__)


class ALEEnvironment():

  def __init__(self, rom_file, args):

    self.ale = ALEInterface()
    if args.display_screen:
      if sys.platform == 'darwin':
        import pygame
        pygame.init()
        self.ale.setBool('sound', False) # Sound doesn't work on OSX
      elif sys.platform.startswith('linux'):
        self.ale.setBool('sound', True)
      self.ale.setBool('display_screen', True)

    self.ale.setInt('frame_skip', args.frame_skip)
    self.ale.setFloat('repeat_action_probability', args.repeat_action_probability)
    self.ale.setBool('color_averaging', args.color_averaging)

    if args.random_seed:
      self.ale.setInt('random_seed', args.random_seed)

    if args.record_screen_path:
      if not os.path.exists(args.record_screen_path):
        logger.info("Creating folder %s" % args.record_screen_path)
        os.makedirs(args.record_screen_path)
      logger.info("Recording screens to %s", args.record_screen_path)
      self.ale.setString('record_screen_dir', args.record_screen_path)

    if args.record_sound_filename:
      logger.info("Recording sound to %s", args.record_sound_filename)
      self.ale.setBool('sound', True)
      self.ale.setString('record_sound_filename', args.record_sound_filename)

    self.ale.loadROM(rom_file)

    if args.minimal_action_set:
      self.actions = self.ale.getMinimalActionSet()
      logger.info("Using minimal action set with size %d" % len(self.actions))
    else:
      self.actions = self.ale.getLegalActionSet()
      logger.info("Using full action set with size %d" % len(self.actions))
    logger.debug("Actions: " + str(self.actions))

    self.screen_width = args.screen_width
    self.screen_height = args.screen_height

    self.mode = "train"
    self.life_lost = False
    self.initSrcreen = self.getScreen()
    self.goalSet = []
    self.goalSet.append([[8, 21], [16, 36]]) # top left door
    self.goalSet.append([[69, 21], [77, 36]]) # top right door
    self.goalSet.append([[37, 40], [47, 53]]) # middle ladder
    self.goalSet.append([[8, 57], [19, 72]]) # lower left ladder
    self.goalSet.append([[66,57], [76, 72]]) # lower right ladder
    self.goalSet.append([[6, 39], [12, 47]]) # key

  def numActions(self):
    return len(self.actions)

  def restart(self):
    # In test mode, the game is simply initialized. In train mode, if the game
    # is in terminal state due to a life loss but not yet game over, then only
    # life loss flag is reset so that the next game starts from the current
    # state. Otherwise, the game is simply initialized.
    if (
                  self.mode == 'test' or
                not self.life_lost or  # `reset` called in a middle of episode
              self.ale.game_over()  # all lives are lost
    ):
      self.ale.reset_game()
    self.life_lost = False

  def act(self, action):
    lives = self.ale.lives()
    reward = self.ale.act(self.actions[action])
    self.life_lost = (not lives == self.ale.lives())
    return reward

  def getScreen(self):
    screen = self.ale.getScreenGrayscale()
    resized = cv2.resize(screen, (self.screen_width, self.screen_height))
    return resized

  def isTerminal(self):
    if self.mode == 'train':
      return self.ale.game_over() or self.life_lost
    return self.ale.game_over()

  def reset(self):
    self.ale.reset_game()

  def goalReached(self, goal):
    goalPosition = self.goalSet[goal]
    goalScreen = self.initSrcreen
    stateScreen = self.getScreen()
    count = 0
    for y in range (goalPosition[0][0], goalPosition[1][0]):
      for x in range (goalPosition[0][1], goalPosition[1][1]):
        if goalScreen[x][y] != stateScreen[x][y]:
          count = count + 1
    if float(count) / ((goalPosition[1][0] - goalPosition[0][0]) * (goalPosition[1][1] - goalPosition[0][1])) > 0.15:
      return True
    return False


