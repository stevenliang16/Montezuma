import sys
import os
import logging
import cv2
import numpy as np
from ale_python_interface import ALEInterface

logger = logging.getLogger(__name__)


class ALEEnvironment():

  def __init__(self, rom_file, args):

    self.ale = ALEInterface()
    self.histLen = 4
    
    if args.display_screen:
      if sys.platform == 'darwin':
        import pygame
        pygame.init()
        self.ale.setBool('sound', False) # Sound doesn't work on OSX
      elif sys.platform.startswith('linux'):
        self.ale.setBool('sound', True)
      self.ale.setBool('display_screen', True)
    
    self.ale.setInt('frame_skip', args.frame_skip)
    #self.ale.setFloat('repeat_action_probability', args.repeat_action_probability)
    self.ale.setBool('color_averaging', args.color_averaging)

    if args.random_seed:
      self.ale.setInt('random_seed', args.random_seed)
    '''
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
    '''
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
    self.goalSet.append([[8, 21], [16, 36]]) # top left door 0
    self.goalSet.append([[69, 21], [77, 36]]) # top right door 1
    # self.goalSet.append([[37, 40], [47, 53]]) # middle ladder 2
    self.goalSet.append([[40, 49],[45, 54]]) # middle ladder bottom 2
    self.goalSet.append([[8, 57], [19, 72]]) # lower left ladder 3
    self.goalSet.append([[66,57], [76, 72]]) # lower right ladder 4
    self.goalSet.append([[6, 39], [12, 47]]) # key 5
    self.goalCenterLoc = []
    self.goalCenterLoc.append([(8.0 + 16.0)/2, (21.0 + 36.0)/2])
    self.goalCenterLoc.append([(69.0 + 77.0)/2, (21.0+36.0)/2])
    # self.goalCenterLoc.append([(37.0 + 47.0)/2, (40.0+53.0)/2])
    self.goalCenterLoc.append([(40.0 + 45.0)/2, (49.0+54.0)/2])
    self.goalCenterLoc.append([(8.0 + 19.0)/2, (57.0+72.0)/2])
    self.goalCenterLoc.append([(66.0 + 76.0)/2, (57.0+72.0)/2])
    self.goalCenterLoc.append([(6.0 + 12.0)/2, (39.0+47.0)/2])
    self.angetOriginLoc = [33, 42]
    self.reachedGoal = [0, 0, 0, 0, 0, 0]
    self.histState = [self.getState(), self.getState(), self.getState(), self.getState()]
    self.histStateInitial = list(self.histState)

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
    for i in range(20):
      self.act(0) #wait for initialization
    self.histState = [self.getState(), self.getState(), self.getState(), self.getState()]
    
  def beginNextLife(self):
    self.life_lost = False
    for i in range(20):
      self.act(0) #wait for initialization
    self.histState = [self.getState(), self.getState(), self.getState(), self.getState()]
    
  def act(self, action):
    lives = self.ale.lives()
    reward = self.ale.act(self.actions[action])
    self.life_lost = (not lives == self.ale.lives())
    return reward

  def getScreen(self):
    screen = self.ale.getScreenGrayscale()
    resized = cv2.resize(screen, (self.screen_width, self.screen_height))
    return resized

  def getScreenRGB(self):
    screen = self.ale.getScreenRGB()
    # resized = cv2.resize(screen, (self.screen_width, self.screen_height))
    resized = screen
    return resized

  def getAgentLoc(self):
    img = self.getScreenRGB()
    man = [200, 72, 72]
    mask = np.zeros(np.shape(img))
    mask[:,:,0] = man[0];
    mask[:,:,1] = man[1];
    mask[:,:,2] = man[2];

    diff = img - mask
    indxs = np.where(diff == 0)
    diff[np.where(diff < 0)] = 0
    diff[np.where(diff > 0)] = 0
    diff[indxs] = 255
    # if (np.shape(indxs[0])[0] == 0):
    #   print indxs
    #   sys.exit()
    mean_y = np.sum(indxs[0]) / np.shape(indxs[0])[0]
    mean_x = np.sum(indxs[1]) / np.shape(indxs[1])[0]
    return (mean_x, mean_y)

  def distanceReward(self, lastGoal, goal):
    if (lastGoal == -1):
      lastGoalCenter = self.angetOriginLoc
      # goalCenter = self.goalCenterLoc[goal]
      # agentX, agentY = self.getAgentLoc()
      # dis = np.sqrt((goalCenter[0] - agentX)*(goalCenter[0] - agentX) + (goalCenter[1]-agentY)*(goalCenter[1]-agentY)) 
      # # print 10 - dis
      # return 10 - dis 
    else:
      lastGoalCenter = self.goalCenterLoc[lastGoal]
    goalCenter = self.goalCenterLoc[goal]
    agentX, agentY = self.getAgentLoc()
    dis = np.sqrt((goalCenter[0] - agentX)*(goalCenter[0] - agentX) + (goalCenter[1]-agentY)*(goalCenter[1]-agentY))
    disLast = np.sqrt((lastGoalCenter[0] - agentX)*(lastGoalCenter[0] - agentX) + (lastGoalCenter[1]-agentY)*(lastGoalCenter[1]-agentY))
    disGoals = np.sqrt((goalCenter[0]-lastGoalCenter[0])*(goalCenter[0]-lastGoalCenter[0]) + (goalCenter[1]-lastGoalCenter[1])*(goalCenter[1]-lastGoalCenter[1]))
    return 1.0 * (disLast - dis) / disGoals
    
  # add color channel for input of network
  def getState(self):
    screen = self.ale.getScreenGrayscale()
    resized = cv2.resize(screen, (self.screen_width, self.screen_height))
    return np.reshape(resized, (84, 84, 1))
  
  def getStackedState(self):
    currState = self.getState()
    self.histState.append(currState)
    self.histState = self.histState[-self.histLen:]
    return np.reshape(self.histState, (84, 84, 4))

  def isTerminal(self):
    if self.mode == 'train':
      return self.ale.game_over() or self.life_lost
    return self.ale.game_over()

  def isGameOver(self):
    return self.ale.game_over()

  def isLifeLost(self):
    return self.life_lost

  def reset(self):
    self.ale.reset_game()
    self.life_lost = False

  def goalReached(self, goal):
    goalPosition = self.goalSet[goal]
    goalScreen = self.initSrcreen
    stateScreen = self.getScreen()
    count = 0
    for y in range (goalPosition[0][0], goalPosition[1][0]):
      for x in range (goalPosition[0][1], goalPosition[1][1]):
        if goalScreen[x][y] != stateScreen[x][y]:
          count = count + 1
    # 30 is total number of pixels of agent
    if float(count) / 30 > 0.5:
      self.reachedGoal[goal] = 1
      return True
    return False

  def goalNotReachedBefore(self, goal):
    if (self.reachedGoal[goal] == 1):
      return False
    return True
  