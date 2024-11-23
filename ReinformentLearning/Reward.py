import numpy as np
import matplotlib.pyplot as plt

class RLReward:
    def __init__(self,activeFunc:int = 0, maxSpeed = 10, laneWidth = 5):
        #List of the reward functions
        self.rewardFuncs = [self.reward0, self.reward1]
        
        self.setActiveReward(activeFunc)
        self._maxSpeed = maxSpeed
        self._laneWidth = laneWidth
        
        
        
    #Check range of the requested active function.
    def setActiveReward(self, activeFunc:int):
        if activeFunc < 0:
            self._activeFunc = 0
        elif  activeFunc >= len(self.rewardFuncs):
            self._activeFunc = len(self.rewardFuncs)-1
        else:
            self._activeFunc = activeFunc
    
    def calcReward(self, vehicleState:dict):
        
        reward = self.rewardFuncs[self._activeFunc](vehicleState)
        
        return reward
    
    def reward0(self,vehicleState: dict):
        #Unpack vehicle state
        speed = vehicleState['speed']
        posError = np.linalg.norm(vehicleState['path_error'])
        collision = 0 #Placeholder for now
        
        #Pos Error Params
        #The two piecewise elements connect at x=lanewidth
        linExpoPt = 0.5 #Y value where the two piecewise elements connect.
        yInt = 1 #Y intercept for the linear portion
        exp = 2 #Expoential value for exponential portion
        
        #Move Incentive
        if speed <= 0:
            move = -10
        else:
            move = speed/self._maxSpeed #Normalize on max speed to keep scaling in check
        
        #Obstacle Avoidance Incentive
        if collision == 1:
            obstacle = -10
        else:
            obstacle = 0
        
        #Path Following
        if abs(posError) < self._laneWidth:
            m = (linExpoPt-yInt)/self._laneWidth
            path = m*abs(posError)+yInt
        else:
            path = -1*((abs(posError)**exp) + (linExpoPt-(self._laneWidth**exp)))+linExpoPt*2
            #path = -1*(exp**(abs(posError))+linExpoPt-(exp**self._laneWidth))+linExpoPt*2
        #TODO - Consider saturating path part of the reward here
        
        return move + obstacle + path
    
    def reward1(self,vehicleState):
        return -50
    
    #reward2, reward3, etc.
    
    

#Testing/Experimenting section
if __name__ == "__main__":
    rewardFunc = RLReward(0)
    #print(rewardFunc.calcReward(5))
    rewardFunc._activeFunc = 1
    #print(rewardFunc.calcReward(5))
    
    laneWidth = 1
    #Pos Error Params
    linExpoPt = 0.5
    yInt = 1
    exp = 2   
    
    x = np.linspace(-10,10,101)
    y = []
    
    for posError in x:
        path = 0
        if abs(posError) < laneWidth:
            m = (linExpoPt-yInt)/laneWidth
            path = m*abs(posError)+yInt
        else:
            path = -1*((abs(posError)**exp) + (linExpoPt-(laneWidth**exp)))+linExpoPt*2
            #path = -1*(exp**(abs(posError))+linExpoPt-(exp**laneWidth))+linExpoPt*2
        y.append(path)
    
    plt.plot(x,y)
    plt.show()
    
