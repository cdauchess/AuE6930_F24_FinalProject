from CoppeliaBridge.CoppeliaBridge import CoppeliaBridge
import matplotlib.pyplot as plt
from math import sin, pi, degrees, atan2, radians

bridge = CoppeliaBridge()

bridge.initEgo()
bridge.initScene()

bridge.startSimulation()

# bridge.setVehicleSpeed(1)

curTime = 0
runTime = 5

# bridge.switchLane(1)

switch = False

while bridge._isRunning and (curTime < runTime):    
    bridge.stepTime()    
    curTime = bridge.getTime()
    og = bridge.getOccupancyGrid()
    
    # # bridge.setMotion(1)
    # # bridge.getEgoPoseAbsolute()
    
    # # if curTime > 0.5*runTime and not switch:
    # #     bridge.switchLane(2)
    # #     print("Switched")
    # #     switch = True
    # # bridge.getVehicleSpeed()
    # # bridge.setSteering(0.2 * sin(4*pi*curTime/runTime))
    
    p,o = bridge.getPathError()
    v = bridge.getVehicleSpeed()
    
    # print(p)
    # # print(-atan2(*p, v))

    # p == 0
    # abs(p) < w

    bridge.setSteering((-o - atan2(p, v))* 0.5)

bridge.setVehicleSpeed(0)
bridge.stopSimulation()