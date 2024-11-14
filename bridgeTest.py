from CoppeliaBridge.CoppeliaBridge import CoppeliaBridge
import matplotlib.pyplot as plt
from math import sin, pi, degrees

bridge = CoppeliaBridge()

bridge.initEgo()
bridge.initScene()

bridge.startSimulation()
# print(bridge.getPathError())

bridge.setVehicleSpeed(0.6)
# bridge.setSteering(0.45)

curTime = 0
runTime = 30

while bridge._isRunning and (curTime < runTime):    
    bridge.stepTime()
    curTime = bridge.getTime()
    # bridge.setMotion(1)
    # bridge.getEgoPoseAbsolute()

    # bridge.getVehicleSpeed()
    # bridge.setSteering(0.2 * sin(4*pi*curTime/runTime))
    p,o = bridge.getPathError()
    # print(o)
    bridge.setSteering(-0.4*o)

bridge.setVehicleSpeed(0)
bridge.stopSimulation()