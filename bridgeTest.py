from CoppeliaBridge.CoppeliaBridge import CoppeliaBridge
import matplotlib.pyplot as plt
from math import sin, pi, degrees

bridge = CoppeliaBridge()

bridge.initEgo()

bridge.startSimulation()

# bridge.getPaths()

# bridge.setVehicleSpeed(0)
# bridge.setSteering(0.4)

curTime = 0
runTime = 4

while bridge._isRunning and (curTime < runTime):    
    bridge.stepTime()
    curTime = bridge.getTime()
    bridge.setMotion(1)
    # bridge.getEgoPoseAbsolute()

    # bridge.getVehicleSpeed()
    # bridge.setSteering(0.2 * sin(4*pi*curTime/runTime))

bridge.setVehicleSpeed(0)
bridge.stopSimulation()