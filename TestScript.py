from CoppeliaBridge.CoppeliaBridge import CoppeliaBridge

bridge = CoppeliaBridge(2)
bridge.startSimulation()
print(bridge.getTimeStepSize())
bridge.setSpeed(10)

curTime = 0
pathError = []
while bridge._isRunning and (curTime<30):
    bridge.stepTime()
    curTime = bridge.getTime()
    pathErrorT,_ = bridge.getPathError(bridge.activePath)
    
    #Simple Steering Controller for testing bridge functionality
    if pathErrorT[1] < 0:
        bridge.setSteering(-0.2)
    else:
        bridge.setSteering(0.2)
    pathError.append(pathErrorT)

print("Time Elapsed!")
print(bridge.getEgoPoseWorld())
bridge.stopSimulation()