from CoppeliaBridge.CoppeliaBridge import CoppeliaBridge
import matplotlib.pyplot as plt

bridge = CoppeliaBridge(2)
bridge.startSimulation()
bridge.setSpeed(10)
bridge.vehicleCollection()

curTime = 0
pathError = []
orientError = []
while bridge._isRunning and (curTime<5):
    bridge.stepTime()
    curTime = bridge.getTime()
    pathErrorT,orientErrorT = bridge.getPathError(bridge.activePath)
    
    #Simple Steering Controller for testing bridge functionality
    if pathErrorT[1] < 0:
        bridge.setSteering(-0.2)
    else:
        bridge.setSteering(0.2)
    pathError.append(pathErrorT)
    orientError.append(orientErrorT)
    collide = bridge.getCollision(bridge._egoVehicle)
    if collide:
        print('Vehicle has Collided')

print("Time Elapsed!")
print(bridge.getEgoPoseWorld())
bridge.stopSimulation()

#plt.plot(orientError)
#plt.show()