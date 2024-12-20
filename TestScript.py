from CoppeliaBridge.CoppeliaBridge import CoppeliaBridge
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import time
import numpy as np

def runSim(bridge: CoppeliaBridge, duration, startPoint, renderEnable):
    #bridge.initScene()
    #bridge.initEgo()
    bridge.startSimulation()
    #bridge.setVehicleSpeed(0.5)
    bridge.setInitPosition(0,startPoint)
    _, orientEr = bridge.getPathErrorPos()
    print("Initial Orientation Error: %0.2f" %orientEr)

    curTime = 0
    pathError = []
    orientError = []
    while bridge._isRunning and (curTime<duration):
        bridge.stepTime()
        curTime = bridge.getTime()
        bridge.setMotion(0.3)
        #print(curTime)
        pathErrorT,orientErrorT = bridge.getPathErrorPos()
        #print(pathErrorT)
        #print(orientErrorT)
        
        #Simple Steering Controller for testing bridge functionality
        if pathErrorT[1] < 0:
            bridge.setSteering(0.2)
        else:
            bridge.setSteering(-0.2)
            
        og = bridge.getOccupancyGrid()
        #print(bridge.checkEgoCollide(og))
        pathError.append(pathErrorT)
        orientError.append(orientErrorT)
        time.sleep(.1)

    print("Time Elapsed!")
    bridge.stopSimulation()
    return pathError,orientError

def runSimRenderTest(bridge:CoppeliaBridge, numRuns, duration):
    
    renderTimes = []
    noRenderTimes = []
    
    #startPos = np.linspace(0,1,numRuns)
    startPos = [0.4]
    #print(startPos)
    
    for run in range(numRuns): #Run the sims
        start = time.time()
        pathEr, orientEr = runSim(bridge,duration, startPos[run],True)
        end = time.time()
        renderTimes.append(end-start)
        print('Render Time: %0.2f' %(end-start))
        time.sleep(1)
        '''
        start = time.time()
        pathEr, orientEr = runSim(bridge,duration,startPos[run],False)
        
        end = time.time()
        noRenderTimes.append(end-start)
        print('No Render Time: %0.2f' %(end-start))
        '''
        #plt.plot(orientEr)
        #plt.show()
        
        time.sleep(1)
        
    return renderTimes, noRenderTimes


bridge = CoppeliaBridge()
episodeDuration = 5
render, noRender = runSimRenderTest(bridge, 1, episodeDuration)

print('%i second simulation, mean render time: %0.2f S, mean no render time: %0.2f S' %(episodeDuration, np.mean(render), np.mean(noRender)))

fig = plt.figure()
ax = plt.subplot(2,1,1)
plt.plot(render)
plt.gca().set_title('Rendered Simulation Times')
plt.ylabel('Time (S)')
plt.xlabel('Episode')
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

ax = plt.subplot(2,1,2)
plt.plot(noRender)
plt.gca().set_title('Non-Rendered Simulation Times')
plt.ylabel('Time (S)')
plt.xlabel('Episode')
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

fig.tight_layout()

plt.show()