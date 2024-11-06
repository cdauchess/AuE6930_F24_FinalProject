#  python -m pip install coppeliasim-zmqremoteapi-client
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np

'''
TODO:
Path Error Determination:
    Need to decide to work in world frame or each path's frame. Likely should work in world frame for simplicity.
    

Obstacle Positions:
    Return distance vectors to obstacles "in view"
    Return contact boolean

Vehicle State:
    Package vehicle state in an easy to use format   
'''


class CoppeliaBridge:
    def __init__(self,numPaths):
        self._initialized = True
        self._isRunning = False
        
        self._client = RemoteAPIClient()        
        self._sim = self._client.require('sim')

        self._sim.setStepping(True)

        self._world = self._sim.getObject('/Floor')
        self._egoVehicle = self._sim.getObject('/Motorbike/CoM')
        self._speedMotor = self._sim.getObject('/motor')
        self._steerMotor = self._sim.getObject('/steeringMotor')
        
        #Path information loading from simulation environment
        self._path = []
        self._pathPositions = []
        self._pathQuaternions = []
        self._pathLengths = []
        for idx in range(numPaths):
            pathPath = '/path' + str(idx+1)
            self._path.append(self._sim.getObject(pathPath))
            tempPos, tempQuaternion, tempLen = self.getPathData(idx) #Load Path Data from Model
            self._pathPositions.append(tempPos)
            self._pathQuaternions.append(tempQuaternion)
            self._pathLengths.append(tempLen)
            
        self.activePath = 0 #Current path being followed by the vehicle

        #pos = self._sim.getObjectPosition(C)

    def startSimulation(self):
        '''
        Starts CoppeliaSim simulator
        '''
        self._sim.startSimulation()
        self._isRunning = True

    def pauseSimulation(self):
        '''
        Pause CoppeliaSim simulator
        '''
        self._sim.pauseSimulation()
        self._isRunning = False

    def stopSimulation(self):
        '''
        Stops CoppeliaSim simulator
        '''
        self._sim.stopSimulation()
        self._isRunning = False

    
    def getEgoPose(self):
        pos = self._sim.getObjectPosition(self._egoVehicle, self._path[self.activePath])
        rot = self._sim.getObjectOrientation(self._egoVehicle, self._world)

        return pos,rot
    
    def getTimeStepSize(self):
        '''
        Gets step size set in CoppeliaSim simulator
        '''
        return self._sim.getSimulationTimeStep()
    
    def stepTime(self):
        '''
        Step forward in time
        '''
        if self._isRunning:
            self._sim.step()

    def getTime(self):
        '''
        Get current time in simulation
        '''
        return self._sim.getSimulationTime()
    
    def setSpeed(self,speedTarget):
        '''
        Set the motor speed
        '''
        self._sim.setJointTargetVelocity(self._speedMotor,speedTarget)

    def setSteering(self,steerTarget):
        '''
        Set the steering
        '''
        self._sim.setJointTargetPosition(self._steerMotor,steerTarget)

    #Path Reference: https://manual.coppeliarobotics.com/en/paths.htm
    def getPathData(self, pathNum):
        '''
        Get the information about the chosen path
        '''
        #Assuming Path info is a list of position+quaterion (7 items per point) 
        self.rawPathInfo = self._sim.unpackDoubleTable(self._sim.readCustomBufferData(self._path[pathNum],'PATH'))
        pathInfo = np.reshape(self.rawPathInfo,(-1,7))
        pathPositions = pathInfo[:, :3].flatten().tolist() #XYZ Positions of each path point
        pathQuaternions = pathInfo[:, 3:].flatten().tolist() #Path point orientations
        pathLengths, self.totalLength = self._sim.getPathLengths(pathPositions, 3) #Length along path for each path point
        
        return pathPositions, pathQuaternions, pathLengths
        
    
    def getPathError(self, pathNum):
        
        pathPos = self._pathPositions[pathNum]
        pathQuaternion = self._pathQuaternions[pathNum]
        pathLen = self._pathLengths[pathNum]
        
        currPos,currOrient = self.getEgoPose()
        posAlongPath = self._sim.getClosestPosOnPath(pathPos, pathLen, currPos)
        nearestPoint = self._sim.getPathInterpolatedConfig(pathPos, pathLen, posAlongPath)
        pathError = np.subtract(currPos,nearestPoint)

        orientErr = None
        return pathError,orientErr
    



    def __del__(self):
        '''
        Destructor
        '''
        if self._initialized:
            if self._isRunning:
                self.stopSimulation()

            self._initialized = False
        pass



bridge = CoppeliaBridge(2)
bridge.startSimulation()
print(bridge.getTimeStepSize())
bridge.setSpeed(5.5)
curTime = 0
pathError = []
while bridge._isRunning and (curTime<15):
    bridge.stepTime()
    curTime = bridge.getTime()
    pathErrorT,_ = bridge.getPathError(bridge.activePath)
    pathError.append(pathErrorT)

print("Time Elapsed!")
bridge.stopSimulation()