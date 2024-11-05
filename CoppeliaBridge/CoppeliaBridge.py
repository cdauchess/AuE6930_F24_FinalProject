#  python -m pip install coppeliasim-zmqremoteapi-client
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

'''
TODO:
Path Definition:
    Define paths from the python script rather than in Coppelia Sim?
    https://manual.coppeliarobotics.com/en/paths.htm

Path Error Determination:
sim.getClosestPosOnPath - Returns nearest position along the path  https://manual.coppeliarobotics.com/en/regularApi/simGetClosestPosOnPath.htm
    This will require choosing which path we should be going along. May also need to limit the points that we choose based on some metric such as distance traveled to prevent the vehicle from shortcutting the track

Obstacle Positions:
    Return distance vectors to obstacles "in view"
    Return contact boolean

Vehicle State:
    Package vehicle state in an easy to use format   

'''


class CoppeliaBridge:
    def __init__(self):
        self._initialized = True
        self._isRunning = False
        
        self._client = RemoteAPIClient()        
        self._sim = self._client.require('sim')

        self._sim.setStepping(True)

        self._world = self._sim.getObject('/Floor')
        self._egoVehicle = self._sim.getObject('/Motorbike/CoM')
        self._speedMotor = self._sim.getObject('/motor')
        self._steerMotor = self._sim.getObject('/steeringMotor')
        self._path1 = self._sim.getObject('/Motorbike/path')

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
        pos = self._sim.getObjectPosition(self._egoVehicle)
        rot = self._sim.getObjectOrientation(self._egoVehicle, self._world)

        return [pos[1], pos[2], rot[2]]
    
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

    def getPathData(self):
        #Assuming Path info is a list of position+quaterion (7 items per point) 
        #https://manual.coppeliarobotics.com/en/regularApi/simGetPathLengths.htm
        pathInfo = self._sim.unpackDoubleTable(self._sim.readCustomBufferData(self._path1,'PATH'))
        #Need to return XYZ points, and path lengths for use in finding closest pos along the path
        return pathInfo
    



    def __del__(self):
        '''
        Destructor
        '''
        if self._initialized:
            if self._isRunning:
                self.stopSimulation()

            self._initialized = False
        pass



bridge = CoppeliaBridge()
bridge.startSimulation()
print(bridge.getTimeStepSize())
bridge.setSpeed(5.5)
curTime = 0
while bridge._isRunning and (curTime<15):
    bridge.stepTime()
    curTime = bridge.getTime()

print(curTime)
print("Time Elapsed!")
bridge.stopSimulation()
print(len(bridge.getPathData()))