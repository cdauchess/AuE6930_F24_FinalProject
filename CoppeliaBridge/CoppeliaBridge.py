#  python -m pip install coppeliasim-zmqremoteapi-client
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

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
while bridge._isRunning and (curTime<45):
    bridge.stepTime()
    curTime = bridge.getTime()

print(curTime)
print("Time Elapsed!")
bridge.stopSimulation()