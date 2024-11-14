#  python -m pip install coppeliasim-zmqremoteapi-client
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import matplotlib.pyplot as plt
from math import pi, tan, atan2, radians, remainder

'''
TODO:
Path Error Determination:
    Orientation Error. There is currently a coordinate system issue here.
    

Obstacle Positions:
    Return distance vectors to obstacles "in view"
    Return contact boolean
'''

# This class is a bridge between Coppelia Sim and Python
class CoppeliaBridge:
    def __init__(self):
        self._initialized = True
        self._isRunning = False
        self._isEgoReady = False
        
        self._client = RemoteAPIClient()        
        self._sim = self._client.require('sim')
        
        self._sim.setStepping(True)
        self._timeStep = self.getTimeStepSize()
        
        self._world = self._sim.getObject('/Floor')
        
    def initEgo(self):
        self._m = 2.57                  # mass [Kg]
        self._r = 0.033                 # tire radius [m]        
        
        self._V_MAX = 3                 # max speed [m/s]
        self._ACC_MAX = 5               # max acceleration [m/s2]
        self._D_MAX = radians(26)       # max steering angle

        self._l = 0.258                 # wheel Base [m]
        self._t = 0.157                 # track width
        self._tf = 0.112                # front king pin width

        self._a = 0.126                 # distance from CG to front axle
        self._b = self._l - self._a     # distance from CG to rear axle
        
        #Vehicle State
        self._v = 0                     # speed
        self._d = 0                     # steering Angle
        

        self._ego = self._sim.getObject('/qcar/CoM')

        self._lDriveWheel = self._sim.getObject('/qcar/base_wheelrl_joint')
        self._rDriveWheel = self._sim.getObject('/qcar/base_wheelrr_joint')        

        self._lSteerAxis = self._sim.getObject('/qcar/base_hubfl_joint')
        self._rSteerAxis = self._sim.getObject('/qcar/base_hubfr_joint')    

        self._isEgoReady = True  

        self.setSteering(self._d) 
        self.setVehicleSpeed(self._v) 

    # def initScene(self, numPaths):
    #     #Path information loading from simulation environment
    #     self._path = []
    #     self._pathPositions = []
    #     self._pathQuaternions = []
    #     self._pathLengths = []
    #     for idx in range(numPaths):
    #         pathPath = '/path' + str(idx+1)
    #         self._path.append(self._sim.getObject(pathPath))
    #         tempPos, tempQuaternion, tempLen = self.getPathData(idx) #Load Path Data from Model
    #         self._pathPositions.append(tempPos)
    #         self._pathQuaternions.append(tempQuaternion)
    #         self._pathLengths.append(tempLen)
            
    #     self.activePath = 0 #Current path being followed by the vehicle        

    def initScene(self):
        self._lanes = []
        self._currentLane = -1

        for i in range(10):
            lane = self._sim.getObject("/path" + str(i + 1), {'noError' : True})
            if lane >= 0:
                self._lanes.append(lane)
            else: 
                break            

        self._laneCount = len(self._lanes)
        self.switchLane(0)        

    # Section: Application behavior
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

    def getTime(self):
        '''
        Get current time in simulation
        '''
        return self._sim.getSimulationTime()
    
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


    # Section: Vehicle behavior     
    def getSteeringAngle(self):
        '''
        Get the current steering angle
        '''
        d = 0
        dl = self._sim.getJointPosition(self._lSteerAxis)
        
        if dl != 0:
            sign = pow(-1, dl<0)                
            R = (self._l / tan(abs(dl))) + self._tf/2
            d = sign * atan2(self._l, R)

        return d

    def setSteering(self, d):
        '''
        Set the steering target
        '''
        self._d = helper.bound(d, -self._D_MAX, self._D_MAX)
        dl = 0
        dr = 0

        if self._d != 0:
            d_abs = abs(self._d)
            sign = pow(-1, self._d < 0)

            R = (self._l / tan(d_abs))

            dl = sign * atan2(self._l, (R - self._tf/2))
            dr = sign * atan2(self._l, (R + self._tf/2))
        
        self._sim.setJointTargetPosition(self._lSteerAxis, dl)
        self._sim.setJointTargetPosition(self._rSteerAxis, dr)

        self._adjustDifferential()
    
    def getEgoPoseRelative(self, frame):
        '''
        Returns the vehicle's pose wrt a given frame
        '''
        pos = self._sim.getObjectPosition(self._ego, frame)
        rot = self._sim.getObjectOrientation(self._ego, frame)

        return pos, rot
    
    def getEgoPoseAbsolute(self):        
        return self.getEgoPoseRelative(self._world)
    
    def getVehicleSpeed(self):
        '''
        Get the current motor speed
        '''
        vl = -self._sim.getJointVelocity(self._lDriveWheel)
        vr =  self._sim.getJointVelocity(self._rDriveWheel)
        
        return((vl + vr)/2)        
    
    def setVehicleSpeed(self, v):
        '''
        Set the vehicle speed in m/s
        '''
        # self._sim.setJointTargetForce(self._speedMotor, 60) #Setting motor torque
        self._v = helper.bound(v, 0, self._V_MAX)
        self._adjustDifferential()
    
    def setMotion(self, throttle):
        throttle = helper.bound(throttle, -1.0, 1.0)                
        self.setVehicleSpeed(self._v + (throttle * self._ACC_MAX * self._timeStep))
    
    def switchLane(self, laneNumber):             
        if self._laneCount > 0:
            if (laneNumber >= 0) and (laneNumber < self._laneCount):
                self._currentLane = laneNumber
        else:
            self._currentLane = -1

        rawPathInfo = self._sim.unpackDoubleTable(self._sim.readCustomBufferData(self._lanes[self._currentLane], 'PATH'))
        # self._pathPts = (np.reshape(rawPathInfo, (-1,7)))[:, :3]
        self._pathPoints = (np.reshape(rawPathInfo, (-1,7)))[:, :3].flatten().tolist()
        self._pathLengths, self._totalLength = self._sim.getPathLengths(self._pathPoints, 3)    

    def _adjustDifferential(self):
        '''
        Mimics a differential and sets Left / Right wheel to match steering
        '''
        vl = self._v
        vr = self._v

        if self._d != 0:
            d_abs = abs(self._d)        
            sign = pow(-1, self._d < 0) # -ve indicates right turn
            
            R = (self._l / tan(d_abs))

            vl = self._v * (R - (sign * self._t/2)) / R
            vr = self._v * (R + (sign * self._t/2)) / R

        self._sim.setJointTargetVelocity(self._lDriveWheel, -vl/self._r)
        self._sim.setJointTargetVelocity(self._rDriveWheel,  vr/self._r)
        
    def getPathError(self):
        '''
        Return the error relative to the closest point along the target path
        Returns orientation error as the rotation error about the z axis.
        '''
        #Get the path info of the current path
        pathPos = self._pathPoints     
        pathLen = self._pathLengths
        
        pos, rot = self.getEgoPoseAbsolute() #Find the vehicle's current pose

        posAlongPath = self._sim.getClosestPosOnPath(pathPos, pathLen, pos)
        nearPt = self._sim.getPathInterpolatedConfig(pathPos, pathLen, posAlongPath) #Convert the position along path to an XYZ position in the path's frame
        
        pathError = np.subtract(pos, nearPt)

        lookAhead = 0.1
        nextPos = posAlongPath + lookAhead
        
        #Compare to the final path position, wrap around if beyond that
        if nextPos > self._totalLength:
            nextPos = remainder(nextPos, self._totalLength)
        
        nextNearPt = self._sim.getPathInterpolatedConfig(pathPos, pathLen, nextPos)
        
        pathVector = np.subtract(nextNearPt, nearPt)
        pathTrajectory = atan2(pathVector[1], pathVector[0])
        orientErr =  helper.pipi(pathTrajectory - rot[2] + pi)
                
        # print(orientErr)
        return pathError, orientErr
    
    def getVehicleState(self):
        '''
        Function to package the current vehicle state into a dictionary
        '''
        pos, rot = self.getEgoPoseAbsolute()
        v = self.getVehicleSpeed()
        d = self.getSteeringAngle()
        
        X = {"Position": pos[0:1],
             "Orientation": rot[2],
             "Speed": v,
             "Steering": d }
        
        return X
    
    def __del__(self):
        '''
        Destructor
        '''
        if self._initialized:
            if self._isRunning:
                self.stopSimulation()

            self._initialized = False
        pass

class helper:
    def bound(a, aMin, aMax):
        return max(min(a, aMax), aMin)
    
    def pipi(a): 
        a = remainder(a, 2*pi)       
        return (2*pi - a) if a > pi else a