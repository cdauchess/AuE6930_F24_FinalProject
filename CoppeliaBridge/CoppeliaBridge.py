#  python -m pip install coppeliasim-zmqremoteapi-client
#  python -m pip install scikit-learn
#  python -m pip install -U scikit-image
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import matplotlib.pyplot as plt

from skimage.draw import polygon
from math import pi, tan, atan2, radians, remainder
plt.ion()
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

        self._fig = plt.figure()
        self._plotter = self._fig.add_subplot(121)
        self._img     = self._fig.add_subplot(122)
        # self._img.invert_yaxis()

        self._plotter.set_xlim(-1.5, 1.5)
        self._plotter.set_ylim(-1.5, 1.5)     

        # self._img.set_xlim(0, 360)
        # self._img.set_ylim(0, 360)        
        
    def initEgo(self):
        # Motion Constraints
        self._V_MAX = 3                 # max speed [m/s]
        self._ACC_MAX = 5               # max acceleration [m/s2]
        self._D_MAX = radians(26)       # max steering angle
        
        # Sensor Constants
        self._SENSOR_FOV = 90            # deg
        self._SENSOR_RES = 0.5           # deg

        # Vehivle Dimensions
        self._m = 2.57                  # mass [Kg]
        self._r = 0.033                 # tire radius [m]        
        
        self._l = 0.258                 # wheel Base [m]
        self._t = 0.157                 # track width
        self._tf = 0.112                # front kingpin width

        self._a = 0.126                 # distance from CG to front axle
        self._b = self._l - self._a     # distance from CG to rear axle
        
        # Vehicle State
        self._v = 0                     # speed
        self._d = 0                     # steering Angle
        
        self._ego = self._sim.getObject('/qcar/CoM')
        self._front = self._sim.getObject('/qcar/Front')

        self._lDriveWheel = self._sim.getObject('/qcar/base_wheelrl_joint')
        self._rDriveWheel = self._sim.getObject('/qcar/base_wheelrr_joint')        

        self._lSteerAxis = self._sim.getObject('/qcar/base_hubfl_joint')
        self._rSteerAxis = self._sim.getObject('/qcar/base_hubfr_joint')    

        self._leftSensor = self._sim.getObject('/qcar/left_lidar')
        self._frontSensor = self._sim.getObject('/qcar/front_lidar')
        self._rightSensor = self._sim.getObject('/qcar/right_lidar')
        self._rearSensor = self._sim.getObject('/qcar/rear_lidar')

        self._isEgoReady = True  

        self.setSteering(self._d) 
        self.setVehicleSpeed(self._v) 

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
    
    def getOccupancyGrid(self):        
        maxDist = 1.5
        plot = False

        rawDataL, resL = self._sim.getVisionSensorDepth(self._leftSensor, 1)
        rawDataF, resF = self._sim.getVisionSensorDepth(self._frontSensor, 1)
        rawDataR, resR = self._sim.getVisionSensorDepth(self._rightSensor, 1)
        rawDataB, resB = self._sim.getVisionSensorDepth(self._rearSensor, 1)
        
        divs = np.arange(-maxDist, maxDist, 2*maxDist/resL[0]) / maxDist

        # 2 rows of data                
        xL = -np.reshape(self._sim.unpackFloatTable(rawDataL), [resL[1], -1])[1]                
        yL = -xL * divs
        
        yF = np.reshape(self._sim.unpackFloatTable(rawDataF), [resF[1], -1])[1]
        xF =  yF * divs

        xR = np.reshape(self._sim.unpackFloatTable(rawDataR), [resR[1], -1])[1]
        yR = -xR * divs
       
        yB = -np.reshape(self._sim.unpackFloatTable(rawDataB), [resB[1], -1])[1]
        xB =  yB * divs
                
        x = np.append(np.append(np.append(xL, xF), xR), xB)
        y = np.append(np.append(np.append(yL, yF), yR), yB)

        xImg = np.array((x + maxDist) * resL[0] / (2 * maxDist), dtype=int)
        yImg = np.array(resL[0]- ((y + maxDist) * resL[0] / (2 * maxDist)), dtype=int)

        og = np.zeros((resL[0], resL[0]), 'uint8')
        rr, cc = polygon(xImg, yImg, og.shape)
        og[cc, rr] = 1

        if(plot):
            self._plotter.clear()                            
            self._plotter.plot(x, y, 'k')            
            self._plotter.axis('square')

            self._img.imshow(og, origin='upper')                
            
            plt.draw()
            plt.pause(.001)   

        return og

    def getPose(self, object, frame):
        '''
        Returns the vehicle's pose wrt a given frame
        '''
        pos = self._sim.getObjectPosition(object, frame)
        rot = self._sim.getObjectOrientation(object, frame)

        return pos, rot
    
    def getEgoPoseRelative(self, frame):
        '''
        Returns the vehicle's pose wrt a given frame
        '''       
        return self.getPose(self._ego, frame)
    
    def getEgoPoseAbsolute(self):        
        '''
        Returns the vehicle's absolute pose
        '''
        return self.getEgoPoseRelative(self._world)
    
    def getVehicleSpeed(self):
        '''
        Get the current motor speed
        '''
        vl = -self._sim.getJointVelocity(self._lDriveWheel)
        vr =  self._sim.getJointVelocity(self._rDriveWheel)
        
        return(self._r*(vl + vr)/2)        
    
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
        
        pos, rot = self.getPose(self._front, self._world) #Find the pose of the center of the front axle

        posAlongPath = self._sim.getClosestPosOnPath(pathPos, pathLen, pos)
        nearPt = self._sim.getPathInterpolatedConfig(pathPos, pathLen, posAlongPath) #Convert the position along path to an XYZ position in the path's frame
        
        pathError = np.subtract(nearPt, pos)

        lookAhead = 0.1
        nextPos = posAlongPath + lookAhead
        
        #Compare to the final path position, wrap around if beyond that
        if nextPos > self._totalLength:
            nextPos = remainder(nextPos, self._totalLength)
        
        nextNearPt = self._sim.getPathInterpolatedConfig(pathPos, pathLen, nextPos)
        
        pathVector = np.subtract(nextNearPt, nearPt)
        pathTrajectory = atan2(pathVector[1], pathVector[0])        
        # print(pathTrajectory)
        # print(rot[2])
        orientErr =  helper.pipi(pathTrajectory - rot[2] + pi)

        pErr = np.sqrt(np.sum(np.power(pathError[0:2], 2))) * (pathError[1]/abs(pathError[1]))        
        # print(orientErr)
        return pErr, orientErr
    
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