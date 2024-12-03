#  python -m pip install coppeliasim-zmqremoteapi-client
#  python -m pip install scikit-learn
#  python -m pip install -U scikit-image
#  python -m pip install scikit-learn
#  python -m pip install -U scikit-image
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import matplotlib.pyplot as plt

from skimage.draw import polygon, line, bezier_curve
from math import pi, tan, atan2, radians, remainder, atan, sin, cos
# plt.ion()


# This class is a bridge between Coppelia Sim and Python
class CoppeliaBridge:
    def __init__(self):
        self._initialized = True
        self._isRunning = False
        self._isEgoReady = False
        
        self._client = RemoteAPIClient()
        self._client.getObject("sim").loadScene("/home/kvadner/Desktop/AuE6930_F24_FinalProject/Scenes/QScene5.ttt")       
        self._sim = self._client.require('sim')
        
        self._sim.setStepping(True)
        self._timeStep = self.getTimeStepSize()
        
        self._world = self._sim.getObject('/Floor')

        self._plot = False

        self.initEgo()
        self.initScene()
        
    def initEgo(self):
        # Motion Constraints
        self._V_MAX = 1.5                 # max speed [m/s]
        self._ACC_MAX = 3               # max acceleration [m/s2]
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
        
        self._ego = self._sim.getObject('/qcar')
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

        self.__iscrazy = False

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
    def startSimulation(self, renderEnable = False):
        '''
        Starts CoppeliaSim simulator
        '''
        self.renderState(renderEnable)
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
    
    def setSimStepping(self, value):
        self._sim.setStepping(value)

    def getTime(self):
        '''
        Get current time in simulation
        '''
        return self._sim.getSimulationTime()
        
    #Path Reference: https://manual.coppeliarobotics.com/en/paths.htm
    def getPathData(self, pathNum):
        '''
        Get the information about the chosen path
        '''
        #Assuming Path info is a list of position+quaterion (7 items per point) 
        self.rawPathInfo = self._sim.unpackDoubleTable(self._sim.readCustomBufferData(self._lanes[pathNum],'PATH'))
        pathInfo = np.reshape(self.rawPathInfo,(-1,7))
        pathPositions = pathInfo[:, :3].flatten().tolist() #XYZ Positions of each path point
        pathQuaternions = pathInfo[:, 3:].flatten().tolist() #Path point orientations
        pathLengths, self.totalLength = self._sim.getPathLengths(pathPositions, 3) #Length along path for each path point
        
        return pathPositions, pathQuaternions, pathLengths    
    
    def setInitPosition(self,pathNum=0,startPos=0):
        '''
        Sets the vehicles position to a position along the selected path
        pathNum: index for which path the vehicle will be place upon (0 indexed)
        startPos: valid values 0-1 (float) with 0 being the beginning of the path and 1 being the end of the path. For a closed path 0 and 1 are nearly the same position
        '''
        
        #pathPos = self._pathPositions[pathNum]
        #pathQuaternion = self._pathQuaternions[pathNum]
        #athLen = self._pathLengths[pathNum]
        
        pathPos,pathQuaternion,pathLen = self.getPathData(pathNum)
        
        #Map starting position to valid values
        startPathPos = startPos*max(pathLen)
        
        startPoint = self._sim.getPathInterpolatedConfig((pathPos), pathLen, startPathPos) # Convert the position along path to an XYZ position in the path's frame
        startPoint[2] = startPoint[2] +.2 #Offset in Z to keep vehicle above ground plane.

        
        lookAhead = 0.1
        nextPos = startPathPos+lookAhead
        #Compare to the final path position, wrap around if beyond that
        if nextPos > pathLen[-1]:
            nextPos = nextPos-pathLen[-1]
        nextNearestPoint = self._sim.getPathInterpolatedConfig(pathPos, pathLen, nextPos)
        
        pathVector = np.subtract(nextNearestPoint,startPoint)
        pathTrajectory = atan(pathVector[1]/pathVector[0])
        
        #Set the vehicle to the starting point
        self._sim.setObjectPosition(self._ego,startPoint[:3],self._lanes[pathNum])
        self._sim.setObjectOrientation(self._ego,[0,0,pathTrajectory],self._lanes[pathNum])
        
    
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

    def isRunning(self):
        '''
        Returns True if sim is running
        '''
        return self._isRunning
    
    def renderState(self, render):
        '''
        Set if rendering is enabled or disabled
        False = rendering disabled
        True = rendering enabled
        '''
        #self._sim.setBoolParam(self._sim.boolparam_display_enabled, render)
        pass

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

            dl = sign * atan2(self._l, (R - sign * self._tf/2))
            dr = sign * atan2(self._l, (R + sign * self._tf/2))
        
        # print("Left : {}, Right: {}".format(dl, dr))

        self._sim.setJointTargetPosition(self._lSteerAxis, dl)
        self._sim.setJointTargetPosition(self._rSteerAxis, dr)        

        self._adjustDifferential() # what are the steering bounds?
    
    def pathForOG(self, ogDim = (180,180), maxDist=1.5):
        '''
        Return points in the occupancy grid that represent the path
        '''
        ogSizeX = ogDim[0]
        ogSizeY = ogDim[1]
        MperPx = (ogSizeX/2)/maxDist
        pathError, orientError = self.getPathErrorPos()
        
        xPt = [pathError[0]]
        yPt = [pathError[1]]
        
        #Calculate the next point on the path relative to the vehicle
        xPt.append(self._lookAhead*sin(orientError)+xPt[0])
        yPt.append(self._lookAhead*cos(orientError)+yPt[0])
        
        #Cast to int and scale to pixel frame
        xPt = [int((val*MperPx)+(ogSizeX/2)) for val in xPt]
        yPt = [int((-1*val*MperPx)+(ogSizeY/2)) for val in yPt] #Negate Y due to occupancy grid more positive Y being behind the vehicle.
        
        #Bound the points to the size of the occupancy grid
        xPt = np.clip(xPt,0,ogSizeX-1)
        yPt = np.clip(yPt,0,ogSizeY-1)
        
        rr,cc = line(xPt[0], yPt[0], xPt[1], yPt[1])
        
        return rr,cc
    
    def pathPosLookAhead(self, LkAhd = 1.4, numPts = 5):
        '''
        Returns the path positions relative to the vehicle evenly spaced up to LkAhd meters ahead along the path
        '''
        xPt = []
        yPt = []
        #Get the path info of the current path
        pathPos = self._pathPoints     
        pathLen = self._pathLengths
        
        lkAhdPts = np.linspace(-LkAhd,LkAhd,numPts)
        
        pos, rot = self.getPose(self._front, self._lanes[0]) #Find the pose of the center of the front axle
        posAlongPath = self._sim.getClosestPosOnPath(pathPos, pathLen, pos)
        
        for pt in lkAhdPts:
            nextPos = posAlongPath+pt
            #Compare to the final path position, wrap around if beyond that
            if nextPos > self._totalLength:
                nextPos = remainder(nextPos, self._totalLength)
            pathPt = self._sim.getPathInterpolatedConfig(pathPos, pathLen, nextPos) #Convert the position along path to an XYZ position in the path's frame
            pathError = np.subtract(pathPt, pos)
            xPt.append(pathError[0])
            yPt.append(pathError[1])

        return xPt,yPt
    
    def pathOGLkAhdBz(self, ogDim = (180,180), maxDist=1.5):
        '''
        Returns path information on the occupancy grid in a series of bezier curves
        '''        
        ogSizeX = ogDim[0]
        ogSizeY = ogDim[1]
        MperPx = (ogSizeX/2)/maxDist
        
        numPts = 5
        weight = 1
        
        img = np.zeros(ogDim, dtype=np.uint8)
        xPt,yPt = self.pathPosLookAhead(LkAhd=maxDist,numPts=numPts)
        __, orientErr = self.getPathErrorPos()
        
        for n in range(numPts):
            xPt[n], yPt[n] = helper.rotatePoint(xPt[n], yPt[n], -1*orientErr)        
        
        #Cast to int and scale to pixel frame
        xPt = [int((val*MperPx)+(ogSizeX/2)) for val in xPt]
        yPt = [int((-1*val*MperPx)+(ogSizeY/2)) for val in yPt] #Negate Y due to occupancy grid more positive Y being behind the vehicle.

        #Bound the points to the size of the occupancy grid
        xPt = np.clip(xPt,0,ogSizeX-1)
        yPt = np.clip(yPt,0,ogSizeY-1)
        
        for n in range(0,numPts-1,2):
            rr,cc = bezier_curve(xPt[n],yPt[n],xPt[n+1],yPt[n+1],xPt[n+2],yPt[n+2],weight=weight,shape=ogDim)
            img[cc,rr] = 2
            
        
        return img
    
    def pathOGLkAhdLin(self, ogDim = (180,180), maxDist=1.5):
        '''
        Returns path information on the occupancy grid in a series of line segments
        '''
        ogSizeX = ogDim[0]
        ogSizeY = ogDim[1]
        MperPx = (ogSizeX/2)/maxDist
        
        numPts = 5
        
        img = np.zeros(ogDim, dtype=np.uint8)
        xPt,yPt = self.pathPosLookAhead(LkAhd=maxDist,numPts=numPts)
        __, orientErr = self.getPathErrorPos()

        for n in range(numPts):
            xPt[n], yPt[n] = helper.rotatePoint(xPt[n], yPt[n], -orientErr + np.pi/2)                   
        
        #Cast to int and scale to pixel frame
        xPt = [int((val*MperPx)+(ogSizeX/2)) for val in xPt]
        yPt = [int((-1*val*MperPx)+(ogSizeY/2)) for val in yPt] #Negate Y due to occupancy grid more positive Y being behind the vehicle.
    
        #Bound the points to the size of the occupancy grid
        xPt = np.clip(xPt,0,ogSizeX-1)
        yPt = np.clip(yPt,0,ogSizeY-1)
        
        for n in range(0,numPts-1,1):
            rr,cc = line(xPt[n],yPt[n],xPt[n+1],yPt[n+1])
            img[cc,rr] = 2
        
        return img
    
    def getOccupancyGridOld(self):  
        '''
        Returns a matrix of 0s and 1s. 1s indicate obstacles. 0s indicate free space
        2 indicates path position
        3 indicates path and obstacle occupying the same space
        '''      
        maxDist = 1.5
        
        rawDataL, resL = self._sim.getVisionSensorDepth(self._leftSensor, 1)
        rawDataF, resF = self._sim.getVisionSensorDepth(self._frontSensor, 1)
        rawDataR, resR = self._sim.getVisionSensorDepth(self._rightSensor, 1)
        rawDataB, resB = self._sim.getVisionSensorDepth(self._rearSensor, 1)
        
        divs = np.arange(-maxDist, maxDist, 2*maxDist/resL[0]) / maxDist

        # 2 rows of data                
        xL = -np.reshape(self._sim.unpackFloatTable(rawDataL), [resL[1], -1])
        yL = -xL * divs
        
        yF = np.reshape(self._sim.unpackFloatTable(rawDataF), [resF[1], -1])
        xF =  yF * divs

        xR = np.reshape(self._sim.unpackFloatTable(rawDataR), [resR[1], -1])
        yR = -xR * divs

        yB = -np.reshape(self._sim.unpackFloatTable(rawDataB), [resB[1], -1])
        xB =  yB * divs
                
        x = np.append(np.append(np.append(xL, xF), xR), xB)
        y = np.append(np.append(np.append(yL, yF), yR), yB)

        xImg = np.array((x + maxDist) * resL[0] / (2 * maxDist), dtype=int)
        yImg = np.array(resL[0]- ((y + maxDist) * resL[0] / (2 * maxDist)), dtype=int)

        og = np.ones((resL[0], resL[0]), 'uint8')
        rr, cc = polygon(xImg, yImg, og.shape)
        og[cc, rr] = 0
        
        #Add path information to the occupancy grid
        #rrP, ccP = self.pathForOG(np.shape(og),maxDist)
        #og[ccP,rrP]+=2
        tempImg = self.pathOGLkAhdLin(np.shape(og),maxDist)
        og+= tempImg

        if(self._plot):
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
    
    def setVehiclePose(self, position, orientation):
        '''
        Set vehicle position and orientation in the world frame
        '''
        self._sim.setObjectPosition(self._ego, self._world, position)
        self._sim.setObjectOrientation(self._ego, self._world, orientation)        

    def resetVehicle(self):
        '''
        Reset vehicle controls
        '''
        self.setVehicleSpeed(0)
        self.setSteering(0)

    def getOccupancyGrid(self):  
        '''
        Returns a multi-channel occupancy grid with shape (channels, H, W)
        Channel 0: Binary obstacles (0: free, 1: occupied)
        Channel 1: Distance to obstacles
        Channel 2: Path information
        Channel 3: Distance to path
        '''      
        maxDist = 1.5
        
        # Get sensor data
        rawDataL, resL = self._sim.getVisionSensorDepth(self._leftSensor, 1)
        rawDataF, resF = self._sim.getVisionSensorDepth(self._frontSensor, 1)
        rawDataR, resR = self._sim.getVisionSensorDepth(self._rightSensor, 1)
        rawDataB, resB = self._sim.getVisionSensorDepth(self._rearSensor, 1)
        
        divs = np.arange(-maxDist, maxDist, 2*maxDist/resL[0]) / maxDist

        # 2 rows of data                
        xL = -np.reshape(self._sim.unpackFloatTable(rawDataL), [resL[1], -1])
        yL = -xL * divs
        
        yF = np.reshape(self._sim.unpackFloatTable(rawDataF), [resF[1], -1])
        xF =  yF * divs

        xR = np.reshape(self._sim.unpackFloatTable(rawDataR), [resR[1], -1])
        yR = -xR * divs

        yB = -np.reshape(self._sim.unpackFloatTable(rawDataB), [resB[1], -1])
        xB =  yB * divs
                    
        x = np.append(np.append(np.append(xL, xF), xR), xB)
        y = np.append(np.append(np.append(yL, yF), yR), yB)

        xImg = np.array((x + maxDist) * resL[0] / (2 * maxDist), dtype=int)
        yImg = np.array(resL[0]- ((y + maxDist) * resL[0] / (2 * maxDist)), dtype=int)

        # Channel 0: Binary obstacle map
        obstacle_channel = np.ones((resL[0], resL[0]), 'uint8')
        rr, cc = polygon(xImg, yImg, obstacle_channel.shape)
        obstacle_channel[cc, rr] = 0

        # Channel 1: Distance transform for obstacles
        from scipy.ndimage import distance_transform_edt
        distance_channel = distance_transform_edt(1 - obstacle_channel)
        # Normalize relative to vehicle position (center of grid)
        center_y, center_x = resL[0]//2, resL[0]//2
        Y, X = np.ogrid[:resL[0], :resL[0]]
        # Adjust distances based on distance from center
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2) / (resL[0]/2)
        distance_channel = distance_channel * (1 - 0.5*dist_from_center)
        distance_channel = distance_channel / distance_channel.max()

        # Channel 2: Enhanced path information with wider path
        path_channel = np.zeros((resL[0], resL[0]), 'uint8')
        # Get base path points
        path_points = self.pathOGLkAhdLin(np.shape(path_channel), maxDist)
        # Dilate the path to make it wider
        from scipy.ndimage import binary_dilation
        path_kernel = np.ones((5,5))  # Adjust size for desired path width
        path_channel = binary_dilation(path_points > 0, structure=path_kernel)
        
        # Channel 3: Distance to path
        path_distance = distance_transform_edt(1 - path_channel)
        path_distance = path_distance / path_distance.max()

        # Stack all channels
        multi_channel_grid = np.stack([
            obstacle_channel,
            distance_channel,
            path_channel,
            path_distance
        ])

        if(self._plot):
            self._plot_occupancy_grid(multi_channel_grid, x, y)

        return multi_channel_grid

    def _plot_occupancy_grid(self, grid, raw_x=None, raw_y=None):
        '''
        Enhanced visualization of the multi-channel occupancy grid
        '''
        if not hasattr(self, '_fig') or not plt.fignum_exists(self._fig.number):
            self._fig = plt.figure(figsize=(15, 5))
            plt.ion()
        
        # Clear previous plots
        plt.clf()
        
        # Plot raw sensor data if available
        if raw_x is not None and raw_y is not None:
            ax_raw = plt.subplot(151)
            ax_raw.plot(raw_x, raw_y, 'k.')
            ax_raw.set_title('Raw Sensor Data')
            ax_raw.axis('equal')
            ax_raw.grid(True)
        
        # Plot each channel
        titles = ['Obstacles', 'Distance Field', 'Path', 'Path Distance']
        for i, (channel, title) in enumerate(zip(grid, titles)):
            ax = plt.subplot(1, 5, i+2)
            if i == 0:  # Binary obstacle map
                im = ax.imshow(channel, cmap='gray_r', origin='upper')
            elif i == 2:  # Path
                im = ax.imshow(channel, cmap='Greens', origin='upper')
            else:  # Distance fields
                im = ax.imshow(channel, cmap='viridis', origin='upper')
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
    
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
        '''
        Switches Lane
        '''          
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

        # print("Left: {}, Right: {}".format(-vl, vr))

        self._sim.setJointTargetVelocity(self._lDriveWheel, -vl/self._r)
        self._sim.setJointTargetVelocity(self._rDriveWheel,  vr/self._r)
        
    def setLookAhead(self,lookahead):
        if lookahead < 0:
            self._lookAhead = 0
        else:
            self._lookAhead = lookahead
            
    def getLookAhead(self):
        return self._lookAhead
        
    def getPathErrorPos(self):
        '''
        Return the error relative to the closest point along the target path (XYZ component error)
        Returns orientation error as the rotation error about the z axis.
        '''
        #Get the path info of the current path
        pathPos = self._pathPoints     
        pathLen = self._pathLengths
        
        pos, rot = self.getPose(self._front, self._lanes[0]) #Find the pose of the center of the front axle

        posAlongPath = self._sim.getClosestPosOnPath(pathPos, pathLen, pos)
        nearPt = self._sim.getPathInterpolatedConfig(pathPos, pathLen, posAlongPath) #Convert the position along path to an XYZ position in the path's frame
        
        pathError = np.subtract(nearPt, pos)

        self._lookAhead = 0.25
        nextPos = posAlongPath + self._lookAhead
        
        #Compare to the final path position, wrap around if beyond that
        if nextPos > self._totalLength:
            nextPos = remainder(nextPos, self._totalLength)
        
        nextNearPt = self._sim.getPathInterpolatedConfig(pathPos, pathLen, nextPos)
        
        pathVector = np.subtract(nextNearPt, nearPt)
        pathTrajectory = atan2(pathVector[1], pathVector[0])        
        # print(pathTrajectory)
        # print(rot[2])
        orientErr = -1*helper.pipi(pathTrajectory - rot[2])
        #print(orientErr)

        return pathError, orientErr
    
    def getPathError(self):
        '''
        Returns the norm of the XYZ path error
        '''
        
        pathError, orientErr = self.getPathErrorPos()
        
        pErr = np.linalg.norm(pathError[0:2])
        
        #if pathError[1] < 0: #Assign side of the path that we're on.
        #    pErr*=-1 
        return pErr, orientErr
    
    def vehicleCollection(self):
        self.vehicleCol =self._sim.createCollection(0)
        self._sim.addItemToCollection(self.vehicleCol,self._sim.handle_tree, self._egoVehicle,0)
    
    def getCollision(self, vehicle, object = None):
        '''
        Check if the specified vehicle has collided with the specified object. Leave object empty if it is to be checked against everything in the environment
        Specific object handling is WIP
        '''
        if object == None: #If nothing specified, check against all objects
            object = self._sim.handle_all
        res, dist, point, obj, n = self._sim.handleProximitySensor(self._proxSens)
        
        return res
    
    def checkEgoCollide(self, og):
        '''
        Check is the ego vehicle has collided with anything in the environment
        Pass in the occupancy grid. Occupancy grid calculated each time for performance reasons
        '''
        
        
        # Extract just the obstacle channel (channel 0)
        obstacle_channel = og[0]
        
        # TODO: Confirm vehicle dimensions in the occupancy grid 70:110 is a bit of buffer
        vehicle_area = obstacle_channel[70:110,70:110]
        
        # Check if any obstacles (1's) are present in the vehicle area
        return np.any(vehicle_area == 1)
    
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

    def rotatePoint(x,y,angle):
        xprime = x*cos(angle)-y*sin(angle)
        yprime = x*sin(angle)+y*cos(angle)
        
        return xprime,yprime