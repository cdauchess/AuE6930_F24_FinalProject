#  python -m pip install coppeliasim-zmqremoteapi-client
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import math
import random

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
    def __init__(self,numPaths):
        self._initialized = True
        self._isRunning = False
        
        self._client = RemoteAPIClient()        
        self._sim = self._client.require('sim')

        self._sim.setStepping(True)
        
        self._world = self._sim.getObject('/Floor')
        
        #These are the handles for the motorbike in the simulation environment
        #self._egoVehicle = self._sim.getObject('/Motorbike/CoM')
        #self._speedMotor = self._sim.getObject('/Motorbike/rearSuspension/motor')
        #self._steerMotor = self._sim.getObject('/Motorbike/steeringMotor')
        #These are the handles for the "Manta Vehicle"
        self._egoVehicle = self._sim.getObject('/Manta/body_dummy')
        self._speedMotor = self._sim.getObject('/Manta/motor_joint')
        self._steerMotor = self._sim.getObject('/Manta/steer_joint')
        
        # Store initial state
        self._initial_position = self._sim.getObjectPosition(self._egoVehicle, self._world)
        self._initial_orientation = self._sim.getObjectOrientation(self._egoVehicle, self._world)
        
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
            
        self.activePath = 0 # Current path being followed by the vehicle
        self.setMaxSteerAngle()

        # Episode management
        self.episode_count = 0
        self.max_episode_steps = 15  # Maximum steps per episode
        self.current_episode_steps = 0
        
        # Randomization ranges for initial state
        self.position_range = 1.0  # meters
        self.orientation_range = 0.5  # radians

        #pos = self._sim.getObjectPosition(C)

    def resetSimulation(self, randomize=True):
        """
        Reset the simulation to initial state, optionally with randomization
        Returns: Initial state dictionary
        """
        self.stopSimulation()
        self._sim.setStepping(True)
        self.startSimulation()
        
        if randomize:
            # Randomize initial position
            random_offset = [
                random.uniform(-self.position_range, self.position_range),
                random.uniform(-self.position_range, self.position_range),
                0  # Keep Z position constant
            ]
            new_position = [
                self._initial_position[0] + random_offset[0],
                self._initial_position[1] + random_offset[1],
                self._initial_position[2]
            ]
            
            # Randomize initial orientation (only yaw)
            random_yaw = random.uniform(-self.orientation_range, self.orientation_range)
            new_orientation = [
                self._initial_orientation[0],
                self._initial_orientation[1],
                self._initial_orientation[2] + random_yaw
            ]
        else:
            new_position = self._initial_position
            new_orientation = self._initial_orientation
        
        # Set position and orientation
        self._sim.setObjectPosition(self._egoVehicle, self._world, new_position)
        self._sim.setObjectOrientation(self._egoVehicle, self._world, new_orientation)
        
        # Reset motors
        self.setSpeed(0)
        self.setSteering(0)
        
        # Reset episode counters
        self.current_episode_steps = 0
        self.episode_count += 1
        
        # Return initial state
        return self.getVehicleState()
    
    def startEpisode(self, randomize=True):
        """
        Start a new episode
        Returns: Initial state dictionary
        """
        return self.resetSimulation(randomize)
    
    def isEpisodeFinished(self):
        """
        Check if current episode should end
        Returns: Boolean indicating if episode is done
        """
        # Get current state
        pos, _ = self.getEgoPoseWorld()
        path_error, _ = self.getPathError(self.activePath)
        
        # Define termination conditions
        max_steps_reached = self.current_episode_steps >= self.max_episode_steps
        off_track = abs(path_error[0]) > 5.0 or abs(path_error[1]) > 5.0
        
        return max_steps_reached or off_track
    
    def stepEpisode(self, speed, steering):
        """
        Perform one episode step with given controls
        Returns: (new_state, done)
        """
        # Apply controls
        self.setSpeed(speed)
        self.setSteering(steering)
        
        # Step simulation
        self.stepTime()
        self.current_episode_steps += 1
        
        # Check if episode is finished
        done = self.isEpisodeFinished()
        
        # Get new state
        new_state = self.getVehicleState()
        
        return new_state, done
    
    def setRandomizationRanges(self, position_range=None, orientation_range=None):
        """
        Set the ranges for initial state randomization
        """
        if position_range is not None:
            self.position_range = position_range
        if orientation_range is not None:
            self.orientation_range = orientation_range
            
    def getEpisodeStats(self):
        """
        Get current episode statistics
        """
        return {
            'episode_number': self.episode_count,
            'current_step': self.current_episode_steps,
            'max_steps': self.max_episode_steps
        }

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

    def getEgoPose(self, frame):
        '''
        Returns the vehicle's pose in a given frame
        '''
        pos = self._sim.getObjectPosition(self._egoVehicle, frame)
        rot = self._sim.getObjectOrientation(self._egoVehicle, frame)

        return pos,rot
    
    def getEgoPoseWorld(self):
        
        return self.getEgoPose(self._world)
    
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
        self._sim.setJointTargetForce(self._speedMotor, 60) #Setting motor torque
        self._sim.setJointTargetVelocity(self._speedMotor,speedTarget)
        
    def getSpeed(self):
        '''
        Get the current motor speed
        '''
        return self._sim.getJointVelocity(self._speedMotor)

    def setSteering(self,steerTarget):
        '''
        Set the steering target
        '''
        #Bound the steering angle
        if steerTarget>self._maxSteerAngle:
            steerTarget = self._maxSteerAngle
        elif steerTarget<(-1*self._maxSteerAngle):
            steerTarget = -1*self._maxSteerAngle

        self._sim.setJointTargetPosition(self._steerMotor,steerTarget)
        
    def getSteeringAngle(self):
        '''
        Get the current steering angle
        '''
        return self._sim.getJointPosition(self._steerMotor)
        
    def setMaxSteerAngle(self,maxAngle = 0.523599):
        '''
        Set the maximum steering angle allowed, in radians
        '''
        self._maxSteerAngle = maxAngle

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
        '''
        Return the error relative to the closest point along the target path
        Returns orientation error as the rotation error about the z axis.
        '''
        #Get the path info of the path in question
        pathPos = self._pathPositions[pathNum]
        pathQuaternion = self._pathQuaternions[pathNum]
        pathLen = self._pathLengths[pathNum]
        
        currPos,currOrient = self.getEgoPose(self._path[self.activePath]) #Find the vehicle's current pose
        posAlongPath = self._sim.getClosestPosOnPath(pathPos, pathLen, currPos)
        nearestPoint = self._sim.getPathInterpolatedConfig(pathPos, pathLen, posAlongPath)#Convert the position along path to an XYZ position in the path's frame
        pathError = np.subtract(currPos,nearestPoint)

        #Find Orientation Error relative to the chosen path
        #Currently a coordinate system issue in orientation error. 
        lookAhead = 0.1
        nextPos = posAlongPath+lookAhead
        #Compare to the final path position, wrap around if beyond that
        if nextPos > pathPos[-1]:
            nextPos = pathPos[0]
        nextNearestPoint = self._sim.getPathInterpolatedConfig(pathPos, pathLen, nextPos)
        
        pathVector = np.subtract(nextNearestPoint,nearestPoint)
        pathTrajectory = math.tan(pathVector[1]/pathVector[0])
        orientErr = currOrient[2]-pathTrajectory
        orientErr = currOrient[2]+math.pi
        
        orientErr = None
        
        return pathError,orientErr
    
    def getVehicleState(self):
        '''
        Function to package the current vehicle state into a dictionary
        '''
        pos,orient = self.getEgoPoseWorld()
        speed = self.getSpeed()
        steer = self.getSteeringAngle()
        
        vehState = {
            "Position": pos,
            "Orientation": orient,
            "Speed": speed,
            "Steering": steer}
        
        return vehState
    
    def __del__(self):
        '''
        Destructor
        '''
        if self._initialized:
            if self._isRunning:
                self.stopSimulation()

            self._initialized = False
        pass