import numpy as np
from numpy import dot
from scipy.linalg import inv, block_diag

import pdb

class FirstOrderRCLowPassFilter():
    def __init__(self):
        self.alpha_ = 0.0
        self.inited_ = False
        self.state_ = np.array([0.0, 0.0])
    
    def SetAlpha(self, alpha):
        self.alpha_ = alpha
        self.inited_ = False

    def AddMeasure(self, z):
        if self.inited_:
            self.state_ = z + self.alpha_ * (self.state_ - z)
        else:
            self.state_ = z
            self.inited_ = True

    def AddMeasure_noinput(self):
        z = self.state_
        self.AddMeasure(z)


    def get_state(self):
        return self.state_

    def isInited(self):
        return self.inited_


class Tracker_center(): # kalman filter which only track the center of bbox
    def __init__(self):
        self.inited_ = False
        self.id = 0 # tracker's id 
        self.obj = {} # object information

        self.hits = 0 # number of detection matches
        self.no_losses = 0 # number of unmatched tracks (track loss)

        # Initialize parameters for Kalman Filtering
        # The state is the (x, y) coordinates of the center of detection box
        # state: [center_c, center_c_dot, center_r, center_r_dot]

        self.x_state_ = []
        self.whRCF = FirstOrderRCLowPassFilter()
        self.whRCF.SetAlpha(0.5)
        self.dt = 1

        #state transition matrix F
        self.F = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.F[0, 1] = self.dt
        self.F[2, 3] = self.dt

        # Measurement matrix H, assuming we can only measure the coordinates
        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])
        
        # Initialize the state covariance P
        self.L = 10.0  #10.0 #no change
        self.P = np.diag(self.L * np.ones(4))

        # Initialize the process covariance
        self.Q_comp_mat = np.array([[self.dt**4/4., self.dt**3/2.],
                                    [self.dt**3/2., self.dt**2]])
        self.Q = block_diag(self.Q_comp_mat, self.Q_comp_mat)
        
        # Initialize the measurement covariance
        self.R_scaler = 1.0 #1.0
        self.R_diag_array = self.R_scaler * np.array([self.L, self.L])
        self.R = np.diag(self.R_diag_array)

    def Init(self, x, wh):
        self.x_state_ = x
        self.inited_ = True
        self.whRCF.AddMeasure(wh)

    def update_R(self):   
        R_diag_array = self.R_scaler * np.array([self.L, self.L])
        self.R = np.diag(R_diag_array)

    def isInited(self):
        if not self.inited_:
            return False
        if not self.whRCF.isInited():
            return False
        return True

    def get_x_state(self):
        if not self.isInited():
            raise ValueError('tracker not initiated.')

        return self.x_state_

    def kalman_filter(self, z, wh): 
        '''
        Implement the Kalman Filter, including the predict and the update stages,
        with the measurement z
        '''
        if not self.isInited():
            raise ValueError('tracker not initiated.')

        x = self.x_state_.astype('float')
        # Predict
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

        #Update
        S = dot(self.H, self.P).dot(self.H.T) + self.R
        K = dot(self.P, self.H.T).dot(inv(S)) # Kalman gain
        y = z - dot(self.H, x) # residual

        x += dot(K, y)
        self.P = self.P - dot(K, self.H).dot(self.P)
        self.x_state_ = x.astype(int) # convert to integer coordinates 
                                     #(pixel values)

        self.whRCF.AddMeasure(wh)
        
    def predict_only(self):  
        '''
        Implment only the predict stage. This is used for unmatched detections and 
        unmatched tracks
        '''
        if not self.isInited():
            raise ValueError('tracker not initiated.')
        x = self.x_state_
        # Predict
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q
        self.x_state_ = x.astype(int)

        self.whRCF.AddMeasure_noinput()

class Tracker(): # class for Kalman Filter-based tracker
    def __init__(self):
        # Initialize parametes for tracker (history)
        self.id = 0  # tracker's id 
        self.obj = {}

        self.hits = 0 # number of detection matches
        self.no_losses = 0 # number of unmatched tracks (track loss)
        
        # Initialize parameters for Kalman Filtering
        # The state is the (x, y) coordinates of the detection box
        # state: [left, left_dot, up, up_dot, right, right_dot, down, down_dot]
        # or[left, left_dot, up, up_dot, width, width_dot, height, height_dot]
        self.x_state=[] 
        self.dt = 1.   # time interval
        
        # Process matrix, assuming constant velocity model
        self.F = np.array([[1, self.dt, 0,  0,  0,  0,  0, 0],
                           [0, 1,  0,  0,  0,  0,  0, 0],
                           [0, 0,  1,  self.dt, 0,  0,  0, 0],
                           [0, 0,  0,  1,  0,  0,  0, 0],
                           [0, 0,  0,  0,  1,  self.dt, 0, 0],
                           [0, 0,  0,  0,  0,  1,  0, 0],
                           [0, 0,  0,  0,  0,  0,  1, self.dt],
                           [0, 0,  0,  0,  0,  0,  0,  1]])
        
        # Measurement matrix, assuming we can only measure the coordinates
        
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0], 
                           [0, 0, 0, 0, 0, 0, 1, 0]])
        
        
        # Initialize the state covariance
        self.L = 10.0
        self.P = np.diag(self.L*np.ones(8))
        
        
        # Initialize the process covariance
        self.Q_comp_mat = np.array([[self.dt**4/4., self.dt**3/2.],
                                    [self.dt**3/2., self.dt**2]])
        self.Q = block_diag(self.Q_comp_mat, self.Q_comp_mat, 
                            self.Q_comp_mat, self.Q_comp_mat)
        
        # Initialize the measurement covariance
        self.R_scaler = 1.0
        self.R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(self.R_diag_array)
        
        
    def update_R(self):   
        R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(R_diag_array)
        
        
        
        
    def kalman_filter(self, z): 
        '''
        Implement the Kalman Filter, including the predict and the update stages,
        with the measurement z
        '''
        x = self.x_state
        # Predict
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

        #Update
        S = dot(self.H, self.P).dot(self.H.T) + self.R
        K = dot(self.P, self.H.T).dot(inv(S)) # Kalman gain
        y = z - dot(self.H, x) # residual
        x += dot(K, y)
        self.P = self.P - dot(K, self.H).dot(self.P)
        self.x_state = x.astype(int) # convert to integer coordinates 
                                     #(pixel values)
        
    def predict_only(self):  
        '''
        Implment only the predict stage. This is used for unmatched detections and 
        unmatched tracks
        '''
        x = self.x_state
        # Predict
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q
        self.x_state = x.astype(int)