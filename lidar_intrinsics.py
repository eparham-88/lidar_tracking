# lidar intrinsics
import numpy as np
import matplotlib.pyplot as plt

class Lidar(object):
    def __init__(self):
        """
        beam_altitude_angles    -> len = 128, straight from page 86 of LiDAR manual (not sure if these are accurate, but a good starting point)
        beam_azimuth_angles     -> len = 128, straight from page 86 of LiDAR manual (not sure if these are accurate, but a good starting point)4
        beam_to_lidar_transform -> from page 87, converted to matrix
        lidar_to_beam_origin    -> just the top right element of the transform, but also provided on page 87
        scan_width              -> based on lidar config
        scan_height             -> based on lidar config
        n                       -> calculated form page 22 of manual
        theta_encoder           -> calculated from page 22 of manual, note that it is broadcast to calculate all encoder angles for the entire width so we can just index it later
        theta_azimuth           -> calcualted from page 22 of manual, note that it is broadcast to calculate all azimuth angles for the entire width so we can just index it later
        phi                     -> calcualted from page 22 of manual, note that it is broadcast to calculate all azimuth angles for the entire width so we can just index it later
        
        """
        self.beam_altitude_angles = np.array([44.47, 43.93, 43.22, 42.34, 41.53, 40.95, 40.23, 39.38, 38.58, 37.98, 37.27, 
                                              36.44, 35.65, 35.05, 34.33, 33.52, 32.74, 32.12, 31.41, 30.63, 29.85, 29.23, 
                                              28.52, 27.75, 26.97, 26.35, 25.65, 24.88, 24.12, 23.49, 22.79, 22.04, 21.29, 
                                              20.65, 19.96, 19.21, 18.48, 17.83, 17.12, 16.41, 15.66, 15.02, 14.32, 13.6, 
                                              12.88, 12.22, 11.53, 10.82, 10.1, 9.44, 8.74, 8.04, 7.33, 6.66, 5.97, 5.27, 
                                              4.56, 3.89, 3.2, 2.5, 1.8, 1.12, 0.43, -0.26, -0.96, -1.65, -2.33, -3.02, 
                                              -3.73, -4.42, -5.1, -5.79, -6.48, -7.2, -7.89, -8.56, -9.26, -9.98, -10.67, 
                                              -11.35, -12.04, -12.77, -13.45, -14.12, -14.83, -15.56, -16.26, -16.93, -17.63, 
                                              -18.37, -19.07, -19.73, -20.44, -21.19, -21.89, -22.55, -23.25, -24.02, -24.74, 
                                              -25.39, -26.09, -26.87, -27.59, -28.24, -28.95, -29.74, -30.46, -31.1, -31.81, 
                                              -32.62, -33.35, -33.99, -34.71, -35.54, -36.27, -36.91, -37.63, -38.47, -39.21, 
                                              -39.84, -40.57, -41.44, -42.2, -42.81, -43.55, -44.45, -45.21, -45.82])
        
        self.beam_azimuth_angles = np.array([11.01, 3.81, -3.25, -10.19, 10.57, 3.63, -3.17, -9.88, 10.18, 3.48, -3.1, -9.6, 
                                             9.84, 3.36, -3.04, -9.37, 9.56, 3.23, -2.99, -9.16, 9.3, 3.14, -2.95, -8.98, 9.08, 
                                             3.05, -2.91, -8.84, 8.9, 2.98, -2.88, -8.71, 8.74, 2.92, -2.85, -8.59, 8.6, 2.87, 
                                             -2.83, -8.51, 8.48, 2.82, -2.81, -8.43, 8.39, 2.78, -2.79, -8.37, 8.31, 2.75, -2.79, 
                                             -8.33, 8.25, 2.72, -2.78, -8.3, 8.22, 2.71, -2.79, -8.29, 8.19, 2.69, -2.79, -8.29, 
                                             8.18, 2.7, -2.8, -8.29, 8.2, 2.7, -2.81, -8.32, 8.22, 2.71, -2.82, -8.36, 8.27, 2.72, 
                                             -2.83, -8.41, 8.32, 2.74, -2.85, -8.48, 8.41, 2.78, -2.87, -8.55, 8.5, 2.81, -2.89, 
                                             -8.65, 8.63, 2.86, -2.92, -8.77, 8.76, 2.92, -2.97, -8.91, 8.93, 2.99, -3, -9.06, 
                                             9.12, 3.06, -3.06, -9.24, 9.34, 3.15, -3.12, -9.46, 9.6, 3.26, -3.18, -9.71, 9.91, 
                                             3.38, -3.25, -10, 10.26, 3.53, -3.34, -10.35, 10.68, 3.7, -3.43, -10.74])
        
        self.beam_to_lidar_transform = np.array([[1, 0, 0, 27.116], 
                                                 [0, 1, 0, 0],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]])
        
        self.lidar_to_sensor_transform = np.array([[-1, 0, 0, 0],
                                                   [0, -1, 0, 0],
                                                   [0, 0, 1, 38.195],
                                                   [0, 0, 0, 1]])
        
        self.lidar_to_beam_origin_mm = 27.116
        
        self.scan_width = 1024
        self.scan_height = 128
        
        self.n = np.sqrt(np.square(self.beam_to_lidar_transform[0][3]) + np.square(self.beam_to_lidar_transform[2][3]))
        
        # self.theta_encoder = 2 * np.pi * (np.ones(self.scan_width) - np.arange(0,self.scan_width) / self.scan_width)
        
        # self.theta_azimuth = -2 * np.pi * self.beam_azimuth_angles / 360
        
        # self.phi = 2 * np.pi * self.beam_altitude_angles / 360
        
        # plt.plot(self.phi)
        # plt.show()
        
        # a = 0
        
    def getXYZCoords(self, u, v, r):
        """ 
        u is height (rows)
        v is width (cols)
        """
        # print(r)
        
        theta_encoder = 2.0 * np.pi * (1.0 - v / self.scan_width)
        theta_azimuth = 0.0 * (-2.0 * np.pi * (self.beam_azimuth_angles[u] / 360.0))
        phi = 2.0 * np.pi * (self.beam_altitude_angles[u] / 360.0)
        
        x = (r - self.n) * np.cos(theta_encoder + theta_azimuth) * np.cos(phi) + self.beam_to_lidar_transform[0,3] * np.cos(theta_encoder)
        y = (r - self.n) * np.sin(theta_encoder + theta_azimuth) * np.cos(phi) + self.beam_to_lidar_transform[0,3] + np.sin(theta_encoder)
        z = (r - self.n) * np.sin(phi) + self.beam_to_lidar_transform[2,3]
        
        # x = (r - self.n)*np.cos(self.theta_encoder[measurement_id] + self.theta_azimuth[i])*np.cos(self.phi[i]) + (self.beam_to_lidar_transform[0][3])*np.cos(self.theta_encoder[measurement_id])
        # y = (r - self.n)*np.sin(self.theta_encoder[measurement_id] + self.theta_azimuth[i])*np.cos(self.phi[i]) + (self.beam_to_lidar_transform[0][3])*np.sin(self.theta_encoder[measurement_id])
        # z = (r - self.n)*np.sin(self.phi[i]) + (self.beam_to_lidar_transform[2,3])


        # Correct for lidar to sensor
        homogeneous = self.lidar_to_sensor_transform @ np.array([[x], [y], [z], [1]])
        homogeneous /= homogeneous[3,0]
        
        return homogeneous.T
    
    def setScanWidth(self, width):
        """ 
        in the event we change the width of the image, we need the encoder counts to change as well
        """
        self.scan_width = width
        self.theta_encoder = 2 * np.pi * (np.ones(self.scan_width) - np.arange(0,self.scan_width) / self.scan_width)