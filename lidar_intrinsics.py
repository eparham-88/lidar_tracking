# lidar intrinsics
import numpy as np

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
        self.beam_altitude_angles = np.array([ 20.38, 20.12, 19.79, 19.45, 19.14, 18.85, 18.55, 18.2, 17.86, 17.58, 17.27, 16.93,
                                    16.58, 16.29, 15.98, 15.61, 15.27, 14.97, 14.66, 14.3, 13.96, 13.65, 13.33, 12.97,
                                    12.62, 12.31, 11.98, 11.63, 11.27, 10.96, 10.63, 10.26, 9.91, 9.59, 9.26, 8.89,
                                    8.54, 8.21, 7.87, 7.52, 7.15, 6.82, 6.47, 6.11, 5.76, 5.42, 5.08, 4.73, 4.36, 4.03,
                                    3.66, 3.31, 2.96, 2.62, 2.27, 1.91, 1.55, 1.22, 0.85, 0.51, 0.16, -0.2, -0.55, -0.91,
                                    -1.26, -1.62, -1.96, -2.3, -2.66, -3.02, -3.36, -3.72, -4.07, -4.42, -4.77, -5.11,
                                    -5.46, -5.82, -6.16, -6.49, -6.85, -7.21, -7.55, -7.88, -8.23, -8.59, -8.93, -9.25,
                                    -9.6, -9.96, -10.31, -10.63, -10.96, -11.32, -11.67, -11.97, -12.31, -12.68, -13,
                                    -13.32, -13.64, -14, -14.33, -14.63, -14.96, -15.31, -15.64, -15.94, -16.26,
                                    -16.62, -16.93, -17.22, -17.54, -17.9, -18.22, -18.49, -18.8, -19.16, -19.47,
                                    -19.73, -20.04, -20.39, -20.7, -20.94, -21.25, -21.6, -21.9, -22.14])
        
        self.beam_azimuth_angles = np.array([4.24, 1.41, -1.42, -4.23, 4.23, 1.41, -1.41, -4.23, 4.23, 1.41, -1.41, -4.21, 4.23,
                                    1.42, -1.4, -4.23, 4.24, 1.41, -1.4, -4.23, 4.24, 1.42, -1.4, -4.22, 4.23, 1.41,
                                    -1.41, -4.22, 4.23, 1.42, -1.4, -4.22, 4.24, 1.41, -1.4, -4.23, 4.23, 1.41, -1.41,
                                    -4.22, 4.23, 1.41, -1.41, -4.23, 4.23, 1.4, -1.42, -4.23, 4.23, 1.41, -1.42, -4.23,
                                    4.23, 1.4, -1.42, -4.24, 4.22, 1.41, -1.43, -4.24, 4.22, 1.4, -1.42, -4.24, 4.22,
                                    1.4, -1.42, -4.23, 4.22, 1.4, -1.4, -4.24, 4.22, 1.4, -1.42, -4.24, 4.22, 1.41,
                                    -1.41, -4.22, 4.22, 1.39, -1.42, -4.23, 4.22, 1.41, -1.41, -4.22, 4.23, 1.41,
                                    -1.41, -4.23, 4.23, 1.41, -1.41, -4.22, 4.23, 1.41, -1.41, -4.22, 4.22, 1.41,
                                    -1.41, -4.22, 4.23, 1.41, -1.4, -4.23, 4.22, 1.41, -1.41, -4.23, 4.22, 1.4, -1.41,
                                    -4.23, 4.22, 1.4, -1.41, -4.24, 4.22, 1.4, -1.42, -4.24, 4.22, 1.4, -1.42, -4.23])
        
        self.beam_to_lidar_transform = np.array([[1, 0, 0, 15.805999755859375], 
                                                 [0, 1, 0, 0],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]])
        
        self.lidar_to_beam_origin_mm = 15.8059998
        
        self.scan_width = 1024
        self.scan_height = 128
        
        self.n = np.sqrt(np.square(self.beam_to_lidar_transform[0][3]) + np.square(self.beam_to_lidar_transform[2][3]))
        
        self.theta_encoder = 2 * np.pi * (np.ones(self.scan_width) - np.arange(0,self.scan_width) / self.scan_width)
        
        self.theta_azimuth = -2 * np.pi * self.beam_azimuth_angles / 360
        
        self.phi = 2 * np.pi * self.beam_altitude_angles / 360
        
    def getXYZCoords(self, u, v, r):
        """ 
        u is just the measurement id, because measurement id is from 1 to the width
        i is where im making an educated guess, and if its correct ill feel like a god damn genius. the azimuth angles wrap from 0 to 360, so we want the index that is the remainder of u and the total number of elements
        x,y,z are taken from page 22 of manual
        honestly, I'm not using v, which is a little concerning and might be why its incorrect.
        """
        
        measurement_id = u
        i = u % np.size(self.beam_azimuth_angles)
        
        x = (r - self.n)*np.cos(self.theta_encoder[measurement_id] + self.theta_azimuth[i])*np.cos(self.phi[i]) + (self.beam_to_lidar_transform[0][3])*np.cos(self.theta_encoder[measurement_id])
        y = (r - self.n)*np.sin(self.theta_encoder[measurement_id] + self.theta_azimuth[i])*np.cos(self.phi[i]) + (self.beam_to_lidar_transform[0][3])*np.sin(self.theta_encoder[measurement_id])
        z = (r - self.n)*np.sin(self.phi[i]) + (self.beam_to_lidar_transform[2,3])
        
        return [x,y,z]
    
    def setScanWidth(self, width):
        """ 
        in the event we change the width of the image, we need the encoder counts to change as well
        """
        self.scan_width = width
        self.theta_encoder = 2 * np.pi * (np.ones(self.scan_width) - np.arange(0,self.scan_width) / self.scan_width)