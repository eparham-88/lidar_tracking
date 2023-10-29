import numpy as np
import cv2
import os
from lidar_corner_detection import world_coordinate
from noise_removal import blur
from lidar_intrinsics import Lidar
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits import mplot3d


class Frame:
    def __init__(self, img_path, depth_path, br):
        img = cv2.imread(img_path, cv2.COLOR_RGB2BGR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print(np.max(img))
        self.depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        self.depth_img = cv2.cvtColor(self.depth_img, cv2.COLOR_RGB2BGR)
        self.depth_img = self.depth_img[:,:,0]
        self.img = blur(img)
        self.kp, self.des = br.detectAndCompute(self.img, None)
        self.filter_keypoints()
        self.xyz = None


    def filter_keypoints(self, display=False):

        # find laplacian to depth_img:
        laplacian = cv2.Laplacian(self.depth_img,cv2.CV_64F)
        laplacian[self.depth_img<5] = 255
        laplacian = cv2.GaussianBlur(laplacian, (9,9), 0)
        mask = (laplacian>1.0)
        mask[60:,:260] = True # temp fix

        if display:
            img = np.zeros_like(laplacian)
            img[mask] = 255

            cv2.imshow('laplacian', img)
            cv2.waitKey(0)

        kp_f = []
        des_f = []

        for i in range(len(self.kp)):

            # pull u, v value from kp descriptor
            u = int(self.kp[i].pt[0])
            v = int(self.kp[i].pt[1])

            if mask[v,u] == True:
                continue
            else:
                kp_f.append(self.kp[i])
                des_f.append(self.des[i])

        self.kp = tuple(kp_f)
        self.des = np.uint8(np.asarray(des_f))

        return
    
    def find_world_coordinates(self, matches, query, my_lidar, display=False):

        # clear xyz from previous
        self.xyz = np.zeros((len(matches), 4))
        
        kps_filtered = []

        for i, match in enumerate(matches):
            if query:
                i_m = match.queryIdx
            else:
                i_m = match.trainIdx

            v = int(self.kp[i_m].pt[0])
            u = int(self.kp[i_m].pt[1])

            point = my_lidar.getXYZCoords(u, v, self.depth_img[u,v])
            
            # print("u, v: {}, {}".format(u, v))
            # print("X, Y, Z: {}, {}, {}".format(point[0], point[1], point[2]))
            # point = world_coordinate(u, v, self.depth_img[u,v])

            self.xyz[i, :] = point
            
            kps_filtered.append(self.kp[i_m])
        
        self.xyz *= 0.001

        if display:
        
            plt.figure("points")
            ax = plt.axes(projection='3d')
            ax.scatter3D(self.xyz[:,0], self.xyz[:,1], self.xyz[:,2])
            ax.set_xlabel("x")
            ax.set_ylabel('y')
            ax.axis('scaled')
            
            img2_annotated = cv2.drawKeypoints(self.img, tuple(kps_filtered), None, color=(0,255,0))
            plt.figure("img")
            imgplot = plt.imshow(img2_annotated)

            plt.show()

        return
        
        
        
    def find_world_coordinates_whole_img(self, my_lidar):
        self.xyz = np.zeros((self.depth_img.shape[0] * self.depth_img.shape[1], 4))
        i = 0
        for u in range(self.depth_img.shape[0]):
            for v in range(self.depth_img.shape[1]):
                point = my_lidar.getXYZCoords(u, v, self.depth_img[u,v])
                self.xyz[i, :] = point
                i+= 1
        
        self.xyz = self.xyz @ my_lidar.lidar_to_sensor_transform * 0.001
        
        plt.figure("points")
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.xyz[:,0], self.xyz[:,1], self.xyz[:,2])
        ax.set_xlabel("x")
        ax.set_ylabel('y')
        ax.axis('scaled')
        plt.show()



def fit_transformation(XYZp, XYZ):
    xs = XYZ[:,0]; ys = XYZ[:,1]; zs = XYZ[:,2]
    xps = XYZp[:,0]; yps = XYZp[:,1]; zps = XYZp[:,2]

    A = np.zeros((3*xs.shape[0], 12))
    b = np.zeros((3*xs.shape[0], 1))

    A[0::3,0] = xs; A[0::3,1] = ys; A[0::3,2] = zs; A[0::3,3] = 1
    A[1::3,4] = xs; A[1::3,5] = ys; A[1::3,6] = zs; A[1::3,7] = 1
    A[2::3,8] = xs; A[2::3,9] = ys; A[2::3,10] = zs; A[2::3,11] = 1

    b[0::3,0] = xps; b[1::3,0] = yps; b[2::3,0] = zps

    M, _, _, _ = np.linalg.lstsq(A,b)

    M = np.vstack((M.reshape(3,4), np.array([0,0,0,1])))

    return M







if __name__=="__main__":

    folder = '2023_10_21_04_10_PM_lidar_camera'
    image_type = 'signal'
    my_lidar = Lidar()

    # grab folder
    wd = os.getcwd()
    lst = os.listdir(os.path.join(folder, image_type))
    lst.sort()

    # create items used throughout
    br = cv2.BRISK_create(thresh=15, octaves=3, patternScale=1.0)
    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    prev_frame = None



    for filename in lst:

        if prev_frame == None:
            prev_frame = Frame(os.path.join(wd,folder,image_type,filename),
                               os.path.join(wd,folder,'range',filename),
                               br)
            continue

        this_frame = Frame(os.path.join(wd,folder,image_type,filename),
                           os.path.join(wd,folder,'range',filename),
                           br)
        
        # Use Flann to identify matches between BRISK descriptions
        preliminary_matches = flann.knnMatch(prev_frame.des, this_frame.des, k=2)

        # Filter out bad matches below a theshold
        matches = set()
        for match in preliminary_matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.55 * n.distance:
                    matches.add(m)
                    
        # prev_frame.find_world_coordinates_whole_img(my_lidar)

        prev_frame.find_world_coordinates(matches, True, my_lidar)

        this_frame.find_world_coordinates(matches, False, my_lidar)

        M = fit_transformation(prev_frame.xyz, this_frame.xyz)

        print(M @ np.array([[0], [0], [0], [1]]))
        
        # transform first previous frame
        prev_xyz_transformed = M @ prev_frame.xyz.T
        prev_xyz_transformed = prev_xyz_transformed.T

        plt.figure("points")
        ax = plt.axes(projection='3d')
        ax.scatter3D(this_frame.xyz[:,0], this_frame.xyz[:,1], this_frame.xyz[:,2], color="b")
        ax.scatter3D(prev_xyz_transformed[:,0], prev_xyz_transformed[:,1], prev_xyz_transformed[:,2], color="r")
        ax.set_xlabel("x")
        ax.set_ylabel('y')
        ax.axis('scaled')
        plt.show()


        prev_frame = this_frame





