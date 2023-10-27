import numpy as np
import cv2
import os
from lidar_corner_detection import world_coordinate
from noise_removal import blur
from lidar_intrinsics import Lidar
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class Frame:
    def __init__(self, img_path, depth_path, br):
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        self.depth_img = cv2.imread(depth_path, cv2.COLOR_BGR2RGB)
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
    
    def find_world_coordinates(self, matches, query):

        # clear xyz from previous
        self.xyz = np.zeros((len(matches), 3))

        for i, match in enumerate(matches):
            if query:
                i_m = match.queryIdx
            else:
                i_m = match.trainIdx

            v = int(self.kp[i_m].pt[0])
            u = int(self.kp[i_m].pt[1])


            point = world_coordinate(u, v, self.depth_img[u,v])

            self.xyz[i] = point




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

    print(M)

    return M







if __name__=="__main__":

    folder = '2023_10_21_04_10_PM_lidar_camera'
    image_type = 'signal'

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


        prev_frame.find_world_coordinates(matches, True)

        this_frame.find_world_coordinates(matches, False)

        fit_transformation(this_frame.xyz, prev_frame.xyz)


        prev_frame = this_frame





