import numpy as np
import cv2
import os
from common import get_AKAZE, get_match_points, find_matches
from noise_removal import blur, remove_sides
from lidar_intrinsics import Lidar
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def magnify(img):
    return cv2.resize(img, (img.shape[1]*3, img.shape[0]*3), interpolation=cv2.INTER_LINEAR)

def process_folder(folder):
    
    wd = os.getcwd()

    lst = os.listdir(folder)
    lst.sort()

    for filename in lst:

        img = np.array(cv2.imread(os.path.join(wd,folder,filename)))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray = cv2.blur(gray, (5, 5))
        cv2.imshow('blurred', magnify(gray))

        gray32 = np.float32(gray)

        dst = cv2.cornerHarris(gray32, 2, 7, 0.04 )
        # dst = cv2.cornerHarris(gray32, 2, 3, 0.04 )
        dst = cv2.dilate(dst,None)

        # cv2.imshow('dst', dst)

        img[dst>0.01*dst.max()]=[0,0,255]

        # Scale image for visualization purposes
        scaled = magnify(img)
        cv2.imshow('img', scaled)


        cv2.waitKey(0)


def FAST_detections(folder):
    
    wd = os.getcwd()

    lst = os.listdir(folder)
    lst.sort()

    img2 = np.array([])

    fast = cv2.FastFeatureDetector_create()
    fast.setType(0)
    # fast.setThreshold(40)

    for filename in lst:

        if img2.size == 0:
            img2 = cv2.imread(os.path.join(wd,folder,filename), cv2.COLOR_BGR2RGB)
            continue
        
        img1 = cv2.imread(os.path.join(wd,folder,filename), cv2.COLOR_BGR2RGB)
        kp = fast.detect(img1, None)
        img1_annotated = cv2.drawKeypoints(img1, kp, None, color=(0,255,0))


        scaled = magnify(img1_annotated)
        cv2.imshow('img', scaled)


        cv2.waitKey(0)



        img2 = img1


# From https://math.libretexts.org/Courses/Monroe_Community_College/MTH_212_Calculus_III/Chapter_11%3A_Vectors_and_the_Geometry_of_Space/11.7%3A_Cylindrical_and_Spherical_Coordinates
def world_coordinate(u, v, depth):
    # r = depth
    # n = None
    # width = 1024
    # height = 128
    # theta_enc = None
    # theta_azi = None
    # beam_to_lidar = None
    
    # theta_azi = (u - (width/2)) * (180 / (width/2))
    # theta_alt = (v - (height/2)) * (45 / (height/2))
    
    # phi = 2*np.pi*theta_alt/360
    
    # x = (r - n)*np.cos(theta_enc + theta_azi)*np.cos(phi) + (beam_to_lidar[0,3])*np.cos(theta_enc)
    # y = (r - n)*np.sin(theta_enc + theta_azi)*np.cos(phi) + (beam_to_lidar[0,3])*np.sin(theta_enc)
    # z = (r - n)*np.sin(phi) + (beam_to_lidar[2,3])
    
    h_res = 1024
    v_res = 16
    h_range = 2*np.pi
    v_range = np.pi / 4

    theta = (1.5*h_range) - (u / h_res)*h_range
    psi   = ((np.pi - v_range)/2) + (v / v_res)*v_range


    return [depth*np.sin(psi)*np.cos(theta),
            depth*np.sin(psi)*np.sin(theta),
            depth*np.cos(psi)]


def describe_keypoints(filename, kp, my_lidar):

    # Grab depth image
    wd = os.getcwd()
    depth_img = cv2.imread(os.path.join(wd,'2023_10_21_04_10_PM_lidar_camera/range',filename), cv2.COLOR_BGR2RGB)

    des = []
    kp_filtered = [] # will store filtered keypoints
    x_coords = []
    y_coords = []
    z_coords = []

    for i in range(len(kp)):

        # pull u, v value from kp descriptor
        u = int(kp[i].pt[0])
        v = int(kp[i].pt[1])

        depth = depth_img[u, v]

        # if depth is, this is a poor keypoint so don't continue and don't append to filtered list
        if depth < 10:
            continue 

        # Pull depth with opposite indices since we're flipped
        # x, y, z = world_coordinate(v, u, depth)
        x, y, z = my_lidar.getXYZCoords(v, u, depth)

        # print('(' + str(u) + ', ' + str(v) + ') -> (' + 
        #       str(x) + ', ' + str(y) + ', ' + str(z) + ')')

        # append x, y, z as the description for least norm brute force
        des.append([x, y, z, u, v])
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)
        
        # append key point to new list
        kp_filtered.append(kp[i])
        
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_coords, y_coords, z_coords)
    plt.show()


    # return np.uint8(des)
    return [tuple(kp_filtered), np.uint8(np.asarray(des))]


def ORB_detections(folder, my_lidar):
    
    wd = os.getcwd()

    lst = os.listdir(folder)
    lst.sort()

    img1 = np.array([])
    kp1 =0
    des1 = 0

    fast = cv2.FastFeatureDetector_create()
    fast.setType(2)
    # fast.setThreshold(40)

    br = cv2.BRISK_create()

    for filename in lst:
        #TODO: I think we should throw out detections that are in the first 20% of columns or last 80% of columns, that should make it so we still get a full pano image but arent getting false detections

        if img1.size == 0:
            img1 = cv2.imread(os.path.join(wd,folder,filename), cv2.COLOR_BGR2RGB)
            img1 = blur(img1)
            img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
            kp1 = fast.detect(img1, None)
            # des1 = describe_keypoints(filename, kp1)
            kp1, des1 = describe_keypoints(filename, kp1, my_lidar)
            continue
        
        img2 = cv2.imread(os.path.join(wd,folder,filename), cv2.COLOR_BGR2RGB)
        img2 = blur(img2)
        img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        kp2 = fast.detect(img2, None)

        # des2 = describe_keypoints(filename, kp2)
        kp2, des2 = describe_keypoints(filename, kp2, my_lidar)


        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # matches = bf.match(des1, des2)

        index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
        
        search_params = {}
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2) # TODO (CAM): how does this work
        
        good_matches = []
        for m, n in matches:
            if m.distance < .45 * n.distance:
                good_matches.append(m)

        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        img2_annotated = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0))
        img2_annotated = magnify(cv2.rotate(img2_annotated, cv2.ROTATE_90_CLOCKWISE))
        cv2.imshow('detections', img2_annotated)

        scaled = magnify(cv2.rotate(img3, cv2.ROTATE_90_CLOCKWISE))
        cv2.imshow('img', scaled)


        cv2.waitKey(0)

        img1 = img2
        kp1 = kp2
        des1 = des2




def AKAZE_detections(folder):
    
    wd = os.getcwd()

    lst = os.listdir(folder)
    lst.sort()

    img2 = np.array([])

    for filename in lst:

        if img2.size == 0:
            img2 = cv2.imread(os.path.join(wd,folder,filename), cv2.COLOR_BGR2RGB)
            continue
        
        img1 = cv2.imread(os.path.join(wd,folder,filename), cv2.COLOR_BGR2RGB)

        kp1, desc1 = get_AKAZE(img1)
        kp2, desc2 = get_AKAZE(img2)
        matches = find_matches(desc1,desc2,0.7)
        XY = get_match_points(kp1,kp2,matches)
        
        img2 = img1



if __name__=="__main__":

    # process_folder(sys.argv[1])
    # process_folder('APT_lidar_camera/signal')
    # process_folder('APT_lidar_camera/reflec')
    # AKAZE_detections('APT_lidar_camera/reflec')
    # FAST_detections('APT_lidar_camera/reflec')
    # ORB_detections('GGB_Hallway_lidar_camera/reflec')
    my_lidar = Lidar()
    ORB_detections('2023_10_21_04_10_PM_lidar_camera/signal', my_lidar)