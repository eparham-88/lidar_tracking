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


# From https://data.ouster.io/downloads/software-user-manual/firmware-user-manual-v3.0.1.pdf
# Page 22
def world_coordinate(u, v, depth):
    h_res = 1024
    v_res = 128
    h_range = 2*np.pi
    v_range = np.pi / 2

    theta = (1.5*h_range) - (u / h_res)*h_range
    psi   = ((np.pi - v_range)/2) + (v / v_res)*v_range


    return [depth*np.sin(psi)*np.cos(theta),
            depth*np.sin(psi)*np.sin(theta),
            depth*np.cos(psi)]



def describe_keypoints(filename, kp, my_lidar, annotate=False):

    # Grab depth image
    wd = os.getcwd()
    depth_img = cv2.imread(os.path.join(wd,'2023_10_21_04_10_PM_lidar_camera/range',filename), cv2.COLOR_BGR2RGB)
    
    depth_img_copy = magnify(depth_img.copy())
    # print('max depth: ' + str(np.max(depth_img)) + ', min depth: ' + str(np.min(depth_img)))

    des = []
    kp_filtered = [] # will store filtered keypoints
    x_coords = []
    y_coords = []
    z_coords = []

    for i in range(len(kp)):

        # pull u, v value from kp descriptor
        u = int(kp[i].pt[0])
        v = int(kp[i].pt[1])

        # print(depth_img.shape)
        depth = depth_img[u, -v]

        # print('depth(' + str(u) + ', ' + str(v) + ') = ' + str(depth))

        # if depth is, this is a poor keypoint so don't continue and don't append to filtered list
        if depth < 13:
            continue 

        # Pull depth with opposite indices since we're flipped

        # x, y, z = world_coordinate(-v, u, depth)
        x, y, z = my_lidar.getXYZCoords(v, u, depth)

        # if annotate:
            # cv2.putText(depth_img_copy,
            #             str(depth),
            #             (depth_img_copy.shape[1]-3*v, 3*u),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             1,
            #             (255,0,0),
            #             1,
            #             2)
        
        des.append([x, y, z])
        
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)

        # append key point to new list
        kp_filtered.append(kp[i])
        
    # if annotate:
        # cv2.imshow('depth', depth_img_copy)
        # cv2.waitKey(0)

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
    kp1_filtered =0
    des1 = 0

    fast = cv2.FastFeatureDetector_create()
    fast.setType(2)
    # fast.setThreshold(20)

    br = cv2.BRISK_create()

    for filename in lst:
        #TODO: I think we should throw out detections that are in the first 20% of columns or last 80% of columns, that should make it so we still get a full pano image but arent getting false detections

        if img1.size == 0:
            img1 = cv2.imread(os.path.join(wd,folder,filename), cv2.COLOR_BGR2RGB)
            img1 = blur(img1)
            img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
            kp1 = fast.detect(img1, None)
            # des1 = describe_keypoints(filename, kp1)

            kp1_filtered, des1 = describe_keypoints(filename, kp1, my_lidar, True)
            
            continue
        
        img2 = cv2.imread(os.path.join(wd,folder,filename), cv2.COLOR_BGR2RGB)
        img2 = blur(img2)
        img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        kp2 = fast.detect(img2, None)

        # des2 = describe_keypoints(filename, kp2)

        kp2_filtered, des2 = describe_keypoints(filename, kp2, my_lidar)


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
            if m.distance < 0.20 * n.distance:
                good_matches.append(m)

        img3 = cv2.drawMatches(img1,kp1_filtered,img2,kp2_filtered,good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        img2_annotated = cv2.drawKeypoints(img2, kp2_filtered, None, color=(0,255,0))
        img2_annotated = magnify(cv2.rotate(img2_annotated, cv2.ROTATE_90_CLOCKWISE))
        cv2.imshow('detections', img2_annotated)

        # scaled = magnify(cv2.rotate(img3, cv2.ROTATE_90_CLOCKWISE))
        scaled = cv2.rotate(img3, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow('img', scaled)


        cv2.waitKey(0)

        img1 = img2
        kp1_filtered = kp2_filtered
        des1 = des2



def maskImageByDepth(img, depth, threshold):
    depth_blurred = cv2.GaussianBlur(depth, (15,15), 0)
    depth_mask = depth_blurred < threshold


    img_masked = img.copy()
    img_masked[depth_mask] = 255
    
    # cv2.imshow('image masked', img_masked)
    # cv2.waitKey(0)

    return img_masked


def filter_keypoints(kp, des, depth_img, display=True):

    kp_f = []
    des_f = []

    # cv2.imshow('depth', cv2.rotate(depth_img, cv2.ROTATE_90_CLOCKWISE))

    # find laplacian to depth_img:
    laplacian = cv2.Laplacian(depth_img,cv2.CV_64F)
    laplacian[depth_img<5] = 255
    laplacian = cv2.GaussianBlur(laplacian, (9,9), 0)
    mask = (laplacian>1.0)
    mask[-260:,60:] = True # temp fix

    if display:
        img = np.zeros_like(laplacian)
        img[mask] = 255

        # cv2.imshow('laplacian', cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
        # cv2.waitKey(0)

    for i in range(len(kp)):

        # pull u, v value from kp descriptor
        u = int(kp[i].pt[0])
        v = int(kp[i].pt[1])

        if mask[v,u] == True:
            continue
        else:
            kp_f.append(kp[i])
            des_f.append(des[i])


    print(str(len(kp)) + ' -> ' + str(len(kp_f)))
    return [tuple(kp_f), np.uint8(np.asarray(des_f))]




def BRISK_detections(folder):
    
    wd = os.getcwd()

    lst = os.listdir(folder)
    lst.sort()

    img1 = np.array([])
    kp1_filtered =0
    des1_filtered = 0

    br = cv2.BRISK_create(thresh=15, octaves=3, patternScale=1.0)

    for filename in lst:

        if img1.size == 0:
            img1 = cv2.imread(os.path.join(wd,folder,filename), cv2.COLOR_BGR2RGB)
            img1 = blur(img1)
            # img1 = maskImageByDepth(img1, cv2.imread(os.path.join(wd,'2023_10_21_04_10_PM_lidar_camera/range',filename), cv2.COLOR_BGR2RGB), 5)
            img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
            kp1, des1 = br.detectAndCompute(img1, None)
            kp1_filtered, des1_filtered = filter_keypoints(kp1, des1, cv2.rotate(cv2.imread(os.path.join(wd,'2023_10_21_04_10_PM_lidar_camera/range',filename), cv2.COLOR_BGR2RGB), cv2.ROTATE_90_COUNTERCLOCKWISE))
            
            continue
        
        img2 = cv2.imread(os.path.join(wd,folder,filename), cv2.COLOR_BGR2RGB)
        img2 = blur(img2)
        # img2 = maskImageByDepth(img2, cv2.imread(os.path.join(wd,'2023_10_21_04_10_PM_lidar_camera/range',filename), cv2.COLOR_BGR2RGB), 5)
        img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        kp2, des2 = br.detectAndCompute(img2, None)
        kp2_filterd, des2_filtered = filter_keypoints(kp2, des2, cv2.rotate(cv2.imread(os.path.join(wd,'2023_10_21_04_10_PM_lidar_camera/range',filename), cv2.COLOR_BGR2RGB), cv2.ROTATE_90_COUNTERCLOCKWISE))

        index_params = dict(algorithm=6,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=2)
        
        search_params = {}
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1_filtered, des2_filtered, k=2)
        
        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.55 * n.distance:
                    good_matches.append(m)

        img3 = cv2.drawMatches(img1,kp1_filtered,img2,kp2_filterd,good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        img2_annotated = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0))
        img2_annotated = magnify(cv2.rotate(img2_annotated, cv2.ROTATE_90_CLOCKWISE))
        cv2.imshow('detections', img2_annotated)

        # scaled = magnify(cv2.rotate(img3, cv2.ROTATE_90_CLOCKWISE))
        scaled = cv2.rotate(img3, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow('img', scaled)


        cv2.waitKey(0)

        img1 = img2
        kp1_filtered = kp2_filterd
        des1_filtered = des2_filtered




if __name__=="__main__":

    my_lidar = Lidar()
    ORB_detections('2023_10_21_04_10_PM_lidar_camera/signal', my_lidar)
    # BRISK_detections('2023_10_21_04_10_PM_lidar_camera/signal')