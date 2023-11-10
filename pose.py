import numpy as np
import cv2
import os
from lidar_corner_detection import world_coordinate
from noise_removal import blur
from lidar_intrinsics import Lidar
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits import mplot3d
import scipy.spatial.transform as rt


class Frame:
    def __init__(self, img_path, depth_path, br, lidar):
        img = cv2.imread(img_path, cv2.COLOR_RGB2BGR)
        self.depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        self.img = blur(img)
        self.kp, self.des = br.detectAndCompute(self.img, None)
        self.filter_keypoints()
        self.find_world_coordinates(lidar)


    def filter_keypoints(self, display=False):

        # find laplacian of depth_img to determine large deltas in depth
        laplacian = cv2.Laplacian(self.depth_img,cv2.CV_64F)
        laplacian = cv2.GaussianBlur(laplacian, (9,9), 0)
        if display:
            cv2.imshow('laplacian', laplacian)

        # add areas that are within ~1m
        laplacian[self.depth_img<200] = 65536
        if display:
            cv2.imshow('laplacian + depth', laplacian)

        # dilate
        laplacian = cv2.dilate(laplacian, np.ones((5,5),np.uint8), iterations=4)
        if display:
            cv2.imshow('dilated', laplacian)


        # blur
        laplacian = cv2.GaussianBlur(laplacian, (19,19), 0)
        if display:
            cv2.imshow('blurred', laplacian)

        # Threshold
        self.mask = (laplacian>350.0)
        if display:
            laplacian = np.zeros_like(laplacian)
            laplacian[self.mask] = 65536
            cv2.imshow('masked',  laplacian)
            cv2.waitKey(0)

        kp_f = []
        des_f = []

        for i in range(len(self.kp)):

            # pull u, v value from kp descriptor
            u = int(self.kp[i].pt[0])
            v = int(self.kp[i].pt[1])

            if self.mask[v,u] == True:
                continue
            else:
                kp_f.append(self.kp[i])
                des_f.append(self.des[i])

        self.kp = tuple(kp_f)
        self.des = np.uint8(np.asarray(des_f))

        return
    
    def find_match_coordinates(self, matches, query, display=False):

        # clear xyz from previous
        self.match_xyz = np.zeros((4, len(matches)))
        kp_match = []
        des_match = []
        

        for i, match in enumerate(matches):
            if query:
                i_m = match.queryIdx
            else:
                i_m = match.trainIdx

            self.match_xyz[:, i] = self.xyz[:, i_m]
            
            kp_match.append(self.kp[i_m])
            des_match.append(self.des[i_m])

        self.match_kp = tuple(kp_match)
        self.match_des = np.uint8(np.asarray(des_match))

        if display:
        
            plt.figure("points")
            ax = plt.axes(projection='3d')
            ax.scatter3D(self.match_xyz[0,:], self.match_xyz[1,:], self.match_xyz[2,:])
            ax.set_xlabel("x")
            ax.set_ylabel('y')
            ax.axis('scaled')
            
            img2_annotated = cv2.drawKeypoints(self.img, self.match_kp, None, color=(0,255,0))
            plt.figure("img")
            imgplot = plt.imshow(img2_annotated)

            plt.show()

        return
    
    def find_world_coordinates(self, lidar):

        self.xyz = np.zeros((4, len(self.kp)))

        for i, keypoint in enumerate(self.kp):
            v = int(keypoint.pt[0])
            u = int(keypoint.pt[1])

            point = lidar.getXYZCoords(u, v, self.depth_img[u,v])

            self.xyz[:, i] = point

        self.xyz[:3, :] *= 0.001

        return


    def filter_inliers(self, mask):
        self.match_xyz = self.match_xyz[:, mask]
        self.match_des = self.match_des[mask]
        self.match_kp = tuple(np.array(self.match_kp)[mask])
        
        
    def find_world_coordinates_whole_img(self, my_lidar):
        self.xyz = np.zeros((self.depth_img.shape[0] * self.depth_img.shape[1], 4))
        i = 0
        for u in range(self.depth_img.shape[0]):
            for v in range(self.depth_img.shape[1]):
                point = my_lidar.getXYZCoords(u, v, self.depth_img[u,v])
                self.xyz[:, i] = point
                i+= 1
        
        self.xyz =  my_lidar.lidar_To_inlierssensor_transform @ self.xyz
        self.xyz[:3, :] *= 0.001
        
        plt.figure("points")
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.xyz[0,:], self.xyz[1,:], self.xyz[2,:])
        ax.set_xlabel("x")
        ax.set_ylabel('y')
        ax.axis('scaled')
        plt.show()


    # def add_previous_frame(self, previous_frame):
    #     self.xyz = np.hstack((self.xyz, previous_frame.xyz))[:, :1000]
    #     self.des = np.vstack((self.des, previous_frame.des))[:1000]
    #     self.kp = np.array(self.kp + previous_frame.kp)[:1000]
    #     self.kp = tuple(self.kp)
    








def fit_transformation(From, To, W=None):

    From_inliers = np.copy(From)
    To_inliers = np.copy(To)
    W_inliers = np.copy(W)

    M = None

    outlier_final_mask = From[3, :] > 0

    last_sum = 0
    for _ in range(20):

        # Find the transformation
        M = find_transform(From_inliers, To_inliers, W_inliers)
        

        # Find norm differences between points
        diff = np.linalg.norm(To - M @ From, axis=0)
        diff_ = np.linalg.norm(To_inliers - M @ From_inliers, axis=0)
        std = np.std(diff_)
        outlier_mask = diff < 0.5*std+np.average(diff)


        if False:
            plt.subplot(3, 1, 1)
            plt.plot(diff, label="magnitude")
            plt.legend()
            plt.subplot(3, 1, 2)
            plt.plot(outlier_mask, label="inlier")
            plt.legend()
            plt.subplot(3, 1, 3)
            if W.size != 0:
                plt.plot(W, label="weight")
            plt.legend()
            plt.show()

        if np.sum(outlier_mask) == last_sum or np.sum(outlier_mask) < 10:
            # no outliers or too few remaining points for another round -> exit
            break

        # update outliers mask with new bad values
        outlier_final_mask = (outlier_final_mask) & (outlier_mask)

        From_inliers = From[:, outlier_final_mask]
        To_inliers = To[:, outlier_final_mask]
        if W.size != 0:
            W_inliers = W[outlier_final_mask]
        last_sum = np.sum(outlier_mask)

    return M, outlier_final_mask



def find_transform(From, To, W=None):
    # https://igl.ethz.ch/projects/ARAP/svd_rot.pdf

    # Reconfigure data into 3xn
    P = From[:3, :]
    Q = To[:3, :]

    # Find centroids
    if W.size != 0:
        P_bar = np.sum(P*W, 1).reshape(-1,1) / np.sum(W)
        Q_bar = np.sum(Q*W, 1).reshape(-1,1) / np.sum(W)
    else:
        P_bar = np.mean(P, 1).reshape(-1,1)
        Q_bar = np.mean(Q, 1).reshape(-1,1)


    # Offset by centroids
    X = P - P_bar
    Y = Q - Q_bar

    # Calculate 3x3 covariance
    if W.size != 0:
        W_diag = np.diag(W)
    else:
        W_diag = np.eye(X.shape[1])
    cov = X @ W_diag @ Y.T

    # Use SVD to calculate the 3x3 Matrices U and V from coveriance
    U, _, V_T = np.linalg.svd(cov); V = V_T.T

    # Find rotation
    m = np.eye(U.shape[0]); m[-1,-1] = np.linalg.det(V @ U.T)
    R = V @ m @ U.T

    # Find translation
    T = Q_bar - R @ P_bar

    M = np.vstack((np.hstack((R, T)), np.array([0, 0, 0, 1]) ))

    return M




def flip_keypoints(f, shape):
    return [cv2.KeyPoint(x = shape[1]-k.pt[1], y = k.pt[0], 
            size = k.size, angle = k.angle, 
            response = k.response, octave = k.octave, 
            class_id = k.class_id) for k in f]


def drawMatches(frame_1, frame_2, matches):

    img_1_masked = cv2.cvtColor(frame_1.img,cv2.COLOR_GRAY2RGB)
    img_2_masked = cv2.cvtColor(frame_2.img,cv2.COLOR_GRAY2RGB)
    img_1_masked[frame_1.mask,1] = 0
    img_2_masked[frame_2.mask,1] = 0

    img_1 = cv2.rotate(img_1_masked, cv2.ROTATE_90_CLOCKWISE)
    img_2 = cv2.rotate(img_2_masked, cv2.ROTATE_90_CLOCKWISE)

    kp_1 = flip_keypoints(frame_1.kp, img_1.shape)
    kp_2 = flip_keypoints(frame_2.kp, img_2.shape)

    img3 = cv2.drawMatches(img_2, kp_2,
                           img_1, kp_1,
                           matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('matches', cv2.rotate(img3, cv2.ROTATE_90_COUNTERCLOCKWISE))
    cv2.waitKey(0)





if __name__=="__main__":

    # folder = '18ft_forward_50_ft_left'
    # folder = '50ft_hallway'
    # folder = '50ft_locker_hallway'
    folder = '120X70X120X71ft_square'
    # folder = '250_room_square_1'
    # folder = '250_room_square_2'
    # folder = '2023_10_21_04_10_PM_lidar_camera'
    image_type = 'signal'
    my_lidar = Lidar()

    # grab folder
    wd = os.getcwd()
    lst = os.listdir(os.path.join(folder, image_type))
    lst.sort()

    # create items used throughout
    br = cv2.BRISK_create(thresh=5, octaves=2, patternScale=1.0)
    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    prev_frame = None

    poses = np.zeros((len(lst), 3))
    transforms = np.zeros((len(lst), 12))
    current_transform = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

    for i, filename in enumerate(lst):

        if prev_frame == None:
            prev_frame = Frame(os.path.join(wd,folder,image_type,filename),
                               os.path.join(wd,folder,'range',filename),
                               br,
                               my_lidar)
            continue

        this_frame = Frame(os.path.join(wd,folder,image_type,filename),
                           os.path.join(wd,folder,'range',filename),
                           br,
                           my_lidar)
        
        # Use Flann to identify matches between BRISK descriptions
        preliminary_matches = flann.knnMatch(prev_frame.des, this_frame.des, k=2)

        # Filter out bad matches below a theshold
        matches = []
        weights = []
        for match in preliminary_matches:
            if len(match) == 0:
                continue
            m_d = match[0].distance
            if len(match) >= 2:
                n_d = match[1].distance
            else:
                n_d = 300

            if m_d < 0.7 * n_d:

                matches.append(match[0])
                if False:
                    # Weighted
                    w = max(200 - m_d, 0)
                    weights.append(w)
                
        weights = np.array(weights)

        prev_frame.find_match_coordinates(matches, True)

        this_frame.find_match_coordinates(matches, False)

        M, inliers_mask = fit_transformation(this_frame.match_xyz, prev_frame.match_xyz, weights)

        this_frame.filter_inliers(inliers_mask)
        prev_frame.filter_inliers(inliers_mask)

        # Store transformations and poses
        transforms[i, :] = M[:3,:].reshape(-1)
        current_transform = current_transform @ M
        pose = current_transform @ np.array([[0], [0], [0], [1]])
        pose /= pose[3,0]
        poses[i, :] = pose[:3,0].reshape(-1)
        

        if False:
            plt.figure("points")
            ax = plt.axes(projection='3d')
            this_frame_match_xyz = M @ this_frame.match_xyz
            ax.scatter3D(this_frame_match_xyz[0,:], this_frame_match_xyz[1,:], this_frame_match_xyz[2,:], color="b")
            ax.scatter3D(prev_frame.match_xyz[0,:], prev_frame.match_xyz[1,:], prev_frame.match_xyz[2,:], color="r")
            ax.set_xlabel("x")
            ax.set_ylabel('y')
            ax.axis('scaled')
            plt.show()
        elif False:
            drawMatches(this_frame, prev_frame, matches)

        print(str(i) + " of " + str(len(lst)))
        prev_frame = this_frame

    
    plt.figure("poses")
    ax = plt.axes(projection='3d')
    ax.scatter3D(poses[:,0], poses[:,1], poses[:,2], c=range(poses.shape[0]))
    ax.set_xlabel("x")
    ax.set_ylabel('y')
    ax.axis('scaled')
    plt.show()

    print()



