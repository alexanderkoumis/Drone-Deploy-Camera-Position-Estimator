#!/usr/bin/env python

"""
This script computes features for the pattern and camera images, finding matches between them. The
matched keypoints are then used by OpenCV's solvePnP function to compute the rotation and
translation vectors of the pattern image relative to the camera image. The rotation vector is
converted to a rotation matrix which is used to compute the Euler angles. Both the rotation and
translation results are inversed in order to find the position of the camera relative to the
pattern.
"""

import argparse
import os

import cv2
import numpy


class Solver(object):

    def __init__(self, pattern_image, cam_images, cam_matrix, dist_coef, orb=False):
        self.pattern_image = pattern_image
        self.cam_images = cam_images
        self.cam_matrix = cam_matrix
        self.dist_coef = dist_coef
        self.orb = orb
        self.detector = cv2.ORB() if orb else cv2.SIFT()
        # If ORB features are used, use BFMatcher. Instead, use FLANN matcher with SIFT features
        self.matcher = (cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) if orb
                        else cv2.FlannBasedMatcher({ 'algorithm': 0, 'trees': 5 }, { 'checks': 50 }))
        # Pattern is 8.8 cm
        self.scale = 88.0 / len(self.pattern_image)
        # This is populated when poses are detected
        self.poses = []

    def _match(self, pattern_descriptors, cam_descriptors):
        """Abstracted match interface"""
        if self.orb:
            return self.matcher.match(pattern_descriptors, cam_descriptors)
        else:
            # Use FLANN-specific parameters
            matches = self.matcher.knnMatch(pattern_descriptors, cam_descriptors, k=2)
            # Filter matches based on Lowe's ratio test
            matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
            return matches

    def solve(self):
        """Compute pose information for each image, filling self.poses with an object containing Euler angles
        of the camera relative to the mattern and the camera's translation
        """

        pattern_keypoints, pattern_descriptors = self.detector.detectAndCompute(self.pattern_image, None)

        for idx, cam_image in enumerate(self.cam_images):

            # Scale image by half to cut down computation time
            cam_image = cv2.resize(cam_image, (0, 0), fx=0.5, fy=0.5)

            print 'Calculating features and matches for image: {}'.format(idx)
            cam_keypoints, cam_descriptors = self.detector.detectAndCompute(cam_image, None)
            matches = self._match(pattern_descriptors, cam_descriptors)

            # Gather pattern and camera matched points into new arrays for input to solvePnP
            object_points = numpy.array([[cam_keypoints[m.trainIdx].pt[0], cam_keypoints[m.trainIdx].pt[1], 1]
                                         for m in matches])
            image_points = numpy.array([pattern_keypoints[m.queryIdx].pt for m in matches])
            _, r_vec, t_vec = cv2.solvePnP(object_points, image_points, self.cam_matrix, self.dist_coef)

            # Convert rotation vector into rotation matrix
            r_mat = numpy.zeros((3,3))
            cv2.Rodrigues(r_vec, r_mat)
            # solvePnP outputs the world coordinates with respect to the camera, we need the inverse
            r_mat = numpy.linalg.inv(r_mat)
            # Convert rotation matrix into Euler angles
            projection_mat = numpy.zeros((3,4))
            projection_mat[0:3, 0:3] = r_mat
            euler_angles = numpy.zeros(3)
            cv2.decomposeProjectionMatrix(projection_mat, _, _, _, _, _, _, euler_angles)

            # Invert translation vector (for same reason we compute inverse of rotation matrix)
            t_vec = numpy.multiply(t_vec, -1)
            # Scale translation vector based on known size of pattern
            t_vec = numpy.multiply(t_vec, self.scale)

            self.poses.append({'r': euler_angles, 't': t_vec})

    def print_results(self):
        """Pretty print poses"""
        for idx, pose in enumerate(self.poses):
            pitch, yaw, roll = pose['r']
            x, y, z = pose['t']
            print "Image {}".format(idx)
            print "  Rotation"
            print "    Pitch: {}".format(pitch)
            print "    Yaw: {}".format(yaw)
            print "    Roll: {}".format(roll)
            print "  Translation"
            print "    X: {}mm".format(x[0])
            print "    Y: {}mm".format(y[0])
            print "    Z: {}mm".format(z[0])

    def display_results_image(self):
        """Show X/Y map of camera positions relative to pattern image"""
        frame_w = 1024
        frame_h = 1024
        frame_center_x = frame_w / 2
        frame_center_y = frame_h / 2
        # We want the pattern to take up 10% of the resulting image
        pattern_original_w = len(self.pattern_image[0])
        pattern_original_h = len(self.pattern_image)
        pattern_resized_w = int(frame_w * 0.1)
        pattern_resized_h = int(pattern_original_h * (float(pattern_resized_w) / pattern_original_w))
        pattern_offset_x = frame_center_x - (pattern_resized_w / 2)
        pattern_offset_y = frame_center_y - (pattern_resized_h / 2)
        resized_image = cv2.resize(self.pattern_image, (pattern_resized_w, pattern_resized_h),
                                   interpolation=cv2.INTER_AREA)
        frame = numpy.zeros((frame_w, frame_h, 3), numpy.uint8)
        # Draw pattern over frame
        frame[pattern_offset_y:pattern_offset_y + pattern_resized_h,
              pattern_offset_x:pattern_offset_x + pattern_resized_w] = resized_image

        for idx, pose in enumerate(self.poses):
            cam_image = self.cam_images[idx]
            cam_image_original_w = len(cam_image[0])
            cam_image_original_h = len(cam_image)
            cam_image_resized_w = int(frame_w * 0.1)
            cam_image_resized_h = int(cam_image_original_h * (float(cam_image_resized_w) / cam_image_original_w))
            r, t = self.poses[idx]['r'], self.poses[idx]['t']
            t_x, t_y, t_z = t
            cam_image_offset_x = int(frame_center_x - (cam_image_resized_w / 2) + t_x)
            cam_image_offset_y = int(frame_center_y - (cam_image_resized_h / 2) + t_y)
            resized_image = cv2.resize(cam_image, (cam_image_resized_w, cam_image_resized_h),
                                       interpolation=cv2.INTER_AREA)
            try:
                # Draw cam image over frame
                frame[cam_image_offset_x:cam_image_offset_x + cam_image_resized_h,
                      cam_image_offset_y:cam_image_offset_y + cam_image_resized_w] = resized_image
            except:
                # Catch errors caused by frame image boundaries (frame_w, frame_h) not being big enough
                print 'Couldn\'t overlay image {}'.format(idx)

        cv2.imshow('Results', frame)
        cv2.waitKey(0)


def load_image(data_dir, image_path):
    """Load image into memory as numpy array from path"""
    image_path = os.path.join(data_dir, image_path)
    if not os.path.isfile(image_path):
        raise Exception('{} is not a file, check your data_dir argument'.format(data_dir))
    image = cv2.imread(image_path)
    return image


def load_images(data_dir, pattern_image_path, cam_image_paths):
    """Load images into memory as numpy arrays from paths""" 
    cam_images = [load_image(data_dir, cam_image) for cam_image in cam_image_paths]
    pattern_image = load_image(data_dir, pattern_image_path)
    return pattern_image, cam_images


def main(data_dir, orb):
    """Main app outer loop"""

    pattern_image_path = 'pattern.png'
    cam_image_paths = [
        'IMG_6719.JPG',
        'IMG_6720.JPG',
        'IMG_6721.JPG',
        'IMG_6722.JPG',
        'IMG_6723.JPG',
        'IMG_6724.JPG',
        'IMG_6725.JPG',
        'IMG_6726.JPG',
        'IMG_6727.JPG'
    ]
    pattern_image, cam_images = load_images(data_dir, pattern_image_path, cam_image_paths)
    # These values were found in spec sheets online
    cam_matrix = numpy.array([
        [1229, 0, 360],
        [0, 1153, 640],
        [0, 0, 1]
    ])
    # These are wrong but shouldn't introduce too much error
    dist_coef = numpy.zeros((5, 1))
    solver = Solver(pattern_image, cam_images, cam_matrix, dist_coef, orb)
    solver.solve()
    solver.print_results()
    solver.display_results_image()

def get_args():
    """Parse user args"""
    default_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    parser = argparse.ArgumentParser(description='DroneDeploy Camera Position Estimator')
    parser.add_argument('--data_dir', help='Path to data directory.', default=default_data_dir)
    parser.add_argument('--orb', help='Use ORB features instead of SIFT (faster, less accurate)',
                                 action='store_true', default=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args.data_dir, args.orb)
