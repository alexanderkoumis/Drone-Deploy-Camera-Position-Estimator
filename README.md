# DroneDeploy Coding Test Option 1

`drone_deploy.py` computes camera positions relative to a pattern from a set of images taken on an iPhone 6.

Usage:
```
usage: drone_deploy.py [-h] [--data_dir DATA_DIR] [--orb]

DroneDeploy Camera Position Estimator

optional arguments:
  -h, --help           show this help message and exit
  --data_dir DATA_DIR  Path to data directory.
  --orb                Use ORB features instead of SIFT (faster, less
                       accurate)
```

Both the `--data_dir` and `--orb` options are optional. By default the script will use the files in the `/data` directory, use `--data_dir` if you would like to use files in a different directory (the names of the files must be the same as the ones in the provided dataset). Use `--orb` to compute ORB features and use a brute force descriptor matcher, instead of computing the default SIFT features and matching with a FLANN matcher.

Running the script will print the results (pitch, yaw, roll, x, y, z of camera relative to pattern) to the console, and show a map displaying the position of the camera images relative to the pattern. Each pixel is a millimeter in this map.

This script requires Python OpenCV to be installed.

## How it works

This script computes features for the pattern and camera images, finding matches between them. The matched keypoints are then used by OpenCV's `solvePnP` function to compute the rotation and translation vectors of the pattern image relative to the camera image. The rotation vector is converted to a rotation matrix which is used to compute the Euler angles. Both the rotation and translation results are inversed in order to find the position of the camera relative to the pattern.
