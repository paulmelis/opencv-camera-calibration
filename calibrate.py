#!/usr/bin/env python

# Calibration pattern shown on screen.
# 9x6 inner points, 3.4 cm between points
#
# Note the quotes:
#
# $ ./calibrate.py 'frames/subsampled00*.png'
#
# Based on OpenCV example?
# PM 20200331 (corona)
# PM 20210812 (still corona)
import os
import numpy as np
import cv2 as cv

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext


def main(image_files, pattern_size, square_size, threads, debug_dir=None):
    """
    pattern_size: the number of *inner* points! So for a grid of 10x7 *squares* there's 9x6 inner points
    square_size: the real-world dimension of a chessboard square, in meters
    """

    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    # Read first image to get resolution
    h, w = cv.imread(image_files[0], cv.IMREAD_GRAYSCALE).shape[:2]  # TODO: use imquery call to retrieve results

    def process_image(fn):
        print('Processing %s... ' % fn)
        img = cv.imread(fn, 0)
        if img is None:
            print("Failed to load", fn)
            return None

        assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
        found, corners = cv.findChessboardCorners(img, pattern_size)
        if found:
            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
            cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        if debug_dir:
            # Write image with detected chessboard overlay
            vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            cv.drawChessboardCorners(vis, pattern_size, corners, found)            
            _, name, _ = splitfn(fn)
            outfile = os.path.join(debug_dir, name + '_chess.png')
            cv.imwrite(outfile, vis)

        if not found:
            print('chessboard not found')
            return None

        print('           %s... OK' % fn)
        return (corners, pattern_points)

    if threads <= 1:
        chessboards = [process_image(fn) for fn in image_files]
    else:
        print("Starting %d threads" % threads)
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool(threads)
        chessboards = pool.map(process_image, image_files)

    obj_points = []
    img_points = []        
    cb_index = 0
    cb_to_image_index = {}
    for img_index, chessboard in enumerate(chessboards):
        if chessboard is None:
            continue
        corners, pattern_points = chessboard
        img_points.append(corners)
        obj_points.append(pattern_points)
        cb_to_image_index[cb_index] = img_index
        cb_index += 1

    num_chessboards = cb_index
        
    print('Found chessboards in %d out of %d images' % (num_chessboards, len(image_files)))

    # Calculate camera distortion
    rms, camera_matrix, dist_coefs, rvecs, tvecs = \
        cv.calibrateCamera(obj_points, img_points, (w, h), None, None) #, None, None, None)

    #print(per_view_errors)
    print("\nRMS:", rms)
    
    # Compute reprojection error
    # After https://docs.opencv.org/4.5.2/dc/dbb/tutorial_py_calibration.html
    mean_error = 0
    for i in range(num_chessboards):
        img_points2, _ = cv.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coefs)        
        error = cv.norm(img_points[i], img_points2, cv.NORM_L2)/len(img_points2)
        mean_error += error
    print("Total reprojection error: ", mean_error/num_chessboards)
    
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients: ", dist_coefs.ravel())
    
    for cb_index in range(num_chessboards):
        img_index = cb_to_image_index[cb_index]
        r = rvecs[cb_index]
        t = tvecs[cb_index]
        print('[%s] rotation (%.6f, %.6f, %.6f), translation (%.6f, %.6f, %.6f)' % \
            (image_files[img_index], r[0][0], r[1][0], r[2][0], t[0][0], t[1][0], t[2][0]))

    # Undistort the image with the calibration
    if debug_dir:
        print('')
        print('Writing undistorted images to %s:' % debug_dir)
        
        for fn in image_files:
            _, name, _ = splitfn(fn)
            img_found = os.path.join(debug_dir, name + '_chess.png')
            outfile = os.path.join(debug_dir, name + '_undistorted.png')

            img = cv.imread(img_found)
            if img is None:
                continue

            h, w = img.shape[:2]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

            dst = cv.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

            # crop and save the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]

            print(outfile)
            cv.imwrite(outfile, dst)

    cv.destroyAllWindows()

if __name__ == '__main__':
    
    import sys, getopt
    from glob import glob
    
    corners = (9, 6)
    debug_dir = None
    square_size = 0.034
    threads = 4
    
    # XXX use defaults
    def usage():
        print('''
OpenCV Camera calibration for distorted images with chessboard samples.
Reads distorted images, calculates the calibration and writes undistorted images.

usage:
    calibrate.py [options] <images>...

default values:
    -c <w>x<h>      Number of *inner* corners of the chessboard pattern (default: 9x6)
    -s <size>       Square size in m (default: 0.0225)
    -t <threads>    Number of threads to use (default: 4)
    -d <dir>        Write debug images to dir
''')

    try:
        options, args = getopt.getopt(sys.argv[1:], 'c:d:s:t:')
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
        
    for o, v in options:
        if o == '-c':
            corners = tuple(map(int, v.split('x')))
        if o == '-d':
            debug_dir = v
        elif o == '-s':
            square_size = float(v)
        elif o == '-t':
            threads = int(v)
    
    #print(options)
    #print(args)

    if debug_dir and not os.path.isdir(debug_dir):
        os.mkdir(debug_dir)

    if len(args) == 0:
        print('No images provided!')
        usage()
        sys.exit(-1)
        
    image_files = args
           
    main(image_files, corners, square_size, threads, debug_dir)
