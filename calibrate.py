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
import json, os
import numpy as np
import cv2

def splitfn(fname):
    path, fname = os.path.split(fname)
    name, ext = os.path.splitext(fname)
    return path, name, ext

def main(image_files, pattern_size, square_size, threads, json_file=None, debug_dir=None):
    """    
    image_files: list of image file names
    pattern_size: the number of *inner* points! So for a grid of 10x7 *squares* there's 9x6 inner points
    square_size: the real-world dimension of a chessboard square, in meters
    threads: number of threads to use
    json_file: JSON file to write calibration data to 
    debug_dir: if set, the path to which debug images with the detected chessboards are written
    """
    
    # JSON data
    j = {}

    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size    
    
    j['chessboard_points'] = pattern_points.tolist()
    j['chessboard_inner_corners'] = pattern_size
    j['chessboard_spacing_mm'] = square_size
        
    # Read first image to get resolution
    # TODO: use imquery call to retrieve results
    img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('Failed to read %s to get resolution!' % image_files[0])
        return
        
    h, w = img.shape[:2]
    print('Image resolution %dx%d' % (w, h))    
    
    j['image_resolution'] = (w, h)
    
    # Process all images

    def process_image(fname):
        sys.stdout.write('.')
        sys.stdout.flush()
        
        img = cv2.imread(fname, 0)
        if img is None:
            return (fname, None, 'Failed to load')
            
        if w != img.shape[1] or h != img.shape[0]:
            return (fname, None, "Size %dx%d doesn't match" % (img.shape[1], img.shape[0]))
        
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        if debug_dir:
            # Write image with detected chessboard overlay
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)            
            _, name, _ = splitfn(fname)
            outfile = os.path.join(debug_dir, name + '_chess.png')
            cv2.imwrite(outfile, vis)

        if not found:
            return (fname, None, 'Chessboard not found')

        return (fname, corners, pattern_points)
    
    if threads <= 1:
        sys.stdout.write('Processing images ')
        results = [process_image(fname) for fname in image_files]
    else:
        sys.stdout.write('Processing images using %d threads ' % threads)
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool(threads)
        results = pool.map(process_image, image_files)
        
    sys.stdout.write(' done')
    sys.stdout.flush()
    
    # Prepare calibration input

    obj_points = []
    img_points = []        
    cb_index = 0
    cb_to_image_index = {}
    
    results.sort(key = lambda e: e[0])
    for img_index, result in enumerate(results):
        corners = result[1]
        if corners is None:
            print('[%s] FAILED: %s' % (result[0], result[2]))
            continue
        img_points.append(corners)
        obj_points.append(result[2])
        cb_to_image_index[cb_index] = img_index
        cb_index += 1

    num_chessboards = cb_index
        
    print('\n')
    print('Found chessboards in %d out of %d images' % (num_chessboards, len(image_files)))
    print()
    
    if num_chessboards == 0:
        print('No chessboards to use!')
        sys.exit(-1)

    # Calculate camera matrix, distortion, etc
    
    rms, camera_matrix, dist_coefs, rvecs, tvecs = \
        cv2.calibrateCamera(obj_points, img_points, (w, h), None, None) #, None, None, None)

    #print(per_view_errors)
    print("RMS:", rms)
    
    # Compute reprojection error
    # After https://docs.opencv2.org/4.5.2/dc/dbb/tutorial_py_calibration.html
    mean_error = 0
    for i in range(num_chessboards):
        img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coefs)        
        error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2)/len(img_points2)
        mean_error += error
    reprojection_error = mean_error/num_chessboards
    print("Total reprojection error: %.6f" % reprojection_error)
    print()
    print("Camera matrix:\n", camera_matrix)    
    print("Distortion coefficients:", dist_coefs.ravel())
    
    j['camera_matrix'] = camera_matrix.tolist()
    j['distortion_coefficients'] = dist_coefs.ravel().tolist()
    j['rms'] = rms
    j['reprojection_error'] = reprojection_error
    
    if sensor_size is not None:
        fovx, fovy, focal_length, principal_point, aspect_ratio = \
            cv2.calibrationMatrixValues(camera_matrix, (w,h), sensor_size[0], sensor_size[1])
        print('FOV: %.6f %.6f degrees' % (fovx, fovy))
        print('Focal length: %.6f mm' % focal_length)
        print('Principal point: %.6f %.6f mm' % principal_point)
        print('Aspect ratio: %.6f' % aspect_ratio)
        
        j['sensor_size_mm'] = sensor_size
        j['fov_degrees'] = (fovx, fovy)
        j['focal_length_mm'] = focal_length
        j['principal_point_mm'] = principal_point
        j['aspect_ratio'] = aspect_ratio
    
    print()
    chessboard_orientations = {}
    for cb_index in range(num_chessboards):
        img_index = cb_to_image_index[cb_index]
        r = rvecs[cb_index]
        t = tvecs[cb_index]
        print('[%s] rotation (%.6f, %.6f, %.6f), translation (%.6f, %.6f, %.6f)' % \
            (image_files[img_index], r[0][0], r[1][0], r[2][0], t[0][0], t[1][0], t[2][0]))
            
        rotation_matrix, _ = cv2.Rodrigues(r)
        chessboard_orientations[image_files[img_index]] = {
            #'rotation_vector': (r[0][0], r[1][0], r[2][0]),
            'rotation_matrix': rotation_matrix.tolist(),
            'translation': (t[0][0], t[1][0], t[2][0])
        }
        
        # OpenCV untransformed camera orientation is X to the right, Y down,
        # Z along the view direction (i.e. right-handed). This aligns x,y pixels
        # coordinates of the image plane with the X,Y coordinates in camera space.
        # The orientations describe the transform needed to bring a detected chessboard
        # from its object space into camera space.
        j['chessboard_orientations'] = chessboard_orientations
            
    # Write to JSON
            
    if json_file is not None:
        json.dump(j, open(json_file, 'wt'))

    # Undistort the image with the calibration
    if debug_dir is not None:
        print('')
        print('Writing undistorted images to %s:' % debug_dir)
        
        for fname in image_files:
            _, name, _ = splitfn(fname)
            img_found = os.path.join(debug_dir, name + '_chessboard.png')
            outfile1 = os.path.join(debug_dir, name + '_undistorted.png')
            outfile2 = os.path.join(debug_dir, name + '_undistorted_cropped.png')

            img = cv2.imread(img_found)
            if img is None:
                continue

            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

            dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
            
            # save uncropped
            cv2.imwrite(outfile1, dst)

            # crop and save the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]            
            cv2.imwrite(outfile2, dst)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    import sys, getopt
    from glob import glob
    
    corners = (9, 6)
    debug_dir = None
    json_file = None
    sensor_size = None
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
    -c <w>x<h>              Number of *inner* corners of the chessboard pattern (default: 9x6)
    -s <size>               Square size in m (default: 0.0225)
    -t <threads>            Number of threads to use (default: 4)
    -j <calibration.json>   Write calibration data to JSON file
    -S <w>x<h>              Physical sensor size in mm (optional)
    -d <dir>                Write debug images to dir
''')

    try:
        options, args = getopt.getopt(sys.argv[1:], 'c:d:j:S:s:t:')
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
        
    for o, v in options:
        if o == '-c':
            corners = tuple(map(int, v.split('x')))
        elif o == '-d':
            debug_dir = v
        elif o == '-j':
            json_file = v
        elif o == '-S':
            sensor_size = tuple(map(float, v.split('x')))
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
           
    main(image_files, corners, square_size, threads, json_file, debug_dir)
