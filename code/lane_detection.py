import numpy as np
import cv2
import matplotlib.pyplot as plt

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):

    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient =='x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    if orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
 
    binary_output = sxbinary
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Calculate the magnitude
    abs_sobel = np.sqrt(np.square(sobelx)+np.square(sobely))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(dir_sobel)
    binary_output[(dir_sobel >= thresh[0]) & (dir_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    
    return binary_output

def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # H = hls[:,:,0]
    # L = hls[:,:,1]
    S = hls[:,:,2]
    # 2) Apply a threshold to the S channel
    binary_output =np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    # 3) Return a binary image of threshold in HLS color spaces
    return binary_output

def rgb_select(img, thresh=(220, 255)):
   
    # 1) Get R channel of image
    R = img[:,:,0]
    # 2) Apply a threshold to the R channel
    binary_output =np.zeros_like(R)
    binary_output[(R > thresh[0]) & (R <= thresh[1])] = 1
    # 3) Return a binary image of threshold in RGB color spaces
    return binary_output

def region_of_interest(img):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    imshape = img.shape
    vertices = np.array([[(0+50,imshape[0]),(int(imshape[1]*0.45), int(imshape[0]*0.6)), (int(imshape[1]*0.546), int(imshape[0]*0.6)), (imshape[1]-50,imshape[0])]], dtype=np.int32)
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def measure_deviation(leftx_start, rightx_start, midpoint):
    xm_per_pix = 3.7/600
    # Calculation of deviation from lane center to camera center in meter
    lane_centre_pixel =(rightx_start-leftx_start)/2 + leftx_start
    deviation = (midpoint - lane_centre_pixel) * xm_per_pix

    return deviation

def draw_lane(warped, image, left_fitx, right_fitx, Minv):
    
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    image = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    return image

def display_info(image, lane_curverad_real, deviation):
    
    txt1 = 'Radius of curvature is ' + str(int(lane_curverad_real)) + 'm'

    if deviation < 0: 
        txt2 = 'Vehicle is '+ str(round(abs(deviation),2)) + 'm left of center'
    else:
        txt2 = 'Vehicle is '+ str(round(abs(deviation),2)) + 'm right of center'
    
    # Using cv2.putText() method to display radius of curvature and offset on video
    image = cv2.putText(image, txt1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255), thickness=1) 
    image = cv2.putText(image, txt2, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255), thickness=1) 
    return image

def combined_thresh(img):
    # Use calibrated camera matrix and distortion coefficient
    mtx = np.array([[1.15777818e+03, 0.00000000e+00, 6.67113857e+02],
       [0.00000000e+00, 1.15282217e+03, 3.86124583e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist = np.array([[-0.24688507, -0.02373155, -0.00109831,  0.00035107, -0.00259868]])

    # Apply distortion correction on image
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # Compute sobel thresholded binaries
    # In x
    gradx = abs_sobel_thresh(undist, orient='x', thresh_min=20, thresh_max=100)
    # In y
    grady = abs_sobel_thresh(undist, orient='y', thresh_min=40, thresh_max=100)

    # Compute gradient magnitudes thresholded binary
    mag_binary = mag_thresh(undist, sobel_kernel=9, mag_thresh=(80, 98))

    # Compute gradient direction thresholded binary
    dir_binary = dir_thresh(undist, sobel_kernel=15, thresh=(0.7, 1.4))

    # Compute S channel thresholded binary
    s_binary = hls_select(undist, thresh=(110, 255))

    # Compute R channel thresholded binary
    r_binary = rgb_select(undist, thresh=(205, 255))

    # Combined gradient thresholded binaries
    grad_binary = np.zeros_like(gradx)
    grad_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] =1

    # Combined color thresholded binaries
    sr_binary = np.zeros_like(s_binary)
    sr_binary[((s_binary == 1)) & (r_binary == 1)] =1

    # Combined gradient and color thresholding binaries
    combined_binary = np.zeros_like(sr_binary)
    combined_binary[(grad_binary == 1) | (sr_binary == 1)] = 1

    return undist, combined_binary

def get_warper(img, src, dst):
    img_size = (img.shape[1], img.shape[0])
    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # use cv2.getPerspectiveTransform() to get Minv, the inverse transform matrix
    Minv = cv2.getPerspectiveTransform(dst, src)

    # apply region of interest
    masked_img = region_of_interest(img)

    # use cv2.warpPerspective() to warp image to a top-down view
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    # warped = cv2.warpPerspective(masked_img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M, Minv


def get_warped_binary(img):

    img, binary = combined_thresh(img)

    img_size = (img.shape[1], img.shape[0])
    # Pick up four points in a trapezoidal shape
    # Define up and low corners on the left
    mid_point = int(img_size[0]/2)
    corner_up_left = (mid_point - 47,450)
    corner_low_left = (mid_point - 448,img_size[1] - 1)
    
    # Define up and low corners on the right
    corner_up_right = (mid_point + 52,450)
    corner_low_right = (mid_point + 488,img_size[1] - 1)

    # Construct four source points
    src =np.float32([corner_up_left, corner_low_left,corner_up_right,corner_low_right])   
    
    # define 4 destination points
    offset = 340
    dst = np.float32([[offset, 0], [offset, img_size[1]-1], 
                                    [img_size[0]-offset, 0], 
                                    [img_size[0]-offset, img_size[1]-1]])

    warped_binary, M, Minv = get_warper(binary, src, dst)

    return img, warped_binary, Minv