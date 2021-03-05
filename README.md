## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Repo of Advanced Lane Finding project of Udacity - Self-Driving Car NanoDegree, all codes presented below are located in `./code` folder, and udacity's original repo goes to [here](https://github.com/kaoozhi/CarND-Advanced-Lane-Lines)
<!-- ![Lanes Image](./examples/example_output.jpg) -->

### Writeup / README
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.jpg "Undistorted"
[image2]: ./output_images/distort_cor_test4.jpg "Road Transformed"
[image3]: ./output_images/grad_output.jpg "Sobel thresholds"
[image4]: ./output_images/grad_mag_output.jpg "Magnitude thresholds"
[image5]: ./output_images/grad_dir_output.jpg "Direction thresholds"
[image6]: ./output_images/sbinary_output.jpg "S Channel thresholds"
[image7]: ./output_images/rbinary_output.jpg "R Channel thresholds"
[image8]: ./output_images/combined_binary_output.jpg "Binary Example"
[image9]: ./output_images/warped_straight_lines.jpg "Warp Example"
[image10]: ./output_images/polyfit_sld_win.jpg "Fit Visual Sliding Window"
[image11]: ./output_images/polyfit_prior_fit.jpg "Fit Visual look ahead"
[image12]: ./output_images/final_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

<!-- ## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.   -->

---

### Camera Calibration

#### 1. Compute camera calibration matrix and distortion coefficients

The code for this step is contained in the IPython notebook `P2_camera_calibration.ipynb`

The size of chessboard size is 9x6 for the project. I start by preparing the object points representing each chessboard corner which has the (x, y, z) coordinates of the chessboard corners in the world. Assuming the chessboard is fixed on the (x, y) plane at z=0, so a array of size (6x9x3) `objp` is finally created for object points, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time all chessboard corners are detected in a test image using `cv2.findChessboardCorners`.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Then the output `objpoints` and `imgpoints` is used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)
The pipeline is defined by the function `Pipeline()` in file `pipeline.py`
#### 1. Apply distortion correction on road image
The distortion correction is involved in function `combined_thresh()` in lines #153 through #159 in `lane_detection.py`, and the camera calibration and distortion coefficients are hardcoded parameters of the function.

Here's an example of a test image (`./test_images/test4.jpg`) after being applied the undistortion:
![alt text][image2]

#### 2. Create a thresholded binary image using color transforms and gradients.

I used a combination of color and gradient thresholds to generate a binary image in `combined_thresh()` the thresholding steps are in lines #151 through #189 in `lane_detection.py` where the sub-functions of thresholds methods are located as well.
##### Sobel Gradient thresholds
I first create two binaries with sobel thresholds in x and y by sub-function `abs_sobel_thresh()` with the following parameters:
```python
# Compute sobel thresholded binaries
# In x
gradx = abs_sobel_thresh(undist, orient='x', thresh_min=20, thresh_max=100)
# In y
grady = abs_sobel_thresh(undist, orient='y', thresh_min=40, thresh_max=100)
```
![alt text][image3]

##### Gradient magnitude thresholds
Then a binary with gradient magnitude by sub-function `mag_thresh()` with the following parameters:
```python
# Compute gradient magnitudes thresholded binary
mag_binary = mag_thresh(undist, sobel_kernel=9, mag_thresh=(80, 98))
```
![alt text][image4]

##### Gradient direction thresholds
Then a binary with gradient direction by sub-function `dir_thresh()` with the following parameters:
```python
# Compute gradient direction thresholded binary
dir_binary = dir_thresh(undist, sobel_kernel=15, thresh=(0.7, 1.4))
```
![alt text][image5]

##### S Channel thresholds
Then I convert image in HSL space, and create a binary with thresholds in S channel by sub-function `hls_select()` with the following parameters:
```python
# Compute S channel thresholded binary
s_binary = hls_select(undist, thresh=(110, 255))
```
![alt text][image6]

##### R Channel thresholds
I create as well a binary with thresholds in R channel by sub-function `rgb_select()` with the following parameters:
```python
# Compute R channel thresholded binary
r_binary = rgb_select(undist, thresh=(205, 255))
```
![alt text][image7]

##### Combination of thresholds
Finally, I used the following method to combine gradient and color thresholds:
```python
# Combined gradient thresholded binaries
grad_binary = np.zeros_like(gradx)
grad_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] =1

# Combined color thresholded binaries
sr_binary = np.zeros_like(s_binary)
sr_binary[((s_binary == 1)) & (r_binary == 1)] =1

# Combined gradient and color thresholded binaries
combined_binary = np.zeros_like(sr_binary)
combined_binary[(grad_binary == 1) | (sr_binary == 1)] = 1
```
The final output of combined thresholded binary (green stack for the gradient thresholds, blue stack for the color thresholds) is:
![alt text][image8]


#### 3. Perform a perspective transform

The code for the perspective transform includes a function called `get_warper()`, which are in lines #193 through #208 in the file `lane_detection.py` .The `get_warper()` function takes as inputs an image (`img`), source points (`src`) and destination points(`dst`). The source and destination points are hardcoded in the following manner:

```python
mid_point = int(img_size[0]/2)
corner_up_left = (mid_point - 47,450)
corner_low_left = (mid_point - 448,img_size[1] - 1)
corner_up_right = (mid_point + 52,450)
corner_low_right = (mid_point + 488,img_size[1] - 1)
src =np.float32([corner_up_left, corner_low_left,corner_up_right,corner_low_right])   

offset = 340
dst = np.float32([[offset, 0], [offset, img_size[1]-1],
                                [img_size[0]-offset, 0],
                                [img_size[0]-offset, img_size[1]-1]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 593, 450      | 340, 0        |
| 192, 719      | 340, 719      |
| 692, 450     | 940, 0      |
| 1128, 719      | 940, 719        |

The perspective transform was working as expected by drawing the `src` and `dst` points onto a test image(`./test_images/straight_lines2.jpg`) and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image9]

#### 4.Identify lane-line pixels and polynomial fit
I created a class `Line()` in file (`Line.py`) to identify lane-line and store useful characteristics. An instance of the class `Line()` will be created for the left and right lane lines in the pipeline. To find lane line pixels, the class method `find_line()` will call another method `search_pixels_sld_win()` which will take a warped combined binary `binary_warped` as input to find lane pixels using sliding window based on the sum of pixel histogram and fit lane-line position with 2nd order polynomial. Here's example with both left and right lane lines fitted on (`./test_images/test4.jpg`):
![alt text][image10]

If the lane is successfully detected in one frame of the video, `find_line()` method can search lane line pixels in the next frame within a window around the previous detection using class method `search_pixels_prior_fit()`. Here's example also on (`./test_images/test4.jpg`) by supposing that the yellow fitted lines are detection from the previous frame

![alt text][image11]


#### 5. Radius of curvature of the lane and the position of the vehicle with respect to center.

I used the class method `measure_curvature()` of `Line()` to calculate the radius of curvature given the polynomial fit coefficients and y breakpoints value in pixel. I hardcoded the scales from pixels to meters in x and y which I measured on a undistorted road image as follows:
```python
ym_per_pix = 3/(680-600) # meters per pixel in y dimension
xm_per_pix = 3.7/600# meters per pixel in x dimension
```
I fit a new polynomials in meters space with the above scales, then I compute the values of radius of curvature in meter using the new fitted coefficients corresponding to the given y breakpoints in pixel and average them as the radius of curvature of the left or right line, and it will be stored by a member variable `Line.radius_of_curvature`
The lane's radius of curvature is computed in the pipeline by taking the average of the left and right ones.

I defined another function `measure_deviation()` in `lane_detection.py` which takes the left and right lines' ftted x values on the bottom and the midpoint in x of image with the same hardcoded scale in x to compute the offset of vehicle from center in meter.

#### 6. Draw lane.

I implemented this step in `lane_detection.py` in the functions `draw_lane()` and `display_info()`.  Here is an example of my result on a test image (`./test_images/test4.jpg`):

![alt text][image12]

---

### Pipeline (video)

#### 1. Final video output

The actual implementation successfully detects lane-line and draw smooth lane area on the project video
Here's a [link to my video result](http://www.youtube.com/watch?v=zbXTBdxCMsg "")

[![](http://img.youtube.com/vi/zbXTBdxCMsg/0.jpg)](http://www.youtube.com/watch?v=zbXTBdxCMsg "")


---

### Discussion

#### Reflexions and improvements

Several methods and class variables of class `Line()` are created for the pipeline to be worked on video. In general, each time I detected line with look-ahead filter by `search_pixels_prior_fit()` method, I will check the similarity of current measurement compared to the past ones by R-squared value, curvature and lane start position. If the check's ok, I append current fit coefficient `current_fit` to the list `recent_fits`. I then average the list `recent_fits` over the last 5 frames to get `best_fit` which will be the final polynomial coefficients to be used for smooth drawing. If the check is not ok, I just retain the previous position of the last frame and do the search for the next frame. If the check fails for both the last 2 frames, I will start over the search by `search_pixels_sld_win()` method to re-establish the measurement.

The choice of source and destination points to perform perspective transform is a first tricky part of the project, I understood it was a very empirical process in order to have reasonably accurate results.

Secondly, the actual combination of thresholding method does not perform well on challenge videos especially when the middle road barrier has sharp shadow or is close to the left lane. One should retry other threshold parameters or different combination methods for the pipeline to work well on more delicate situations.
