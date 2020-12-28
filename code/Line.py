import numpy as np
import cv2
# import lane_detection as ld
import matplotlib.pyplot as plt
# from lane_detection import fit_polynomial

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, side=None):

        # Define side of line: left or right, left by default
        if side is None:
            self.side = 'left'
        else:
            self.side = side
        
        # was the line detected in the last iteration?
        self.detected = False 
        # # polynomial coefficients of the last n fits of the line
        self.recent_fits = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = np.array([0,0,0], dtype='float')  
        #polynomial coefficients for current fit
        self.current_fit = np.array([0,0,0], dtype='float')
        # x values of current fit of the line
        self.current_fitx = []         
        
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 

        #line fit sanity checks over the last n iterations
        self.recent_sanities = []

        #line fit sanity check  for current iteration
        self.sanity = False
        #Frame number in video
        self.nframe = 0

    def lane_fit_check(self):

        # R squared value based sanity check
        if self.bestx is None:
            return True
        else:
            correlation_matrix = np.corrcoef(self.current_fitx, self.bestx)
            correlation_xy = correlation_matrix[0,1]
            r_squared = correlation_xy**2

            # Potential lane curvature change
            if r_squared < 0.98:
                return False
            else: 
                return True
            # return None  

    def lane_pos_check(self):
        if self.bestx is None:
            return True
        else:
            line_pos_diff = np.asscalar(np.abs(self.current_fitx[-1] - self.bestx[-1]))

            if (line_pos_diff > 100): 
            #lane_width > 3.7 * 1.2 | lane_width < 3.7 * 0.8:
                # self.detected = False
                return False
            else: 
                # self.detected = True
                return True
    
    def lane_curvature_check(self, ploty, fitx):
        if self.radius_of_curvature is None:
            return True
        else:
            rc = self.measure_curvature(ploty, fitx) 
            if np.abs(rc-self.radius_of_curvature) > 1000:
                return False
            else:
                return True

    def recent_sanity_check(self):
        
        # lane detection fails for number of frames in a row
        n_fail = 2
        if len(self.recent_sanities) >=2:
            if np.count_nonzero(np.array(self.recent_sanities[-n_fail:])) == 0:
            # if len(self.recent_sanities) - np.count_nonzero(np.array(self.recent_sanities)) >= 2:
                self.detected = False
            else:
                self.detected = True
                
        # else:
        #     self.detected = True
        
        return self.detected

    def find_line(self, binary_warped):
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        if not self.detected:
            # Search line using histogram within sliding window
            self.allx, self.ally = self.search_pixels_sld_win(binary_warped)
            self.current_fit, self.current_fitx = self.fit_poly(self.ally, self.allx, ploty)
            self.reset() 
        else:
            self.allx, self.ally = self.search_pixels_prior_fit(binary_warped)
            self.current_fit, self.current_fitx = self.fit_poly(self.ally, self.allx, ploty)
            # Current measurement sanity check
            self.sanity = self.lane_fit_check() and self.lane_pos_check() and self.lane_curvature_check(ploty, self.current_fitx)
            # Append current sanity check to the sanities list
            self.recent_sanities.append(self.sanity)
            # If current sanity check's ok, append current fit to the list
            if self.sanity:
                self.recent_fits.append(self.current_fit)
            # Perform historical sanity check to decide if restart with sliding window is needed
            self.recent_sanity_check()
            self.set_bestx(ploty)

        self.update_info(ploty)
        
        return None

    def set_bestx(self, ploty):
        # Average fit coefficients over past iterations
        if len(self.recent_fits) > 1:
            self.best_fit = np.mean(np.array(self.recent_fits), axis = 0)
        else:
            self.best_fit = np.array(self.recent_fits).reshape(3,)

        self.bestx = self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]

    def reset(self):

        # Reset parameters after restart with sliding window search
        self.recent_fits =[]
        
        if self.nframe > 0: 
            self.recent_fits.append(self.current_fit)

        self.best_fit = self.current_fit
        self.bestx = self.current_fitx

        self.recent_sanities = []
        self.sanity = True
        self.recent_sanities.append(self.sanity)
        self.detected = True


    def update_info(self,ploty):
        # Save the characteristics of detected line
        # save parameters over last n frames
        n_last = 5
        if len(self.recent_fits) > n_last:
            self.recent_fits.pop(1)
            self.recent_sanities.pop(1)

        self.nframe += 1
        self.radius_of_curvature = self.measure_curvature(ploty, self.bestx)
        return None

    def search_pixels_sld_win(self, binary_warped):
        # Take a histogram of the bottom half of the image
        
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)

        if self.side == 'left':
            x_base = np.argmax(histogram[:midpoint])

        if self.side == 'right':
            x_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        x_current = x_base

        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            # Find the below boundaries of the window 
            win_x_low = x_current - margin 
            win_x_high = x_current + margin
            
            # Draw the windows on the visualization image
            # cv2.rectangle(out_img,(win_x_low,win_y_low),
            # (win_x_high,win_y_high),(0,255,0), 2) 

            # Identify the nonzero pixels in x and y within the window 
            inds_win_y = (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
            inds_win_x = (nonzerox >= win_x_low) & (nonzerox < win_x_high)
            
            good_inds = (inds_win_x & inds_win_y).nonzero()[0]

            # Append these indices to the lists
            lane_inds.append(good_inds)
            
            # Recenter next window if number pixels > minpix pixels, 
            ### (`right` or `leftx_current`) on their mean position ###
            if good_inds.size > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            lane_inds = np.concatenate(lane_inds)
            # right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        allx = nonzerox[lane_inds]
        ally = nonzeroy[lane_inds] 

        return allx, ally
    
    def fit_poly(self, ally, allx, ploty):
        # Fit a second order polynomial to each using `np.polyfit` ###
        fit = np.polyfit(ally,allx,2)

        # Generate x and y values for plotting
        try:
            fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
        except TypeError:
            # Avoids an error if fit are still none or incorrect
            print('The function failed to fit a line!')
            fitx = 1*ploty**2 + 1*ploty

        return fit, fitx

    def search_pixels_prior_fit(self, binary_warped):

        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        margin = 100

        # Get activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Set the search window to find activated pixels within the +/- margin using the prior polynomial coefficients 
        nonzerofitx = self.best_fit[0]*nonzeroy**2 + self.best_fit[1]*nonzeroy + self.best_fit[2]
        lane_inds = (nonzerox<(nonzerofitx+margin)) & (nonzerox>= (nonzerofitx-margin))
        
        # Extract left and right line pixel positions
        allx = nonzerox[lane_inds]
        ally = nonzeroy[lane_inds]
        
        return allx, ally

    def measure_curvature(self, ploty, fitx):
        '''
        Calculates the curvature of polynomial functions in meters
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 3/(680-600) # meters per pixel in y dimension
        xm_per_pix = 3.7/600# meters per pixel in x dimension
        
        # Define y-value where we want radius of curvature
        y_eval = ploty

        # Fit a new polynomials from pixel positions to x, y in world space
        fit_cr = np.polyfit(ploty*ym_per_pix, fitx*xm_per_pix, 2)
        
        # Calculation of average radius of curvature in meter given y-value
        lane_curverad_real= np.mean(((1 + (2*fit_cr[0]*y_eval*ym_per_pix  + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0]))
        
        return lane_curverad_real