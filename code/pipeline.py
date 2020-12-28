import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from Line import Line

from lane_detection import *

left_line = Line(side='left')
right_line = Line(side='right')

def Pipeline(image):

    image, warped_binary, Minv = get_warped_binary(image)
    left_line.find_line(warped_binary)
    right_line.find_line(warped_binary)
    radius_of_curvature = (left_line.radius_of_curvature + right_line.radius_of_curvature)/2
    deviation = measure_deviation(left_line.bestx[-1], right_line.bestx[-1], midpoint = int(image.shape[1]/2))
    image = draw_lane(warped_binary, image, left_line.bestx, right_line.bestx, Minv)
    image = display_info(image,radius_of_curvature,deviation)

    return image

# Pipeline (single images)
# image = mpimg.imread('./test_images/test4.jpg')
# image = Pipeline(image)
# plt.figure(figsize = (12,7))
# plt.imshow(image)
# plt.show()

# Pipeline (video)
output = 'output_videos/project_video_output1.mp4'
clip1 = VideoFileClip("./project_video.mp4")
clip_out = clip1.fl_image(Pipeline)
clip_out.write_videofile(output, audio=False)
