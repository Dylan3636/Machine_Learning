# import numpy as np
# import gym
# import cv2
# import matplotlib.pyplot as plt
# env = gym.make('CarRacing-v0')
# plt.ion()
# for i in range(1):
#     env.reset()
#     for j in range(100):
#         env.render()
#         action = env.action_space.sample()
#         obs,_,done,_=env.step(action)
#         # if j in range(30, 50):
#         #     plt.imsave('car_racing_img_{}.png'.format(j), obs)
#         if done:
#             break

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def compute_steering_speed_gyro_abs(a):
    right_steering = a[6, 36:46].mean() / 255
    left_steering = a[6, 26:36].mean() / 255
    steering = (right_steering - left_steering + 1.0) / 2

    left_gyro = a[6, 46:60].mean() / 255
    right_gyro = a[6, 60:76].mean() / 255
    gyro = (right_gyro - left_gyro + 1.0) / 2

    speed = a[:, 0][:-2].mean() / 255
    abs1 = a[:, 6][:-2].mean() / 255
    abs2 = a[:, 8][:-2].mean() / 255
    abs3 = a[:, 10][:-2].mean() / 255
    abs4 = a[:, 12][:-2].mean() / 255

    #     white = np.ones((round(speed * 100), 10))
    #     black = np.zeros((round(100 - speed * 100), 10))
    #     speed_display = np.concatenate((black, white))*255

    #     cv2.imshow('sensors', speed_display)
    #     cv2.waitKey(1)


    return [steering, speed, gyro, abs1, abs2, abs3, abs4]

img = cv2.imread('car_racing_img_30.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgs = [img, img_gray]
for i in range(30, 50):
    img = cv2.imread('car_racing_img_{}.png'.format(i))
    print(np.shape(img))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img_gray, 150, 200, cv2.THRESH_BINARY_INV)
    img_gray = cv2.bitwise_and(img_gray, img_gray, mask=mask)
    #plt.scatter(range(len(img_gray.ravel())), img_gray.ravel())
    #plt.show()
    cv2.namedWindow('Gray', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Gray', 600, 600)
    cv2.imshow('Gray', cv2.resize(img_gray,(60,60)))
    k = cv2.waitKey(5) & 0xFF
    time.sleep(1)
