# coding:utf-8

import os
import scipy.io
import urllib.request
import urllib.parse
import numpy as np
import random as rr
import math as mt
from keras.preprocessing.image import ImageDataGenerator


class DataGenerator():
    def __init__(self, mat_path, width, height, batch_size, load_image_num):
        self.width = width
        self.height = height
        image_num = batch_size
        # self.load_image(mat_path, load_image_num)

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        self.train_generator = train_datagen.flow_from_directory(
            'data',
            target_size=(self.width, self.height),
            batch_size=batch_size,
            class_mode='input')

    def load_image(self, mat_path, load_image_num):
        save_path = "./data/0"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        mat_data = scipy.io.loadmat(mat_path)
        sun_urls = mat_data['SUN']
        cntr = 0
        for urls in sun_urls:
            for u in urls:
                try:
                    img_data = urllib.request.urlopen(u[2][0][0][0]).read()
                    file_name = "{:010}.jpg".format(cntr)
                    f = open(os.path.join(
                        save_path, file_name), 'wb')

                    f.write(img_data)
                    f.close()
                    cntr += 1
                    if cntr == load_image_num:
                        return

                except Exception as e:
                    print(e)

    def get_data(self):
        image, _ = next(self.train_generator)
        masks = self.create_mask(image.shape[0], 50, 30, 3.14, 5, 15)

        return image, masks, image*masks

    def create_mask(self, image_num, maxLen, maxWid, maxAng, maxNum, maxVer, minLen=20, minWid=15, minVer=5):

        mask = np.ones((image_num, self.width, self.width, 3))

        num = rr.randint(3, maxNum)

        for i in range(num):
            startX = rr.randint(0, self.width)
            startY = rr.randint(0, self.width)
            numVer = rr.randint(minVer, maxVer)
            width = rr.randint(minWid, maxWid)
            for j in range(numVer):
                angle = rr.uniform(-maxAng, maxAng)
                length = rr.randint(minLen, maxLen)

                endX = min(
                    self.width-1, max(0, int(startX + length * mt.sin(angle))))
                endY = min(
                    self.width-1, max(0, int(startY + length * mt.cos(angle))))

                if endX >= startX:
                    lowx = startX
                    highx = endX
                else:
                    lowx = endX
                    highx = startX
                if endY >= startY:
                    lowy = startY
                    highy = endY
                else:
                    lowy = endY
                    highy = startY

                if abs(startY-endY) + abs(startX - endX) != 0:

                    wlx = max(0, lowx-int(abs(width * mt.cos(angle))))
                    whx = min(self.width - 1,  highx+1 +
                              int(abs(width * mt.cos(angle))))
                    wly = max(0, lowy - int(abs(width * mt.sin(angle))))
                    why = min(self.width - 1, highy+1 +
                              int(abs(width * mt.sin(angle))))

                    for x in range(wlx, whx):
                        for y in range(wly, why):

                            d = abs((endY-startY)*x - (endX - startX)*y - endY*startX +
                                    startY*endX) / mt.sqrt((startY-endY)**2 + (startX - endX)**2)

                            if d <= width:
                                mask[:, x, y, :] = 0

                wlx = max(0, lowx-width)
                whx = min(self.width - 1, highx+width+1)
                wly = max(0, lowy - width)
                why = min(self.width - 1, highy + width + 1)

                for x2 in range(wlx, whx):
                    for y2 in range(wly, why):

                        d1 = (startX - x2) ** 2 + (startY - y2) ** 2
                        d2 = (endX - x2) ** 2 + (endY - y2) ** 2

                        if np.sqrt(d1) <= width:
                            mask[:, x2, y2, :] = 0
                        if np.sqrt(d2) <= width:
                            mask[:, x2, y2, :] = 0
                startX = endX
                startY = endY

        return mask


t = DataGenerator("./SUN_urls.mat", 140, 140, 12, 5)
t.get_data()
