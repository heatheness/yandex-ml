# -*- coding: utf-8 -*-

__author__ = 'nyash myash'

import numpy as np
import skimage
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.io import imread
import math


image = skimage.img_as_float(imread('parrots.jpg'))

print image.shape[0]
print image.shape[1]


print image.shape

x = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))

print x.shape

clr = KMeans(random_state=241)
y = clr.fit_predict(x)

def resize(x,y,resize_func):
    cluster_num = {}
    resised_colors = {}

    # pixel to cluster dict
    for i in xrange(len(x)):
        cluster_num.setdefault(y[i], [])
        cluster_num[y[i]].append(x[i])

    # finds resized color for RGB coordinate
    for i in cluster_num:
        red = resize_func(map(lambda z: z[0], cluster_num[i]))
        green = resize_func(map(lambda z: z[1], cluster_num[i]))
        blue = resize_func(map(lambda z: z[2], cluster_num[i]))
        resised_colors[i] = [red, green, blue]

    resized_x = np.asarray(map(lambda z: resised_colors[z], y))

    return resized_x



def MSE(x, resized_x):
    # x = np.reshape(x, image.shape)
    # resized_x = np.reshape(resized_x, image.shape)
    mn = float(image.shape[0] * image.shape[1])
    dif = np.subtract(x, resized_x)
    mse = np.sum(np.power(dif, 2)) / mn / 3


    return mse


def PSNR(x, resized_x):


    mn = float(image.shape[0] * image.shape[1])
    dif = np.subtract(x, resized_x)
    mse = np.sum(np.power(dif, 2)) / mn / 3
    psnr = 20 * math.log10(1/math.sqrt(mse))


    return psnr

res = {}

for i in xrange(1,21):
    clr = KMeans(n_clusters=i, init='k-means++', random_state=241)
    y = clr.fit_predict(x)

    mean_x = resize(x,y,np.mean)
    median_x = resize(x,y,np.median)

    # print mean_x[2:3]
    # print x[2:3]
    # print np.power(np.subtract(x[2:3], mean_x[2:3]),2)
    # break

    # print type(mean_x)
    # print mean_x.shape

    MSE_mean = MSE(x, mean_x)
    MSE_median = MSE(x, median_x)

    print "MSE mean", MSE_mean
    print "MSE median", MSE_median



    mean_resize_psnr = np.round(PSNR(x, mean_x),2)
    median_resize_psnr = np.round(PSNR(x, median_x),2)

    res[i] = [mean_resize_psnr, median_resize_psnr]

    print "Cluster size", i
    print "Mean PSNR", mean_resize_psnr
    print "Median PSNR", median_resize_psnr


