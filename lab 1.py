
from scipy.ndimage import imread
import matplotlib.pyplot as plt
import numpy as np
import math

#load and display files for use in this projects
cat = imread('cat.jpg')
kitten = imread('kitten.png')
plt.imshow(cat);
plt.show()
plt.imshow(kitten);
plt.show()

#convert an image to gray scale - looses a dimension
def toGrayScale(image):
    rscale = image[:, :, 0] * .299
    gscale = image[:, :, 1] * .587
    bscale = image[:, :, 2] * .114
    return rscale + gscale + bscale


# Test Case
gray_cat = toGrayScale(cat)
gray_kitten = toGrayScale(kitten)
plt.imshow(gray_cat, cmap="Greys_r", vmin=0, vmax=255)
plt.title("Grayscale Cat")
plt.show()
plt.imshow(gray_kitten, cmap="Greys_r", vmin=0, vmax=255)
plt.title("Grayscale Kitten")
plt.show()

# adds a integer value to the image, all pixels are effected
def brightAdjust(image, c):
    print(image.shape)
    light = image[:, :] + c
    light = np.clip(light, 0, 255)
    return light


# Test Case
bright_cat = brightAdjust(gray_cat, 100)
plt.imshow(bright_cat, cmap="Greys_r", vmin=0, vmax=255)
plt.title("Bright Cat")
plt.show()
dark_kitten = brightAdjust(gray_kitten, -100)
plt.imshow(dark_kitten, cmap="Greys_r", vmin=0, vmax=255)
plt.title("Dark Kitten")
plt.show()

# a helper function: convolves an image with the given 3X3 kernel and divides it by div.
def convolution(image, kernel, div):
    r = np.shape(image)[0]
    c = np.shape(image)[1]
    img_out = np.zeros((r, c))
    region = np.zeros((r - 3, c - 3))
    region = image[0:(r - 3), 0:(c - 3)]

    for s in range(3):
        for t in range(3):
            img_out += np.lib.pad(region, (((0 + s), (3 - s)), ((0 + t), (3 - t))), 'constant',
                                  constant_values=((0, 0), (0, 0))) * kernel.item((s, t))
            # print(kernel.item((s, t)))
            # print(img_out)
            # print("\n")
    # print("\nfinal\n")
    # print(img_out)
    img_out[:] = img_out[:] / div
    # print("\nDivision\n")
    # print(img_out)
    return img_out

#filters an image for the median value in an size X size kernel
def medianFilter(image, size=3):
    rows = np.shape(image)[0]
    colums = np.shape(image)[1]
    median = np.zeros((rows, colums))
    offset = size // 2
    bounds = offset * 2
    for x in range(rows - bounds):
        for y in range(colums - bounds):
            seed = image[x:(x + size), y:(y + size)]
            median[(x + offset)][(y + offset)] = np.median(seed)
    return median

# Test Cases
median_kitten = medianFilter(gray_kitten)
plt.imshow(median_kitten, cmap="Greys_r", vmin=0, vmax=255);
plt.title("Median Filtering")
plt.show()
median_cat = medianFilter(gray_cat)
plt.imshow(median_cat, cmap="Greys_r", vmin=0, vmax=255);
plt.title("Median Filtering")
plt.show()


#test case for blurring
plt.imshow(gray_kitten, cmap="Greys_r", vmin=0, vmax=255);
plt.title("before Uniform Blurring")
plt.show()

blur_kernel = np.matrix([[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]])

con_kitten = convolution(gray_kitten, blur_kernel, 9)
plt.imshow(con_kitten, cmap="gray", vmin=0, vmax=255);
plt.title("after Uniform Bluring")
plt.show()

con_cat = convolution(gray_cat, blur_kernel, 9)
plt.imshow(con_cat, cmap="gray", vmin=0, vmax=255);
plt.title("after Uniform Bluring")
plt.show()

#test cases for unsharp masking​
sharp_kernel = np.matrix([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])

plt.imshow(gray_kitten, cmap="Greys_r", vmin=0, vmax=255);
plt.title("before Sharpening")
plt.show()

con_kitten = convolution(gray_kitten, sharp_kernel, 1)
plt.imshow(con_kitten, cmap="gray", vmin=0, vmax=255);
plt.title("after Sharpening")
plt.show()

plt.imshow(gray_cat, cmap="Greys_r", vmin=0, vmax=255);
plt.title("before Sharpening")
plt.show()

con_cat = convolution(gray_cat, sharp_kernel, 1)
plt.imshow(con_cat, cmap="gray", vmin=0, vmax=255);
plt.title("after Sharpening")
plt.show()

#test cases for gradient magnitude
gx = np.matrix([[1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]])
gy = np.matrix([[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]])

plt.imshow(gray_kitten, cmap="Greys_r", vmin=0, vmax=255);
plt.title("before Gradient")
plt.show()

gx_kitten = convolution(gray_kitten, gx, 1)
gy_kitten = convolution(gray_kitten, gy, 1)
mag_kitten = np.sqrt((gx_kitten ** 2) + (gy_kitten ** 2))
plt.imshow(mag_kitten, cmap="gray", vmin=0, vmax=255);
plt.title("after Gradient")
plt.show()

plt.imshow(gray_cat, cmap="Greys_r", vmin=0, vmax=255);
plt.title("before Gradient")
plt.show()

gx_cat = convolution(gray_cat, gx, 1)
gy_cat = convolution(gray_cat, gy, 1)
mag_cat = np.sqrt((gx_cat ** 2) + (gy_cat ** 2))
plt.imshow(mag_cat, cmap="gray", vmin=0, vmax=255);
plt.title("after Gradient")
plt.show()
