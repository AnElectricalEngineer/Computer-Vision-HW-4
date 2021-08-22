# Homographies

## About Homographies

In this work, we use homographies to create panorama images from scratch (without using functions from OpenCV/Scipy/other), and to warp planar images into other images. 

## Panorama Creation

This is accomplished using several functions. Corresponding image points between two images are either manually selected using `ginput()`, or automatically detected using SIFT keypoints. The homography between the two images is then calculated by solving the equation Ah = 0. A warping function uses either linear or cubic interpolation to warp one image to the perspective of the other. The images can then be stitched together. The complete panorama is formed by performing the aforementioned steps iteratively. For a complete explanation, see the attached PDF of our report.

## Planar Image Warping

In order to warp a planar image (for example, an image of a book cover), warping is first applied to a raw image of a book to ensure that it is relatively planar. Next, the homography between the (planar) input image and the section of the image to be replaced is calcuated. Finally, a mask is used to stitch the input image onto the output image, in the proper location and perspective.

## Selected Screenshots

![image](https://user-images.githubusercontent.com/47844685/130359677-50efd468-d977-4040-ad57-f55ed91dac54.png)
![image](https://user-images.githubusercontent.com/47844685/130359701-a9a09da9-da8f-4df8-bf15-6f731722916a.png)
![image](https://user-images.githubusercontent.com/47844685/130359716-a6f04ccd-0610-4276-9d9d-fcd6b7101817.png)

For more details on implementation and analysis, see the attached PDF of our report.
