# cs445-Final-Project
Final project for cs445 - Gradient sketching

This Python script is intended to take a still image and create a gif of the image being "sketched" as if an artist were drawing it. This is the final project for Tom Phelan and Will Wang for CS445 - Computational Photography, Spring 2020 semester.

To use this code, open the sketcher.py file or use the Final.ipynb Jupyter Notebook (code is identical). Load an image file using cv2.imread() and convert to grayscale as in example provided. Then use the gif_creator() function with three arguments - the image, the speed of pixel fill-in per frame, and the filepath for the output gif file.

The resulting "sketching" gif will be saved where the filepath designates.
