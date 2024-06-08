from __future__ import print_function
import os
import sys
import cv2
import numpy as np

def load_images(directory):
    print("Loading images from " + directory, end = "...")
    image_list = []

    for file_name in sorted(os.listdir(directory)):
        file_extension = os.path.splitext(file_name)[1]
        if file_extension in [".jpg", ".jpeg"]:
            full_path = os.path.join(directory, file_name)
            img = cv2.imread(full_path)

            if img is None :
                print("Image:{} not read properly".format(full_path))
            else :
                img = np.float32(img) / 255.0
                image_list.append(img)
                flipped_img = cv2.flip(img, 1)
                image_list.append(flipped_img)
    num_images = int(len(image_list) / 2)

    if num_images == 0 :
        print("No images found")
        sys.exit(0)
    print(str(num_images) + " files loaded.")
    return image_list

def generate_data_matrix(image_list):
    print("Generating data matrix", end = " ... ")
    num_images = len(image_list)
    size = image_list[0].shape
    data_matrix = np.zeros((num_images, size[0] * size[1] * size[2]), dtype = np.float32)

    for i in range(num_images):
        img_flat = image_list[i].flatten()
        data_matrix[i, :] = img_flat
    print("DONE")
    return data_matrix

def generate_new_face(*args):
    output_img = mean_face

    for i in range(NUM_EIGEN_FACES):
        slider_values[i] = cv2.getTrackbarPos("Weight" + str(i), "Trackbars")
        weight = slider_values[i] - MAX_SLIDER_VALUE / 2
        output_img = np.add(output_img, eigen_faces[i] * weight)
    output_img = cv2.resize(output_img, (0, 0), fx = 2, fy = 2)
    cv2.imshow("Result", output_img)

def reset_sliders(*args):
    for i in range(NUM_EIGEN_FACES):
        cv2.setTrackbarPos("Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE / 2))
    generate_new_face()

if __name__ == '__main__':
    NUM_EIGEN_FACES = 10
    MAX_SLIDER_VALUE = 255

    images_dir = "images"
    image_list = load_images(images_dir)
    img_size = image_list[0].shape
    data_matrix = generate_data_matrix(image_list)
    print("Performing PCA ", end = "...")
    mean, eigen_vectors = cv2.PCACompute(data_matrix, mean = None, maxComponents = NUM_EIGEN_FACES)
    print ("DONE")
    mean_face = mean.reshape(img_size)
    eigen_faces  = []

    for vector in eigen_vectors:
        eigen_face = vector.reshape(img_size)
        eigen_faces.append(eigen_face)

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Average", cv2.WINDOW_NORMAL)

    output_img = cv2.resize(mean_face, (0, 0), fx = 2, fy = 2)
    
    cv2.imshow("Result", output_img)
    cv2.imshow("Average", mean_face)
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)

    slider_values = []

    for i in range(NUM_EIGEN_FACES):
        slider_values.append(int(MAX_SLIDER_VALUE / 2))
        cv2.createTrackbar("Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE / 2), MAX_SLIDER_VALUE, generate_new_face)
    cv2.setMouseCallback("Average", reset_sliders)

    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()
