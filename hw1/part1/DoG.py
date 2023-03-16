import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        gaussian_images.append(image)

        for octave in range(self.num_octaves):
            base_image = gaussian_images[-1]
            for i in range(self.num_DoG_images_per_octave):
                gaussian_images.append(cv2.GaussianBlur(base_image, (0, 0), self.sigma**(i+1)))

            if octave != self.num_octaves - 1:
                half_size = (int(base_image.shape[1]/2), int(base_image.shape[0]/2))
                gaussian_images.append(cv2.resize(gaussian_images[-1], half_size, interpolation=cv2.INTER_NEAREST))


        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for octave in range(self.num_octaves):
            base_index = octave * self.num_guassian_images_per_octave
            for i in range(self.num_DoG_images_per_octave):
                dog_images.append(cv2.subtract(gaussian_images[base_index + i + 1], gaussian_images[base_index + i]))

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = np.empty((0,2), int)

        for octave in range(self.num_octaves):
            base_index = octave * self.num_DoG_images_per_octave
            for index in range(self.num_DoG_images_per_octave - 2):
                image_index = index + 1 + base_index
                image_size_l = dog_images[image_index].shape[0]
                image_size_w = dog_images[image_index].shape[1]
                for i_x in range(image_size_w - 2):
                    x = i_x + 1
                    for i_y in range(image_size_l - 2):
                        y = i_y + 1
                        if abs(dog_images[image_index][y][x]) < 3:
                            continue

                        maximum = True
                        minimum = True
                        for layer in range(3):
                            for j in range(3):
                                for k in range(3):
                                    if layer == 1 and j == 1 and k == 1:
                                        continue
                                    if dog_images[image_index][y][x] < dog_images[image_index - 1 + layer][y - 1 + j][x - 1 + k]:
                                        maximum = False
                                    if not maximum:
                                        break
                                if not maximum:
                                    break
                            if not maximum:
                                break
                        
                        if not maximum:
                            for layer in range(3):
                                for j in range(3):
                                    for k in range(3):
                                        if layer == 1 and j == 1 and k == 1:
                                            continue
                                        if dog_images[image_index][y][x] > dog_images[image_index - 1 + layer][y - 1 + j][x - 1 + k]:
                                            minimum = False
                                        if not minimum:
                                            break
                                    if not minimum:
                                        break
                                if not minimum:
                                    break
                        
                        if maximum or minimum:
                            keypoints = np.append(keypoints, np.array([[y * (octave + 1), x * (octave + 1)]]), axis=0)
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
