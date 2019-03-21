import cv2 as cv
import numpy as np
from numpy.random import randn 
from random import randint as rand

# Hyperparameters
AREA_THRESHOLD=0.5
WARP = 0.05
NOISE_SIGMA = 0
BRIGHTNESS_SIGMA = 15
CONTRAST_MEAN = 1
CONTRAST_SIGMA = 0.1
HUE_MEAN = 1
HUE_SIGMA = 0.1
SATURATION_MEAN = 1
SATURATION_SIGMA = 0.1

def augment(image: str, box, output_image: str):
    """
    Create an augmented product of sign and background image, and save it into the 
    output image. Also return the new bounding box coordinates.
    """

    def perspective_warp(img: np.ndarray, box: np.ndarray, p: float):
        """
        Warp the image dimensions with a ratio of p on either dimensions.
        """

        # Randomly generate warping anchors
        x1 = rand(0, int(img.shape[1] * p))
        x2 = rand(0, int(img.shape[1] * p))
        y1 = rand(0, int(img.shape[0] * p))
        y2 = rand(0, int(img.shape[0] * p))

        # Generate transformation matrix
        dst = [[0,0], [0, img.shape[1]], [img.shape[0], img.shape[1]], [img.shape[0], 0]]
        rect = [[y1, x1], [y2, img.shape[1]-x2], [img.shape[0]-y2, img.shape[1]-x2], [img.shape[0]-y1, x2]]
        dst = np.array(dst, np.float32)
        rect = np.array(rect, np.float32)
        M = cv.getPerspectiveTransform(rect, dst)
        new_box = np.matmul(box, M)

        # Transform and return
        img = cv.warpPerspective(img, M, (img.shape[1], img.shape[0]))
        return img, new_box


    def add_noise(img: np.ndarray, sigma: float):
        """
        Add noise with std. deviation of sigma to the image.
        """
        # Add noise
        return np.uint8(img + np.random.randn(img.shape[0], img.shape[1], img.shape[2])*sigma)


    def color_augmentation(img: np.ndarray, brightness: float, contrast: float, hue: float, saturation: float):
        """
        Augment the sign color by tweaking brightness, contrast, hue and saturation.
        """
        # Brightness/contrast
        img_contrast = img * contrast
        img_bright = img_contrast + brightness
        normalized = np.uint8(np.clip(img_bright, 0, 255))
        # Hue/saturation
        img_hsv = cv.cvtColor(normalized, cv.COLOR_BGR2HSV).astype('float32')
        img_hsv[:,:,0] *= hue
        img_hsv[:,:,1] *= saturation
        normalized = np.uint8(np.clip(img_hsv, 0, 255))
        # Convert back to rgb
        rgb = cv.cvtColor(normalized, cv.COLOR_HSV2BGR)
        return rgb

    def pipeline(img: np.ndarray, box):
        """
        Apply the whole pipeline of creating a composite of background and foreground.
        """
        # Generate randomness
        brightness = randn() * BRIGHTNESS_SIGMA
        contrast = randn() * CONTRAST_SIGMA + CONTRAST_MEAN
        hue = randn() * HUE_SIGMA + HUE_MEAN
        saturation = randn() * SATURATION_SIGMA + SATURATION_MEAN
        # Apply transformations
        i, b = perspective_warp(img, box, WARP)
        i = add_noise(i, NOISE_SIGMA)
        i = color_augmentation(i, brightness, contrast, hue, saturation)
        # Join images
        return i, b

    output, box = pipeline(image, box)

    cv.imwrite(output_image, output)

    (x1, y1) = box[0, :2]
    (x2, y2) = box[1, :2]
    return {
        "x_min": x1,
        "y_min": y1,
        "x_max": x2,
        "y_max": y2
    }


# from os import listdir as ls, path
# def process(images_folder, labels_folder, bg_folder, images_out, labels_out):
#     for bg_name in ls(bg_folder):
#         bg_path = path.join(bg_folder, bg_name)
#         bg = cv.imread(bg_path)
#         for img_name in ls(images_folder):
#             identifier = img_name.split('.')[0]
#             img_path = path.join(images_folder, img_name)
#             label_path = path.join(labels_folder, identifier + '.txt')

#             image = cv.imread(img_path)
#             annotations = open(label_path, 'r').readlines()
#             for annotation in annotations:
#                 label = annotation.split(' ')[0]
#                 x1, y1, x2, y2 = list(map(int, annotation.split(' ')[4:8]))
#                 img = image[y1:y2, x1:x2]

#                 output, box = pipeline(bg, img)
#                 x, y, w, h = box
#                 new_label = f'{label} 0 0 0 {x} {y} {x+w} {y+h} 0 0 0 0 0 0 0 0'
                
#                 new_label_path = path.join(labels_out, identifier + '.txt')
#                 new_image_path = path.join(images_out, identifier + '.' + bg_name + '.png')
#                 open(new_label_path, 'w').write(new_label)
#                 cv.imwrite(new_image_path, output)



# for image in images:
#     train_image = cv.imread(image)
#     i = 1
#     for bg in backgrounds:
#         for ant in annotations[image]:
#             # Setup new path
#             new_path = image + f".aug{i}.png"
#             # Extract ROI
#             x1, y1, x2, y2 = ant['x_min'], ant['y_min'], ant['x_max'], ant['y_max']
#             sign_image = train_image[y1:y2, x1:x2]
#             # Augment and modify class
#             a = augment(sign_image, bg, new_path)
#             a['class'] = ant['class']
#             # Add annotation
#             annotations[new_path] = [a]
#             # Save image path
#             images.append(new_path)
#             i += 1