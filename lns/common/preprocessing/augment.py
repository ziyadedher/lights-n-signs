import cv2 as cv
import numpy as np
from numpy.random import randn 
from random import randint as rand

# Hyperparameters
WARP = 0.05
NOISE_SIGMA = 5
BRIGHTNESS_SIGMA = 25
CONTRAST_MEAN = 1
CONTRAST_SIGMA = 0.1
HUE_MEAN = 1
HUE_SIGMA = 0.5
SATURATION_MEAN = 1
SATURATION_SIGMA = 0.2
HEIGHT_LOW = 50
HEIGHT_HIGH = 300

def augment(sign_image: np.ndarray, background_image: str, output_image: str):
    """
    Create an augmented product of sign and background image, and save it into the 
    output image. Also return the new bounding box coordinates.
    """

    def get_sign_mask(sign_img: np.ndarray):
        """
        Generate a mask around the sign to avoid discontinuity in
        the composite image.
        """
        # Load image and detect edges
        gray = cv.cvtColor(sign_img, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        canny = cv.Canny(blur, 50, 150)
        # Find biggest contour's hull
        cnts = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[1]
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)
        if len(cnts) == 0: return None
        cnts = [cv.convexHull(cnts[0])]
        # Draw mask of the enclosing shape
        mask = np.zeros_like(sign_img)
        cv.drawContours(mask, cnts, 0, (255, 255, 255), -1)
        # Return mask
        return mask


    def random_resize(img: np.ndarray, mask: np.ndarray, h_low: int, h_high: int):
        """
        Randomly resize image and mask proportionately between height of
        h_low and h_high.
        """
        # Select random height
        height = rand(h_low, h_high)
        newsize = (height, int(mask.shape[0]*height/mask.shape[1]))
        img = cv.resize(img, newsize)
        mask = cv.resize(mask, newsize)
        return img, mask

        
    def perspective_warp(img: np.ndarray, mask: np.ndarray, p: float):
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

        # Transform and return
        img = cv.warpPerspective(img, M, (img.shape[1], img.shape[0]))
        mask = cv.warpPerspective(mask, M, (img.shape[1], img.shape[0]))
        return img, mask


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

    def join_images(bg: np.ndarray, fg: np.ndarray, mask: np.ndarray):
        """
        Create composite image by joining background and foreground wrt the mask.
        """
        # Fetch region of interest
        x, y = (rand(0, bg.shape[1]), rand(0, bg.shape[0]))
        h, w = mask.shape[0], mask.shape[1]
        roi = bg[y:y+h, x:x+w]

        # If size matching successful
        if fg.shape == roi.shape:
            # Create composite image
            composite = np.uint8(fg * mask + (1-mask) * roi)
            output = np.copy(bg)
            output[y:y+h, x:x+w] = composite
            # Return image and bounding box
            bounding_box = (x, y, w, h) # x, y, w, h
            return output, bounding_box
        return None

    def pipeline(bg: np.ndarray, img: np.ndarray):
        """
        Apply the whole pipeline of creating a composite of background and foreground.
        """
        # Generate randomness
        brightness = randn() * BRIGHTNESS_SIGMA
        contrast = randn() * CONTRAST_SIGMA + CONTRAST_MEAN
        hue = randn() * HUE_SIGMA + HUE_MEAN
        saturation = randn() * SATURATION_SIGMA + SATURATION_MEAN
        # Load images
        mask = get_sign_mask(img)
        if mask is None: return None, None
        # Apply transformations
        i, m = perspective_warp(img, mask, WARP)
        i = add_noise(i, NOISE_SIGMA)
        i = color_augmentation(i, brightness, contrast, hue, saturation)
        i, m = random_resize(i, m, HEIGHT_LOW, HEIGHT_HIGH)
        # Join images
        joined = None
        while joined is None: joined = join_images(bg, i, m/255)
        return joined

    background_image = cv.imread(background_image)
    output, box = pipeline(background_image, sign_image)
    if output is None: return None

    cv.imwrite(output_image, output)

    (x, y, w, h) = box
    return {
        "x_min": x,
        "y_min": y,
        "x_max": x + w,
        "y_max": y + h
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