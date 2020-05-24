from keras.utils import to_categorical
import cv2
import numpy as np
import matplotlib.pyplot as plt 


def remove_9(x, y):
    mask = y != 9
    return x[mask], y[mask]

def replicate(X, left=13, right=13, up=21, down=21):
    """
    Given a batch of images (of shape (num samples, image height, image width, num channels)),
    replicates all the images' border on the 4 sides.
    """
    # left
    X = np.concatenate((np.tile(X[:, :, :1], (1, 1, left, 1)), X), axis=2)

    # right
    X = np.concatenate((X, np.tile(X[:, :, -1:], (1, 1, right, 1))), axis=2)

    # up
    X = np.concatenate((np.tile(X[:, :1, :], (1, up, 1, 1)), X), axis=1)

    # down
    X = np.concatenate((X, np.tile(X[:, -1:, :], (1, down, 1, 1))), axis=1)

    return X

# def replicate_img(x, left=13, right=13, up=21, down=21):
#     """
#     Given a batch of images (of shape (num samples, image height, image width, num channels)),
#     replicates all the images' border on the 4 sides.
#     """
#     # left
#     x = np.concatenate((np.tile(x[:, :1], (1, left, 1)), x), axis=1)

#     # right
#     x = np.concatenate((x, np.tile(x[:, -1:], (1, right, 1))), axis=1)

#     # up
#     x = np.concatenate((np.tile(x[:1, :], (up, 1, 1)), x), axis=0)

#     # down
#     x = np.concatenate((x, np.tile(x[-1:, :], (down, 1, 1))), axis=0)

#     return x

# def load_mnist_data():
    

def preprocessing_mnist_data(train_digits, train_labels, test_digits, test_labels, num_classes=9):
    image_height = train_digits.shape[1]  
    image_width = train_digits.shape[2]
    num_channels = 1  # we have grayscale images

    # re-shape the images data
    train_data = np.reshape(train_digits, (train_digits.shape[0], image_height, image_width, num_channels))
    test_data = np.reshape(test_digits, (test_digits.shape[0],image_height, image_width, num_channels))
    
    # threshold
    train_data[train_data > 0] = 255

#     train_data = train_digits.reshape(-1, 784)
#     test_data = test_digits.reshape(-1, 784)

    # re-scale the image data to values between (0.0,1.0]
#     train_data = train_data.astype('float32') / 255.
#     test_data = test_data.astype('float32') / 255.

    # add padding to have image similar to our images
    train_data = normalize(train_data, batch=True)
    test_data = normalize(test_data, batch=True)
   

    # one-hot encode the labels - we have 9 output classes (esclude 9!)
    # so 3 -> [0 0 0 1 0 0 0 0 0 0], 5 -> [0 0 0 0 0 1 0 0 0 0] & so on
    train_labels_cat = to_categorical(train_labels, num_classes)
    test_labels_cat = to_categorical(test_labels, num_classes)
    return train_data, train_labels_cat, test_data, test_labels_cat

def preprocessing_our_data(objects, to_exclude=False):
    toRecognize = []
    
    # esclude images with index: 2, 3, 6, 9
    idx = 0

    for im in objects:
        im = im * 255.
        resized_digit = cv2.resize(im, (26,26))

        padded_digit = np.pad(resized_digit, ((1,1),(1,1)), "constant", constant_values=0)
        im[im > 0] = 255
        if(not to_exclude):
            toRecognize.append(normalize(padded_digit))
        elif(to_exclude and ((idx in [2, 3, 6, 9]) == False)):
            toRecognize.append(normalize(padded_digit))
#         else: #check we excluded digits
#             plt.imshow(padded_digit)
#             plt.show()
        idx = idx + 1

    toRecognize = np.stack(toRecognize)
    return toRecognize

def normalize(x, batch=False):
    if batch:
        return (x - x.mean(axis=(1, 2)).reshape(x.shape[0], 1, 1, 1))  / x.std(axis=(1, 2)).reshape(x.shape[0], 1, 1, 1)
    return np.array((x - x.mean()) / x.std())