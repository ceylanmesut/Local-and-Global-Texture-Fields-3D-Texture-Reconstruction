import numpy as np
from skimage.measure import compare_ssim as ssim
import imageio
import os
from skimage.transform import resize # added by Mesut to resize real images into 224x224

def calculate_ssim_l1_given_paths(paths):
    file_list = os.listdir(paths[0])
    ssim_value = 0
    l1_value = 0
    for f in file_list:
        # assert(i[0] == i[1])
        fake = load_img(paths[0] + f)
        #print("FAKEEE IN SSIM FUNCTION", fake)
        real = load_img(paths[1] + f)
        #print("REALL IN SSIM FUNCTION", real)
        ssim_value += np.mean(
            ssim(fake, real, multichannel=True))
        l1_value += np.mean(abs(fake - real))
    
    ssim_value = ssim_value/float(len(file_list))
    l1_value = l1_value/float(len(file_list))

    return ssim_value, l1_value


def calculate_ssim_l1_given_tensor(images_fake, images_real):
    bs = images_fake.size(0)
    images_fake = images_fake.permute(0, 2, 3, 1).cpu().numpy()
    images_real = images_real.permute(0, 2, 3, 1).cpu().numpy()

    ssim_value = 0
    l1_value = 0
    for i in range(bs):
        # assert(i[0] == i[1])
        fake = images_fake[i]
        real = images_real[i]
        ssim_value += np.mean(
            ssim(fake, real, multichannel=True))
        l1_value += np.mean(abs(fake - real))
    ssim_value = ssim_value/float(bs)
    l1_value = l1_value/float(bs)

    return ssim_value, l1_value


def load_img(path):

    #print("PATHHHHH IN LOAD IMAGE FUNCTION IN SSIM_L1===============", path)
    part_of_path = str(path.split('/')[0:-1])
    #print("THOSE ARE PART OF PATH", part_of_path)
    #print("THIS IS THE PART THAT WE NEED TO CHECK========================================",part_of_path[-6:-2])
    img = imageio.imread(path)

    img = img.astype(np.float64) / 255

    # adding this line to process only real ones which has the shape of 256.
    if part_of_path[-6:-2]==str('real'):
        #print("=========================0WE ARE IN THE IMAGE RESIZING PART==================================")
        img=resize_image(img, 224,224) # Added by Mesut for resizing


    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    return img


def resize_image(img, height, width):
    if img.ndim == 3:
        h, w, c = img.shape
        img2 = np.zeros([height, width, c])
        for k in range(c):
            img2[:, :, k] = resize(img[:, :, k], (height, width), anti_aliasing=True)
    else:
        h, w = img.shape
        img2 = resize(img, (height, width), anti_aliasing=True)

    return img2
