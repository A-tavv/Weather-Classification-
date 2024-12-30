train_data_dir = 'MWD/train/'
test_data_dir = 'MWD/test/'

labs = {'cloudy':0, 'rain':1, 'shine':2, 'sunrise':3}
image_size = (224, 224)


def load_data(main_root):
    imList = []
    lbList = []

    for lb_name in os.listdir(main_root):
        new_path=os.path.join(main_root,lb_name)

        if os.path.isdir(new_path): # if is a directory or not
            for filename in os.listdir(new_path):
                im_path = os.path.join(new_path,filename)
                img = cv2.imread(im_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    #resize images
                    resized_image=cv2.resize(img,(image_size))
                    #normalize images
                    norm_img = cv2.normalize(resized_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    imList.append(norm_img)
                    lbList.append(labs[lb_name])

    return imList, lbList

train_im, train_lb = load_data(train_data_dir)
test_im, test_lb = load_data(test_data_dir)

train_im = np.array(train_im)
test_im = np.array(test_im)
print("train data shape:", train_im.shape)
print("test data shape", test_im.shape)

from sklearn.utils import shuffle
train_im,train_lb = shuffle(train_im, train_lb, random_state=2)
test_im,test_lb = shuffle(test_im, test_lb, random_state=2)

train_lb = tf.keras.utils.to_categorical(train_lb,4)
test_lb = tf.keras.utils.to_categorical(test_lb,4)

import matplotlib.pyplot as plt
import numpy as np


indices = [0, 20, 30, 50, 100, 55, 1, 18, 33, 46, 170, 57, 5, 25, 35, 40, 200, 10]

fig, axs = plt.subplots(3, 6, figsize=(18, 10))

for i, ax in enumerate(axs.flat):
    index = indices[i]
    ax.imshow(test_im[index])
    ax.set_title(str(np.argmax(test_lb[index])))
    ax.set_axis_off()

plt.subplots_adjust(wspace=0.1, hspace=0)
plt.show()
