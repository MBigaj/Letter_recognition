import matplotlib.pyplot as plt

# IMAGE PLOTTING FUNCTIONS FOR CLARITY

def plot_image(img):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    # plt.axis('off')
    plt.show()


def plot_two_images(img_1, img_2):
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img_1, cmap='gray')
    ax[1].imshow(img_2, cmap='gray')
    # ax[0].axis('off')
    # ax[1].axis('off')
    plt.show()