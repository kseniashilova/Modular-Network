from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt

def get_any_image(path, draw=False, scaling=0.1):
    img = Image.open(path)
    img = img.convert("L")
    width, height = np.array(img).shape[:2]
    resized_image = img.resize((int(scaling*height), int(scaling*width)), Image.LANCZOS)
    if draw:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title(f'original, shape: {np.array(img).shape}')

        # Plot the second image
        axes[1].imshow(resized_image, cmap='gray')
        axes[1].set_title(f'resized, shape: {np.array(resized_image).shape}')

        for ax in axes:
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        plt.tight_layout()
        plt.show()

    resized_image = np.array(resized_image)
    print('Resized image shape: ', resized_image.shape)
    return resized_image



def get_one_image(draw=True):
    # Assuming you have the necessary IDs to query the image data
    boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
    data_set = boc.get_ophys_experiment_data(ophys_experiment_id=501498760)

    image_data = data_set.get_stimulus_template('naturalscene.png')
    image = image_data[0]  # a numpy array of pixel values

    if draw:
        plt.imshow(image, cmap='gray')
        plt.colorbar()
        plt.show()

    return image
