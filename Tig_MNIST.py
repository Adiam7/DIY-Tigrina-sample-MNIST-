from tkinter import image_types
from PIL import Image
from matplotlib.animation import ImageMagickFileWriter
import matplotlib.pyplot as plt
import numpy as np

input_folder = '/Users/yonatandawit/Documents/MIT ReAct/MIT edx/Project/'

filename = input_folder+'image_400.png'
image = Image.open(filename)
image = image.resize((630, 300))

def show(img, figsize=(8, 4), title="Geez Tig Alphabet Sample"):
    plt.figure(figsize=figsize)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.show()

show(image)

from PIL import ImageEnhance

bw_image = image.convert(mode='L') #L is 8-bit black-and-white image mode
show(bw_image, figsize=(9, 9))
bw_image = ImageEnhance.Contrast(bw_image).enhance(1.5)
show(bw_image, figsize=(9, 9))

SIZE = 30
samples = [] # array to store cut images
for alphabet, y in enumerate(range(0, bw_image.height, SIZE)):
    #print('Cutting height :',alphabet+1, y)
    cuts=[]
    for x in range(0, bw_image.width, SIZE+10):
        cut = bw_image.crop(box=(x+3, y+3, x+SIZE-3, y+SIZE-4))
        cuts.append(cut)
    samples.append(cuts)
print(f'Cut {len(samples)*len(samples[0])} images total.')


f = plt.figure(figsize=(9,2))
ax = f.subplots(1, 6)
for i, n in enumerate(np.random.randint(0, len(samples), size=6)):
    m = (np.random.randint(0, len(samples[n])))
    ax[i].imshow(samples[n][m])
    ax[i].set_title(f'Fidel: [{n}]')
plt.show()



sample = samples[2][0]
show(sample, figsize=(2, 2))

from PIL import ImageOps
import matplotlib.patches as patches

# Inver sample, get bbox and display all that stuff.
inv_sample = ImageOps.invert(sample)
bbox = inv_sample.getbbox()

fig = plt.figure(figsize=(2, 2))
ax = fig.add_axes([0,0,1,1])

ax.imshow(inv_sample)
rect = patches.Rectangle(
    (bbox[0], bbox[3]), bbox[2]-bbox[0], -bbox[3]+bbox[1]-1,
    fill=False, alpha=1, edgecolor='w')
ax.add_patch(rect)
plt.show()

crop = inv_sample.crop(bbox)
show(crop, title='Image cropped to bounding box')

#resize back
new_size = 28
delta_w = new_size - crop.size[0]
delta_h = new_size - crop.size[1]
padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
new_im = ImageOps.expand(crop, padding)
show(new_im, title='Resized and centered to 28x28')

def resize_and_center(sample, new_size=28):
    inv_sample = ImageOps.invert(sample)
    bbox = inv_sample.getbbox()
    crop = inv_sample.crop(bbox)
    delta_w = new_size - crop.size[0]
    delta_h = new_size - crop.size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    return ImageOps.expand(crop, padding)

resized_samples = []
for row in samples:
    resized_samples.append([resize_and_center(sample) for sample in row])


f = plt.figure(figsize=(8,2))
ax = f.subplots(1, 6)
for i, n in enumerate(np.random.randint(0, len(resized_samples), size=6)):
    m = (np.random.randint(0, len(resized_samples[n])))
    ax[i].imshow(resized_samples[n][m])
    ax[i].set_title(f'Fidel r: [{n}]')
plt.show()

preview = Image.new('L', (len(samples[0])*new_size, len(samples)*new_size))


x = 0
y = 0
for row in resized_samples:
    for sample in row:
        preview.paste(sample, (x, y))
        x += new_size 
    y+=new_size 
    x = 0

show(preview, figsize=(9,9), title='Processed images')
preview.save('preview.png')


"""
Save the result in numpy binary format

Image.getdata() method returns image bytes, which we put into numpy array.
"""

binary_samples = np.array([[sample.getdata() for sample in row] for row in resized_samples])
binary_samples = binary_samples.reshape(len(resized_samples)*len(resized_samples[0]), 28, 28)

show(binary_samples[128], figsize=(3,3))

# As we had 16 columns and 10 rows in the original picture, 
# now we generate a target array with corresponging alphabets


classes = np.array([[ 'በ'],  ['ቡ'],  ['ቢ'],  ['ባ'], ['ቤ'], ['ብ'], ['ቦ'], ['ረ'],['ሩ'], ['ሪ'], ['ራ'], ['ሬ'], ['ር'], ['ሮ'],  ['ቀ'],  ['ቐ'],
                    [ 'ሰ'],  ['ሱ'],  ['ሲ'],  ['ሳ'], ['ሴ'],  ['ስ'],  ['ሶ'],  ['ፈ'],  ['ፉ'],  ['ፊ'], ['ፋ'], ['ፌ'], ['ፍ'], ['ፎ'], ['ቁ'], ['ቑ'],
                    [ 'ሸ'],  ['ሹ'],  ['ሺ'],  ['ሻ'], ['ሼ'],  ['ሽ'],  ['ሾ'],  ['ጠ'],  ['ጡ'],  ['ጢ'], ['ጣ'], ['ጤ'],  ['ጥ'],  ['ጦ'],  ['ቂ'],  ['ቒ'],
                    [ 'ለ'],  ['ሉ'],  ['ሊ'],  ['ላ'], ['ሌ'],  ['ል'],  ['ሎ'],  ['ጨ'],  ['ጩ'],  ['ጪ'], ['ጫ'], ['ጬ'],  ['ጭ'],  ['ጮ'],  ['ቃ'],  ['ቓ'],
                    [ 'አ'],  ['ኡ'],  ['ኢ'],  ['ኣ'], ['ኤ'],  ['እ'],  ['ኦ'],  ['ሐ'],  ['ሑ'],  ['ሒ'], ['ሓ'],  ['ሔ'],  ['ሕ'],  ['ሖ'],  ['ቄ'],  ['ቔ'],
                    [ 'ከ'],  ['ኩ'],  ['ኪ'],  ['ካ'], ['ኬ'],  ['ክ'],  ['ኮ'],  ['መ'],  ['ሙ'],  ['ሚ'], ['ማ'],  ['ሜ'],  ['ም'],  ['ሞ'],  ['ቅ'],  ['ቕ'],
                    [ 'ኸ'],  ['ኹ'], ['ኺ'],  ['ኻ'], ['ኼ'],  ['ኽ'],  ['ኾ'],  ['ወ'],  ['ዉ'], ['ዊ'], ['ዋ'],  ['ዌ'],  ['ው'],  ['ዎ'],  ['ቆ'],  ['ቖ'],
                    [ 'ነ'],  ['ኑ'],  ['ኒ'],  ['ና'], ['ኔ'], ['ን'], ['ኖ'],  ['ዐ'],  ['ዑ'],  ['ዒ'], ['ዓ'],  ['ዔ'],  ['ዕ'],  ['ዖ'],  ['ጸ'],  ['ጰ'],
                    [ 'ኘ'],  ['ኙ'], ['ኚ'],  ['ኛ'], ['ኜ'],  ['ኝ'],  ['ኞ'],  ['የ'],  ['ዩ'],  ['ዪ'], ['ያ'],  ['ዬ'],  ['ይ'],  ['ዮ'],  ['ደ'],  ['ቨ'],
                    [ 'ተ'],  ['ቱ'], ['ቲ'], ['ታ'], ['ቴ'], ['ት'], ['ቶ'],  ['ገ'],  ['ጉ'],  ['ጊ'], ['ጋ'],  ['ጌ'],  ['ግ'],  ['ጎ'],  ['ጀ'],  ['ሆ']]).reshape(-1)


print(f'X shape: {binary_samples.shape}')
print(f'y shape: {classes.shape}')

xfile = 'Alphabet_x_test.npy'
yfile = 'Alphabet_y_test.npy'
np.save(xfile, binary_samples)
np.save(yfile, classes)


x_test = np.load(xfile)
y_test = np.load(yfile)
x_test.shape, y_test.shape


for i in np.random.randint(x_test.shape[0], size=6):
    show(x_test[i], title=f'Fidel [{y_test[i]}]', figsize=(3,3))