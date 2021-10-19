import glob
import os
from PIL import Image

Input_dir = 'ziffer_sortiert_raw'
Output_dir = 'ziffer_sortiert_resize'

target_size_x = 20
target_size_y = 32

files = glob.glob(Output_dir + '/*.jpg')
i = 0
for f in files:
    os.remove(f)
    i = i + 1
print(str(i) + " files have been deleted.")


files = glob.glob(Input_dir + '/*.jpg')
for aktfile in files:
    print(aktfile)
    test_image = Image.open(aktfile)
    test_image = test_image.resize(
        (target_size_x, target_size_y),
        Image.NEAREST)
    base = os.path.basename(aktfile)
    save_name = Output_dir + '/' + base
    test_image.save(save_name, "JPEG")
