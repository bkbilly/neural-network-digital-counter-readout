import os
from pathlib import Path
from shutil import copyfile


inputdir = 'digits'
files = Path(inputdir).rglob('*.jpg')
x_data = []
y_data = []

for aktfile in files:
    base = os.path.basename(aktfile)
    target = base[0:1]
    category = int(target)
    dst = f"out/{category}/{base}"
    print(category, aktfile, dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    copyfile(aktfile, dst)
