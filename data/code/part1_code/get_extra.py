import random 
from captcha.image import ImageCaptcha
from tqdm.auto import tqdm 

# fonts_path = 'data/DejaVuMathTeXGyre.ttf'
fonts_path = 'data/ntailu.ttf'
image = ImageCaptcha(width=100,height=40,fonts=[fonts_path],font_sizes=[30])
total_pic = 5000
source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97 + 26)]
source += [chr(i) for i in range(65, 65 + 26)]
alphabet = ''.join(source)
weights = [5]*len(alphabet)
weights_alpha = ['0','i','l','1','o','v','u','I','O','U','V']
for i ,a in enumerate(alphabet):
    if a in weights_alpha:
        weights[i] = 100

import os 
os.makedirs(name='data/mkdata',exist_ok=True)
labels = []
paths  = []
for i in tqdm(range(total_pic)):
    label = ''.join(random.choices(alphabet,k=4,weights=weights))
    if label not in labels:
        labels.append(label)
        paths.append(f'data/mkdata/{label}.png')
        image.write(label, f'data/mkdata/{label}.png')
import pandas as pd 
pd.DataFrame({
    'image_path':paths,
    'label':labels,
    'fold':-1,
}).to_csv('data/mkdata.csv',index=False)