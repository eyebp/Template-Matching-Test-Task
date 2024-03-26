#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from typing import Tuple
import os
import csv
import argparse

# In[ ]:

parser = argparse.ArgumentParser(
    description='Calculate number of wall cabinets on the architectural drawing.')
parser.add_argument("drawing_path", type=str, help="path to architectural drawing")
args = parser.parse_args()

img = cv.imread(args.drawing_path, cv.IMREAD_GRAYSCALE)
# In[ ]:

COEF = 2.1  # Approx scale of templates vs blueprint
blur, sigx = 3, 2  # Gauss blur params
# In[ ]:

# The below block is used to extract features from templates 
# and save these cropped templates to a new directory.
"""
# Read and display template image, and crop exclusive features.
template_n = 4
template = cv.imread(os.path.join('templates', f'{template_n}.png'), cv.IMREAD_GRAYSCALE)
plt.imshow(template)
plt.grid(True);

template = template[98:130, 75:115]

# Apply transformations and save template to a new image.
template = cv.resize(template, (int(round(template.shape[1] / COEF)), int(round(template.shape[0] / COEF))))
# Gauss blur is applied to templates and to the blueprint to improve matching.
template = cv.GaussianBlur(template, (blur, blur), sigx)
plt.imsave(os.path.join('templates_adj', f't{template_n}.png'), template)
"""

# In[ ]:

# Apply blur to the entire blueprint
img = cv.GaussianBlur(img, (blur, blur), sigx)

# In[ ]:

def find_pattern(img: np.ndarray, pat: np.ndarray, th: int = .85) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Execute template matching and return coordinates of matches.

    :param img: Image to search templates for.
    :param pat: Pattern (template) to search.
    :param th: Threshold.
    :returns: Coordinates of matches.
    '''
    img_ = img.copy()
    res = cv.matchTemplate(img_, pat, cv.TM_CCOEFF_NORMED)
    return np.where( res >= th)

# In[ ]:

# Iterate over cropped templates and fill in the list of matched coordinates
# while adding rectangles on the image.
dir_name = 'templates_adj'
locs = []
img_original = img.copy()
for template_path in os.listdir(dir_name):
    template = cv.imread(os.path.join(dir_name, template_path), cv.IMREAD_GRAYSCALE)
    w, h = template.shape[::-1]
    loc = find_pattern(img_original, template)
    for pt in zip(*loc[::-1]):
        locs.append(pt)
        cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), 0, 1)

# In[ ]:

# Use DBSCAN for deduplication
points = np.array(locs)
dbscan = DBSCAN(eps=20, min_samples=1)
labels = dbscan.fit_predict(points)

# In[ ]:

# Iterate over matched locations to filter out tall cabinets.
to_drop = set()
for i, pt1 in enumerate(locs):
    if labels[i] in to_drop:
        continue
    for j, pt2 in enumerate(locs):
        if pt1 == pt2 or labels[j] in to_drop:
            continue
        
        # Tall twin-compartment cabinet detection rule
        if abs(pt1[0] - pt2[0]) < 5 and abs(pt1[1] - pt2[1]) < 150 and labels[i] != labels[j]:
            to_drop |= {labels[i], labels[j]}

# In[ ]:

# Drop labels associated with tall cabinets.
unique_cabs = []
for label in np.unique(labels):
    if label in to_drop:
        continue
    # Merge duplicate matches into single location by label.
    unique_cabs.append(list(np.median(points[np.where(labels == label)[0]], axis=0).round()))
unique_cabs = np.array(unique_cabs, dtype=np.int16)

# In[ ]:

# Save output image and csv.
plt.style.use('dark_background')
plt.figure(figsize=(20, 16))
plt.scatter(unique_cabs[:, 0], unique_cabs[:, 1], 50, 'red', '*')
plt.imshow(img_original, cmap='gray')
plt.tight_layout(pad=0)
plt.savefig('out.png')

with open('out.csv', 'w') as fw:
    writer = csv.writer(fw)
    writer.writerow(('X', 'Y'))
    for xy in unique_cabs:
        writer.writerow(xy)
