#%matplotlib inline
import os
import random
import numpy as np
import json
import matplotlib.pyplot
import matplotlib.patheffects as PathEffects
import pickle
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from shutil import copy
from shutil import copytree, rmtree
from collections import defaultdict
import seaborn as sns

def load_images_from_folder(folder, num_classes, images_to_take, randomize):
    pic_width = 300
    pic_height = 300
    images = np.zeros((num_classes*images_to_take,1+(3*pic_width*pic_height)))
    labels = np.zeros(num_classes*images_to_take)
    i = 0
    j = 0
    label = 0
    for folder_name in os.listdir(folder):
        path = folder+"\\"+folder_name
        j = 0
        for filename in os.listdir(path):
            if j >= images_to_take:
                break
            else:
                j += 1
                if randomize:
                    #takes random pictures from folder
                    filename = random.choice(os.listdir(path))
                    #print(filename)
                img = Image.open(os.path.join(folder,folder_name,filename))
                #pic resize for uniform size
                img = img.resize((300, 300))
                img = np.asarray(img)
                #image reshape(heightxsize -> feature number = pixel number*number of colors(RGB))
                img = np.array(img).reshape(img.shape[0]*img.shape[1]*img.shape[2])
                if img is not None:
                    images[i,0] = i
                    images[i,1:(1+(3*pic_width*pic_height))] = img
                    labels[i] = label
                    i=i+1
        label+=1
    return images, labels
#plotting setup
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
#path to specify
folder="C:\AML\Food-101\Food-101\Test2"
images_to_take = 24
#load images from folder
images, labels = load_images_from_folder(folder, 4, images_to_take, True)
#Creating a dataframe
feat_cols = [ 'pixel'+str(i) for i in range(images.shape[1])]
df = pd.DataFrame(images,columns=feat_cols)
df['y'] = labels
df['label'] = df['y'].apply(lambda i: str(i))
#print(labels)

num_classes = 4

#PCA to reduce dimensionality for TSNE (recommended)
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(images)

#TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=8, n_iter=400)
tsne_results = tsne.fit_transform(X_reduced)
#Dataframe setup for second plot option
rndperm = np.random.permutation(df.shape[0])
df_subset = df.loc[rndperm[:images_to_take*4],:].copy()
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]

#Plots
#Plot type 1
def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts
fashion_scatter(tsne_results, labels)
plt.show()
'''
#Plot type 2
plt.figure(figsize=(16,10))
sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    hue=labels,
    palette=sns.color_palette("hls", 4),
    data=df_subset,
    legend="full",
    alpha=0.9
)
plt.show()
'''