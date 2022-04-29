import numpy as np
from evaluations import extract_features
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
from utils import to_numpy
from scipy.cluster.hierarchy import fcluster, linkage


def Information_extract(training_txt):
    label_txt = training_txt
    file = open(label_txt)
    images_anon = file.readlines()

    images = []
    labels = []

    for img_anon in images_anon:
        [img, label] = img_anon.split(' ')
        images.append(img)
        labels.append(int(label))
    return images, labels


def Clustering(model,gallery_loader,data):
    gallery_feature, gallery_labels = extract_features(model, gallery_loader, print_freq=1e5, metric=None, pool_feature=False)
    X = [to_numpy(x) for x in gallery_feature]
    X = np.array(X)
    
    if data == 'car':
        n_cluster = 98
    elif data == 'cub':
        n_cluster = 100
    elif data == 'product':
        n_cluster = 565
    
    
    
    kmeans = KMeans(n_clusters=n_cluster, n_jobs=-1, random_state=0).fit(X)

    return kmeans.labels_

def generate_pseudo(imaging,pseudo_labels):
    images = imaging
    labels = pseudo_labels

    numSamples = len(images)
	
    with open('./train.txt', 'w') as f:
        for i in range(numSamples):
            line = images[i] + ' {}\n'.format(labels[i])
            f.writelines(line)


def hierarchical_clusteirng(model,gallery_loader,data,K,imaging):
    gallery_feature, gallery_labels = extract_features(model, gallery_loader, print_freq=1e5, metric=None, pool_feature=False)
    X = [to_numpy(x) for x in gallery_feature]
    X = np.array(X)
    
    Z = linkage(X, 'ward')

    if data == 'car':
        if K == 2:
            n_cluster = [98,49]
        elif K == 3:
            n_cluster = [98,50,10]
    elif data == 'cub':
        if K == 2:
            n_cluster = [100,50]
        elif K == 3:
            n_cluster = [100,50,10]
    elif data == 'product':
        if K == 2:
            n_cluster = [10000,1000]
        elif K == 3:
            n_cluster = [10000,1000,100]
            
    numSamples = len(imaging)       
    
    if K == 2:
        labels_1 = fcluster(Z, n_cluster[0], criterion='maxclust')
        labels_2 = fcluster(Z, n_cluster[1], criterion='maxclust')
        
        with open('./train.txt', 'w') as f:
            for i in range(numSamples):
                line = imaging[i] + ' {} {}\n'.format(labels_1[i],labels_2[i])
                f.writelines(line)
            
    elif K == 3:
        labels_1 = fcluster(Z, n_cluster[0], criterion='maxclust')
        labels_2 = fcluster(Z, n_cluster[1], criterion='maxclust')
        labels_3 = fcluster(Z, n_cluster[0], criterion='maxclust')

        with open('./train.txt', 'w') as f:
            for i in range(numSamples):
                line = imaging[i] + ' {} {} {}\n'.format(labels_1[i], labels_2[i], labels_3[i])
                f.writelines(line)
    
    