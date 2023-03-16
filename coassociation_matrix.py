import FixaTons
import plotly.graph_objects as go
import plotly.express as px
from scipy.ndimage.filters import gaussian_filter
from scipy import signal
import numpy as np
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import skimage.io as sio
import cv2 as cv
from PIL import Image, ImageChops
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score



DATASET_NAME = 'MIT1003'
STIMULUS_SET = [#'i05june05_static_street_boston_p1010785.jpeg' # 2 cars
                 # 'i1032358.jpeg',
                 # 'i202396633.jpeg',
                 # 'i2244445589.jpeg', # Girl and Abe Lincoln
                  # 'i1040585936.jpeg' # Plate of food and guy
                  'i2289665173.jpeg' # Golden gate bridge
                # 'i666418509.jpeg' # tiger
                ]


coassociation_matrix = np.zeros([15,15])
for stimulus in STIMULUS_SET:
    image = FixaTons.get.stimulus(DATASET_NAME, stimulus)
    subjects = FixaTons.info.subjects(DATASET_NAME, stimulus)
    distance_matrix = np.zeros([15,15])
    image_width, image_height = FixaTons.get.stimulus_size(DATASET_NAME, stimulus)
    for SUBJECT_NAME_1 in subjects:
        for SUBJECT_NAME_2 in subjects:
            result = FixaTons.metrics.string_based_time_delay_embedding_distance(
                                    FixaTons.get.scanpath(DATASET_NAME, stimulus, SUBJECT_NAME_1),
                                    FixaTons.get.scanpath(DATASET_NAME, stimulus, SUBJECT_NAME_2),
                                    image_width, image_height, k = 3, distance_mode='Hausdorff')
            print(stimulus, SUBJECT_NAME_1, SUBJECT_NAME_2, result)
            distance_matrix[subjects.index(SUBJECT_NAME_1)][subjects.index(SUBJECT_NAME_2)] = result
            # if result < 100: distance_matrix[subjects.index(SUBJECT_NAME_1)][subjects.index(SUBJECT_NAME_2)] = 1
            # elif result < 200: distance_matrix[subjects.index(SUBJECT_NAME_1)][subjects.index(SUBJECT_NAME_2)] = 2
            # elif result < 300: distance_matrix[subjects.index(SUBJECT_NAME_1)][subjects.index(SUBJECT_NAME_2)] = 3
            # else: distance_matrix[subjects.index(SUBJECT_NAME_1)][subjects.index(SUBJECT_NAME_2)] = 4


    # for SUBJECT_NAME_1 in subjects:
    #     for SUBJECT_NAME_2 in subjects:
    #         result = FixaTons.metrics.time_delay_embedding_distance(
    #                                 FixaTons.get.scanpath(DATASET_NAME, stimulus, SUBJECT_NAME_1),
    #                                 FixaTons.get.scanpath(DATASET_NAME, stimulus, SUBJECT_NAME_2), 3,
    #                                 'Mean')
    #         print(stimulus, SUBJECT_NAME_1, SUBJECT_NAME_2, result)
    #         distance_matrix[subjects.index(SUBJECT_NAME_1)][subjects.index(SUBJECT_NAME_2)] = result
    #         # if result < 100: distance_matrix[subjects.index(SUBJECT_NAME_1)][subjects.index(SUBJECT_NAME_2)] = 1
    #         # elif result < 200: distance_matrix[subjects.index(SUBJECT_NAME_1)][subjects.index(SUBJECT_NAME_2)] = 2
    #         # elif result < 300: distance_matrix[subjects.index(SUBJECT_NAME_1)][subjects.index(SUBJECT_NAME_2)] = 3
    #         # else: distance_matrix[subjects.index(SUBJECT_NAME_1)][subjects.index(SUBJECT_NAME_2)] = 4

    # for SUBJECT_NAME_1 in subjects:
    #     for SUBJECT_NAME_2 in subjects:
    #         result = FixaTons.metrics.scaled_time_delay_embedding_distance(
    #                                 FixaTons.get.scanpath(DATASET_NAME, stimulus, SUBJECT_NAME_1),
    #                                 FixaTons.get.scanpath(DATASET_NAME, stimulus, SUBJECT_NAME_2), image)
    #         # if result > 0.5:
    #         distance_matrix[subjects.index(SUBJECT_NAME_1)][subjects.index(SUBJECT_NAME_2)] = result

    # fig2 = px.imshow(distance_matrix, text_auto=True, x=subjects, y=subjects)
    # fig2.show()

    #Feature scaling
    distance_matrix = np.nan_to_num(distance_matrix)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(distance_matrix)

    k = 4

    #K-means
    kmeans = KMeans(init="random",n_clusters=k,n_init=10,max_iter=300) # k=4, as determined by the elbow method
    kmeans.fit(scaled_features)
    labels_kmeans = kmeans.labels_
    print(subjects)
    print(labels_kmeans)

    #Hierarchical clustering
    hierarchical_cluster = AgglomerativeClustering(n_clusters=k,  linkage='ward', affinity="euclidean")
    labels_hierarchical = hierarchical_cluster.fit_predict(scaled_features)
    print(labels_hierarchical)

    #Spectral clustering
    spectral_model_nn = SpectralClustering(n_clusters=k, affinity='nearest_neighbors')
    labels_spectral = spectral_model_nn.fit_predict(scaled_features)
    print(labels_spectral)

    # #DBSCAN clustering
    # dbscan = DBSCAN(eps = 0.001, min_samples = 2).fit(scaled_features)
    # labels_dbscan = dbscan.labels_
    # print(labels_dbscan)

    for subject_1 in range(len(subjects)):
        for subject_2 in range(len(subjects)):
            if (labels_kmeans[subject_1] == labels_kmeans[subject_2]):
                coassociation_matrix[subject_1][subject_2] += 1
            if (labels_hierarchical[subject_1] == labels_hierarchical[subject_2]):
                coassociation_matrix[subject_1][subject_2] += 1
            if (labels_spectral[subject_1] == labels_spectral[subject_2]):
                coassociation_matrix[subject_1][subject_2] += 1
            # if (labels_dbscan[subject_1] == labels_dbscan[subject_2]):
            #     coassociation_matrix[subject_1][subject_2] += 1

coassociation_matrix = coassociation_matrix/3  # Average of 3 clustering method results
fig1 = px.imshow(coassociation_matrix, x=subjects, y=subjects, color_continuous_scale='balance',  range_color=[0,1])
fig1.show()
#
# kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 300}
# sse = []
# for k in range(1, 11):
#     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
#     kmeans.fit(scaled_features)
#     sse.append(kmeans.inertia_)
#
# plt.style.use("fivethirtyeight")
# plt.plot(range(1, 11), sse)
# plt.xticks(range(1, 11))
# plt.xlabel("Number of Clusters")
# plt.ylabel("SSE")
# plt.show()
# kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
# print(kl.elbow)


# fig1 = px.imshow(distance_matrix, text_auto=True, x=subjects, y=subjects)
# fig1.show()

