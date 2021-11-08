import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import InterclusterDistance
from yellowbrick.cluster import KElbowVisualizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
from scipy import linalg
import matplotlib as mpl
import itertools
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm

from sklearn.utils import shuffle
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans


def load_grades_data():
    data = pd.read_csv('data/collegePlace.csv')
    dummy_gender = pd.get_dummies(data["Gender"])
    dummy_stream = pd.get_dummies(data["Stream"])
    data = pd.concat([data.drop(["Gender", "Stream"], axis=1),
                     dummy_gender, dummy_stream], axis=1)

    # reorder data
    data = data[['Age', 'Male', 'Female',
                 'Electronics And Communication',
                 'Computer Science', 'Information Technology',
                 'Mechanical', 'Electrical', "Civil",
                 "Internships", "CGPA", 'Hostel',
                 'HistoryOfBacklogs', 'PlacedOrNot']]

    X = data.drop('PlacedOrNot', axis=1)
    X = preprocessing.scale(X)
    X = pd.DataFrame(X, columns=data.columns[:-1])
    y = data['PlacedOrNot']
    return X, y


def load_heart_data():
    data = pd.read_csv('data/heart.csv')

    a = pd.get_dummies(data['cp'], prefix="cp")
    b = pd.get_dummies(data['thal'], prefix="thal")
    c = pd.get_dummies(data['slope'], prefix="slope")

    frames = [data, a, b, c]
    data = pd.concat(frames, axis=1)
    data = data.drop(columns=['cp', 'thal', 'slope'])
    X = data.drop('target', axis=1)
    X = preprocessing.scale(X)
    y = data['target']

    return X, y


def load_heart_data():
    data = pd.read_csv('data/heart.csv')

    a = pd.get_dummies(data['cp'], prefix="cp")
    b = pd.get_dummies(data['thal'], prefix="thal")
    c = pd.get_dummies(data['slope'], prefix="slope")

    frames = [data, a, b, c]
    data = pd.concat(frames, axis=1)
    data = data.drop(columns=['cp', 'thal', 'slope'])
    X = data.drop('target', axis=1)
    X = preprocessing.scale(X)
    y = data['target']

    return X, y


def load_wine_data():
    data = pd.read_csv("data/wine.csv")
    sc = StandardScaler()
    scaled_data = data.copy()
    scaled_data = sc.fit_transform(scaled_data)
    return scaled_data


def SelBest(arr: list, X: int) -> list:
    '''
    returns the set of X configurations with shorter distance
    '''
    dx = np.argsort(arr)[:X]
    return arr[dx]


def gmm_js(gmm_p, gmm_q, n_samples=10**5):
    # https://towardsdatascience.com/gaussian-mixture-model-clusterization-how-to-select-the-number-of-components-clusters-553bef45f6e4

    X = gmm_p.sample(n_samples)[0]
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y = gmm_q.sample(n_samples)[0]
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    return np.sqrt((log_p_X.mean() - (log_mix_X.mean() - np.log(2))
                    + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2)


def generate_kmeans_SV_ICD_plots(X, k):
    plot_nums = len(k)
    fig, axes = plt.subplots(plot_nums, 2, figsize=[25, 40])
    col_ = 0
    for i in k:
        kmeans = KMeans(n_clusters=i, algorithm="full")
        visualizer = SilhouetteVisualizer(
            kmeans, colors='yellowbrick', ax=axes[col_][0])
        visualizer.fit(X)
        visualizer.finalize()

        kmeans = KMeans(n_clusters=i, algorithm="full")
        visualizer = InterclusterDistance(kmeans, ax=axes[col_][1])
        visualizer.fit(X)
        visualizer.finalize()

        col_ += 1
    plt.show()


def generate_silhoutte_score_plot(X, k, model):
    n_clusters = np.arange(2, k)
    sils = []
    sils_err = []
    iterations = k
    for n in n_clusters:
        tmp_sil = []
        for _ in range(iterations):
            clf = model(n).fit(X)
            labels = clf.predict(X)
            sil = metrics.silhouette_score(X, labels, metric='euclidean')
            tmp_sil.append(sil)
        val = np.mean(SelBest(np.array(tmp_sil), int(iterations/5)))
        err = np.std(tmp_sil)
        sils.append(val)
        sils_err.append(err)
    plt.errorbar(n_clusters, sils, yerr=sils_err)
    plt.title("Silhouette Scores", fontsize=20)
    plt.xticks(n_clusters)
    plt.xlabel("N. of clusters ({})".format("Test"))
    plt.ylabel("Score")


def generate_distance_bw_gmms_plots(X, n):
    # https://towardsdatascience.com/gaussian-mixture-model-clusterization-how-to-select-the-number-of-components-clusters-553bef45f6e4

    n_clusters = np.arange(2, n)
    iterations = n
    results = []
    res_sigs = []
    for n in n_clusters:
        dist = []

        for iteration in range(iterations):
            train, test = train_test_split(X, test_size=0.5)

            gmm_train = GaussianMixture(n, n_init=2).fit(train)
            gmm_test = GaussianMixture(n, n_init=2).fit(test)
            dist.append(gmm_js(gmm_train, gmm_test))
        selec = SelBest(np.array(dist), int(iterations/5))
        result = np.mean(selec)
        res_sig = np.std(selec)
        results.append(result)
        res_sigs.append(res_sig)

    plt.errorbar(n_clusters, results, yerr=res_sigs)
    plt.title("Distance between Train and Test GMMs", fontsize=20)
    plt.xticks(n_clusters)
    plt.xlabel("N. of components")
    plt.ylabel("Distance")
    plt.show()


def generate_bic_plot(X, n):
    n_clusters = np.arange(2, n)
    bics = []
    bics_err = []
    iterations = n
    for n in n_clusters:
        tmp_bic = []
        for _ in range(iterations):
            gmm = GaussianMixture(n, n_init=2).fit(X)

            tmp_bic.append(gmm.bic(X))
        val = np.mean(SelBest(np.array(tmp_bic), int(iterations/5)))
        err = np.std(tmp_bic)
        bics.append(val)
        bics_err.append(err)

    plt.errorbar(n_clusters, bics, yerr=bics_err, label='BIC')
    plt.title("BIC Scores", fontsize=20)
    plt.xticks(n_clusters)
    plt.xlabel("N. of components")
    plt.ylabel("Score")
    plt.legend()
    plt.show()
    plt.clf()

    plt.errorbar(n_clusters, np.gradient(bics), yerr=bics_err, label='BIC')
    plt.title("Gradient of BIC Scores", fontsize=20)
    plt.xticks(n_clusters)
    plt.xlabel("N. of components")
    plt.ylabel("grad(BIC)")
    plt.legend()
    plt.show()
    plt.clf()
