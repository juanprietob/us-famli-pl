import os

import numpy as np
import pandas as pd

import torch
import pickle
import argparse

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
import matplotlib.pyplot as plt

def main(args):
    
    csv_path = args.csv

    if(os.path.splitext(csv_path)[1] == ".csv"):        
        test_df = pd.read_csv(csv_path)
    else:
        test_df = pd.read_parquet(csv_path)

    features_path = args.features

    with open(features_path, 'rb') as f:
        features = pickle.load(f)
        features = features.squeeze()

    if args.split_fn:
        idx_sample = list(pd.read_csv(args.split_fn, header=None)[1])
    elif 0 < args.split and args.split < 1.0:
        idx_sample = np.random.choice(np.arange(features.shape[0]), size=int(features.shape[0]*args.split))

        print("Sample:", len(idx_sample))

        test_df = test_df.loc[idx_sample].reset_index(drop=True)
        features = features[idx_sample]

    if args.query is not None:
        test_df = test_df.query(args.query)
        features = features[test_df.index]
        test_df = test_df.reset_index(drop=True)

    cluster_centers = 'k-means++'
    if args.cluster_centers:
        if(os.path.splitext(args.cluster_centers)[1] == ".pickle"):
            with open(args.cluster_centers, 'rb') as f:
                cluster_centers = pickle.load(f)
        else:
            cluster_centers = torch.from_numpy(np.array(pd.read_csv(args.cluster_centers, header=None)))     

    # kmeans

    if args.elbow:
        # Instantiate the clustering model and visualizer
        model = KMeans()
        visualizer = KElbowVisualizer(
            model, k=(args.n_clusters_min, args.n_clusters), metric='distortion'
        )
        visualizer.fit(features)        # Fit the data to the visualizer

        n_clusters_optim = visualizer.elbow_value_
        
        out_elbow = features_path.replace(".pickle", "_elbow.png")
        print("Writing:", out_elbow)
        visualizer.show(outpath=out_elbow)        # Finalize and render the figure

        plt.figure()
        plt.clf()
        # Instantiate the clustering model and visualizer        
        visualizer = KElbowVisualizer(
            model, k=(args.n_clusters_min, args.n_clusters), metric='calinski_harabasz', locate_elbow=False
        )
        visualizer.fit(features)        # Fit the data to the visualizer
        
        out_elbow = features_path.replace(".pickle", "_elbow_calinski_harabasz.png")
        print("Writing:", out_elbow)
        visualizer.show(outpath=out_elbow)        # Finalize and render the figure


        plt.figure()
        plt.clf()
        # Instantiate the clustering model and visualizer        
        visualizer = KElbowVisualizer(
            model, k=(args.n_clusters_min, args.n_clusters), metric='silhouette', locate_elbow=False
        )
        visualizer.fit(features)        # Fit the data to the visualizer
        
        out_elbow = features_path.replace(".pickle", "_elbow_silhouette.png")
        print("Writing:", out_elbow)
        visualizer.show(outpath=out_elbow)        # Finalize and render the figure

        args.n_clusters = n_clusters_optim


    km = KMeans(
        n_clusters=args.n_clusters, init=cluster_centers, verbose=True, max_iter=args.max_iter
    ).fit(features)


    if args.silhouette:
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick')
        visualizer.fit(features)        # Fit the data to the visualizer

        out_silhouette = features_path.replace(".pickle", "_silhouette.png")
        print("Writing:", out_silhouette)
        visualizer.show(outpath=out_silhouette) 

    if isinstance(cluster_centers, str) and cluster_centers == "k-means++":
        out_centers = features_path.replace(".pickle", "_centers.pickle")
        print("Writing:", out_centers)
        with open(out_centers, 'wb') as f:
            pickle.dump(km.cluster_centers_, f)


    out_kmeans = features_path.replace(".pickle", "_kmeans_model.pickle")
    print("Writing:", out_kmeans)
    with open(out_kmeans, 'wb') as f:
        pickle.dump(km, f)

    cluster_centers = km.cluster_centers_

    test_df["pred_cluster"] = km.labels_

    if args.tsne and args.split_tsne is not None or args.tsne == 0:

        if(os.path.splitext(csv_path)[1] == ".csv"):        
            out_csv = features_path.replace(".pickle", "_kmeans_sample.csv")
            print("Writing:", out_csv)
            test_df.to_csv(out_csv, index=False)
        else:
            out_csv = features_path.replace(".pickle", "_kmeans_sample.parquet")
            print("Writing:", out_csv)
            test_df.to_parquet(out_csv, index=False)
        
        

        out_features = features_path.replace(".pickle", "_kmeans_sample.pickle")
        print("Writing:", out_features)
        with open(out_features, 'wb') as f:
            pickle.dump(features, f)
    

    if args.tsne:

        split_tsne = ""

        if args.split_tsne:
            idx_sample = np.random.choice(np.arange(features.shape[0]), size=int(features.shape[0]*args.split_tsne))
            test_df = test_df.loc[idx_sample].reset_index(drop=True)
            features = features[idx_sample]

            split_tsne = "_tsne_" + str(args.split_tsne)

        split_tsne += "_perplexity_" + str(args.perplexity)

        pca = PCA(n_components=2)
        pca.fit(features)
        print(pca.explained_variance_ratio_)
        print(pca.singular_values_)

        features_embedded_pca = pca.transform(features)

        test_df["pca_0"] = features_embedded_pca[:,0]
        test_df["pca_1"] = features_embedded_pca[:,1]

        plt.figure()
        plt.clf()
        sns.scatterplot(test_df, x='pca_0',y='pca_1', hue='pred_cluster', palette=sns.color_palette("hls", args.n_clusters), legend="brief")
        out_tsne = features_path.replace(".pickle", split_tsne + "pca_sample.png")
        plt.savefig(out_tsne)

        
        features_embedded = TSNE(n_components=2, perplexity=args.perplexity, random_state=123).fit_transform(features)    
        test_df["tsne_0"] = features_embedded[:,0]
        test_df["tsne_1"] = features_embedded[:,1]

        plt.figure()
        plt.clf()
        sns.scatterplot(test_df, x='tsne_0',y='tsne_1', hue='pred_cluster', palette=sns.color_palette("hls", args.n_clusters), legend="brief")
        out_tsne = features_path.replace(".pickle", split_tsne + "tsne_sample.png")
        plt.savefig(out_tsne)

        if "tag" in test_df.columns:
            plt.figure()
            plt.clf()
            sns.scatterplot(test_df, x='tsne_0',y='tsne_1', hue='tag', palette=sns.color_palette("hls", args.n_clusters), legend="brief")
            out_tsne = features_path.replace(".pickle", split_tsne + "tsne_sample_tag.png")
            plt.savefig(out_tsne)
        
        if(os.path.splitext(csv_path)[1] == ".csv"):        
            out_csv = features_path.replace(".pickle", split_tsne + "_sample.csv")
            print("Writing:", out_csv)
            test_df.to_csv(out_csv, index=False)
        else:
            out_csv = features_path.replace(".pickle", split_tsne + "_sample.parquet")
            print("Writing:", out_csv)
            test_df.to_parquet(out_csv, index=False)

        

        out_features = features_path.replace(".pickle", split_tsne + "_sample.pickle")
        print("Writing:", out_features)
        with open(out_features, 'wb') as f:
            pickle.dump(features, f)

        

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Clustering with torch implementation of kmeans')    
    parser.add_argument('--csv', type=str, help='CSV file for clustering', required=True)    
    parser.add_argument('--features', type=str, help='Features path', required=True)
    parser.add_argument('--tsne', type=int, help='Apply TSNE', default=1)
    parser.add_argument('--perplexity', type=int, help='perplexity for TSNE', default=300)
    parser.add_argument('--n_clusters_min', help='Minimum number of clusters for elbow analysis (starting k)', type=int, default=2)
    parser.add_argument('--silhouette', help='Perform silhouette analysis for kmeans and identify the optimal number of clusters', type=bool, default=False)
    parser.add_argument('--elbow', help='Perform elbow analysis for kmeans and identify the optimal number of clusters', type=bool, default=False)
    parser.add_argument('--n_clusters', help='Number of clusters', type=int, default=10)
    parser.add_argument('--cluster_centers', help='CSV_files with cluster centers', type=str, default=None)
    parser.add_argument('--max_iter', help='Maximum number of iterations', type=int, default=300)
    parser.add_argument('--split', help='Split the sample', type=float, default=1.0)
    parser.add_argument('--split_tsne', help='Split the sample for TSNE', type=float, default=None)
    parser.add_argument('--split_fn', help='Text file single column with indices to grab from features and dataframe', type=str, default=None)
    parser.add_argument('--query', help='Query to filter the dataframe', type=str, default=None)


    args = parser.parse_args()

    main(args)