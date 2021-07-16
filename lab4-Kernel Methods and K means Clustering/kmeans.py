import argparse
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5927)

class KMeans():
    def __init__(self, D, n_clusters):
        self.n_clusters = n_clusters
        self.cluster_centers = np.zeros((n_clusters, D))

    def init_clusters(self, data):
        ### TODO
        ### Initialize cluster_centers using n_clusters points sampled from data
        row_index = np.arange(self.n_clusters)
        self.cluster_centers = data[row_index,:]
        ### END TODO

    def pred(self, x):
        ### TODO: Given a sample x, return id of closest cluster center
        #print(self.cluster_centers-x.reshape(1,-1))
        return np.argmin(np.linalg.norm(self.cluster_centers-x.reshape(1,-1),axis=1))
        ### END TODO

    def train(self, data, max_iter=10000, epsilon=1e-4):
        for it in range(max_iter):
            ### TODO
            ### Declare and initialize required variables
            labels = np.zeros((data.shape[0],1))
            old_clusters = self.cluster_centers.copy()
            ### Update labels for each point
            for i in range(data.shape[0]):
                labels[i][0] = self.pred(data[i,:])

            ### Update cluster centers
            ### Note: If some cluster is empty, do not update the cluster center
            for i in range(self.n_clusters):
                bool_array = np.full(data.shape,False)
                bool_array += (labels==i)
                #print(bool_array)
                if data[bool_array].shape[0] != 0:
                    self.cluster_centers[i,:] = np.mean(data[bool_array].reshape(-1,data.shape[1]),axis=0)
                #print("hi")

            ### Check for convergence
            ### Stop if distance between each of the old and new cluster centers is less than epsilon
            if np.all(np.linalg.norm(old_clusters-self.cluster_centers,axis=1)<epsilon):
                # print(old_clusters)
                # print(self.cluster_centers)
                # print(it)
                break
            ### END TODO
        return it

    def replace_by_center(self, data):
        out = np.zeros_like(data)
        for i, x in enumerate(data):
            out[i] = self.cluster_centers[self.pred(x)]
        return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='1', choices=['1', '2', '3'])
    parser.add_argument('--k', default=5, type=int)

    args = parser.parse_args()

    image = plt.imread(f'data/{args.image}.png')
    x = image.reshape(-1, 3)
    kmeans = KMeans(D=3, n_clusters=args.k)
    kmeans.init_clusters(x)
    kmeans.train(x)
    out = kmeans.replace_by_center(x)
    plt.imshow(out.reshape(image.shape))
    plt.show()

