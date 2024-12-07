import os
import shlex
import argparse
from tqdm import tqdm

# for python3: read in python2 pickled files
import _pickle as cPickle

import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
import numpy as np
import cv2
from parmap import parmap

def parseArgs(parser):
    parser.add_argument('--labels_test', 
                        help='contains test images/descriptors to load + labels')
    parser.add_argument('--labels_train', 
                        help='contains training images/descriptors to load + labels')
    parser.add_argument('-s', '--suffix',
                        default='_SIFT_patch_pr.pkl.gz',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--in_test',
                        help='the input folder of the test images / features')
    parser.add_argument('--in_train',
                        help='the input folder of the training images / features')
    parser.add_argument('--overwrite', action='store_true',
                        help='do not load pre-computed encodings')
    parser.add_argument('--powernorm', action='store_true',
                        help='use powernorm')
    parser.add_argument('--gmp', action='store_true',
                        help='use generalized max pooling')
    parser.add_argument('--gamma', default=1, type=float,
                        help='regularization parameter of GMP')
    parser.add_argument('--C', default=1000, type=float, 
                        help='C parameter of the SVM')
    return parser

def getFiles(folder, pattern, labelfile):
    """ 
    returns files and associated labels by reading the labelfile 
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels 
    """
    # read labelfile
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()
    
    # get filenames from labelfile
    all_files = []
    labels = []
    check = True
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings, note: os.path.splitext() doesnt work for
        # '.' in the filenames, so let's do it this way...
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb','.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p,'')

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels

def loadRandomDescriptors(files, max_descriptors):
    """ 
    load roughly `max_descriptors` random descriptors
    parameters:
        files: list of filenames containing local features of dimension D
        max_descriptors: maximum number of descriptors (Q)
    returns: QxD matrix of descriptors
    """
    # let's just take 100 files to speed-up the process
    max_files = 100
    indices = np.random.permutation(max_files)
    files = np.array(files)[indices]
   
    # rough number of descriptors per file that we have to load
    max_descs_per_file = int(max_descriptors / len(files))

    descriptors = []
    for i in tqdm(range(len(files))):
        with gzip.open(files[i], 'rb') as ff:
            # for python2
            # desc = cPickle.load(ff)
            # for python3
            desc = cPickle.load(ff, encoding='latin1')
            
        # get some random ones
        indices = np.random.choice(len(desc),
                                   min(len(desc),
                                       int(max_descs_per_file)),
                                   replace=False)
        desc = desc[ indices ]
        descriptors.append(desc)
    
    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors

def dictionary(descriptors, n_clusters):
    """ 
    return cluster centers for the descriptors 
    parameters:
        descriptors: NxD matrix of local descriptors
        n_clusters: number of clusters = K
    returns: KxD matrix of K clusters
    """
    # TODO task a
    print(f"Clustering {descriptors.shape[0]} descriptors into {n_clusters} clusters...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000, random_state=42)
    kmeans.fit(descriptors)
    return kmeans.cluster_centers_
    
def assignments(descriptors, clusters):
    """ 
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """
    # compute nearest neighbors
    # TODO

    # create hard assignment
    assignment = np.zeros( (len(descriptors), len(clusters)) )
    # TODO

    return assignment

def vlad(files, mus, powernorm, gmp=False, gamma=1000):
    """
    compute VLAD encoding for each files
    parameters: 
        files: list of N files containing each T local descriptors of dimension
        D
        mus: KxD matrix of cluster centers
        gmp: if set to True use generalized max pooling instead of sum pooling
    returns: NxK*D matrix of encodings
    """
    K = mus.shape[0]
    encodings = []

    for f in tqdm(files):
        with gzip.open(f, 'rb') as ff:
            desc = cPickle.load(ff, encoding='latin1')
        a = assignments(desc, mus)
        
        T,D = desc.shape
        f_enc = np.zeros( (D*K), dtype=np.float32)
        for k in range(mus.shape[0]):
            pass
            # it's faster to select only those descriptors that have
            # this cluster as nearest neighbor and then compute the 
            # difference to the cluster center than computing the differences
            # first and then select

   
        # c) power normalization
        if powernorm:
            pass
            # TODO

        # l2 normalization
        # TODO


    return encodings

def esvm(encs_test, encs_train, C=1000):
    """ 
    compute a new embedding using Exemplar Classification
    compute for each encs_test encoding an E-SVM using the
    encs_train as negatives   
    parameters: 
        encs_test: NxD matrix
        encs_train: MxD matrix

    returns: new encs_test matrix (NxD)
    """


    # set up labels
    # TODO

    def loop(i):
        # compute SVM 
        # and make feature transformation
        # TODO
        return x

    # let's do that in parallel: 
    # if that doesn't work for you, just exchange 'parmap' with 'map'
    # Even better: use DASK arrays instead, then everything should be
    # parallelized
    new_encs = list(parmap( loop, tqdm(range(len(encs_test)))))
    new_encs = np.concatenate(new_encs, axis=0)
    # return new encodings
    return new_encs


def distances(encs):
    """ 
    compute pairwise distances 

    parameters:
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """
    # compute cosine distance = 1 - dot product between l2-normalized
    # encodings
    # TODO
    # mask out distance with itself
    np.fill_diagonal(dists, np.finfo(dists.dtype).max)
    return dists

def evaluate(encs, labels):
    """
    evaluate encodings assuming using associated labels
    parameters:
        encs: TxK*D encoding matrix
        labels: array/list of T labels
    """
    dist_matrix = distances(encs)
    # sort each row of the distance matrix
    indices = dist_matrix.argsort()

    n_encs = len(encs)

    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs-1):
            if labels[ indices[r,k] ] == labels[ r ]:
                rel += 1
                precisions.append( rel / float(k+1) )
                if k == 0:
                    correct += 1
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
    mAP = np.mean(mAP)

    print('Top-1 accuracy: {} - mAP: {}'.format(float(correct) / n_encs, mAP))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('retrieval')
    parser = parseArgs(parser)
    args = parser.parse_args()
    np.random.seed(42) # fix random seed
   
    # a) dictionary
    files_train, labels_train = getFiles(args.in_train, args.suffix,
                                         args.labels_train)
    print('#train: {}'.format(len(files_train)))
    
    if not os.path.exists('mus.pkl.gz'):
        # TODO task a
        print("Loading descriptors...")
        descriptors = loadRandomDescriptors(files_train, max_descriptors=500000)
        print(f"Loaded {descriptors.shape[0]} descriptors with dimension {descriptors.shape[1]}.")

        print("Clustering descriptors to create codebook...")
        mus = dictionary(descriptors, n_clusters=100)

        # Save the codebook
        with gzip.open('mus.pkl.gz', 'wb') as fOut:
            cPickle.dump(mus, fOut)
        print("Codebook saved as 'mus.pkl.gz'.")

    else:
        print("Loading existing codebook...")
        with gzip.open('mus.pkl.gz', 'rb') as f:
            mus = cPickle.load(f)
        print("Codebook loaded.")

  
    # # b) VLAD encoding
    # print('> compute VLAD for test')
    # files_test, labels_test = getFiles(args.in_test, args.suffix,
    #                                    args.labels_test)
    # print('#test: {}'.format(len(files_test)))
    # fname = 'enc_test_gmp{}.pkl.gz'.format(gamma) if args.gmp else 'enc_test.pkl.gz'
    # if not os.path.exists(fname) or args.overwrite:
    #     # TODO
    #     with gzip.open(fname, 'wb') as fOut:
    #         cPickle.dump(enc_test, fOut, -1)
    # else:
    #     with gzip.open(fname, 'rb') as f:
    #         enc_test = cPickle.load(f)
   
    # # cross-evaluate test encodings
    # print('> evaluate')
    # evaluate(enc_test, labels_test)

    # # d) compute exemplar svms
    # print('> compute VLAD for train (for E-SVM)')
    # fname = 'enc_train_gmp{}.pkl.gz'.format(gamma) if args.gmp else 'enc_train.pkl.gz'
    # if not os.path.exists(fname) or args.overwrite:
    #     # TODO
    #     with gzip.open(fname, 'wb') as fOut:
    #         cPickle.dump(enc_train, fOut, -1)
    # else:
    #     with gzip.open(fname, 'rb') as f:
    #         enc_train = cPickle.load(f)

    # print('> esvm computation')
    # # TODO

    # # eval
    # evaluate(enc_test, labels_test)
    # print('> evaluate')
