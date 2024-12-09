import os
import shlex
import argparse
from tqdm import tqdm

# for python3: read in python2 pickled files
import _pickle as cPickle

import gzip
from sklearn.metrics.pairwise import cosine_similarity
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
    # TODO task b
    # Initialize BFMatcher
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.match(descriptors.astype(np.float32), clusters.astype(np.float32))
    
    # Create hard assignment matrix
    assignment = np.zeros((len(descriptors), len(clusters)), dtype=np.float32)
    for i, match in enumerate(matches):
        assignment[i, match.trainIdx] = 1  # Mark the closest cluster
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

    # TODO task b, c
    K, D = mus.shape  # K clusters, D dimensions
    encodings = []

    for f in tqdm(files):
        with gzip.open(f, 'rb') as ff:
            descriptors = cPickle.load(ff, encoding='latin1')

        # Compute assignments
        assignment = assignments(descriptors, mus)

        # Compute residuals and aggregate
        residuals = np.zeros((K, D), dtype=np.float32)
        for k in range(K):
            # Get descriptors assigned to cluster k
            assigned_descriptors = descriptors[assignment[:, k] > 0]
            if len(assigned_descriptors) > 0:
                residuals[k] = np.sum(assigned_descriptors - mus[k], axis=0)

        # Flatten residuals into a single vector
        vlad_vector = residuals.flatten()

        # Power normalization
        if powernorm:
            vlad_vector = np.sign(vlad_vector) * np.sqrt(np.abs(vlad_vector))

        # L2 normalization
        vlad_vector = normalize(vlad_vector.reshape(1, -1), norm='l2').flatten()

        encodings.append(vlad_vector)

    return np.array(encodings)

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
    # TODO d

    new_encs = []

    for i, test_vec in enumerate(tqdm(encs_test)):
        # Prepare training data for this SVM
        X = np.vstack([test_vec, encs_train])  # Combine test vector and all train vectors
        y = np.array([1] + [-1] * len(encs_train))  # Positive label for test_vec, negative for train

        # Train SVM
        clf = LinearSVC(C=C, class_weight='balanced', max_iter=1000)
        clf.fit(X, y)

        # Extract and normalize weight vector
        coef = clf.coef_.flatten()
        coef_normalized = normalize(coef.reshape(1, -1), norm='l2').flatten()

        new_encs.append(coef_normalized)

    return np.array(new_encs)


def distances(encs):
    """ 
    compute pairwise distances 

    parameters:
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """
    # compute cosine distance = 1 - dot product between l2-normalized
    # encodings
    # TODO task b
    # mask out distance with itself
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(encs)
    
    # Convert to distance (1 - similarity)
    dist_matrix = 1 - similarity_matrix
    
    # Mask out distances with itself
    np.fill_diagonal(dist_matrix, np.finfo(dist_matrix.dtype).max)
    return dist_matrix

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
    files_train, labels_train = getFiles(args.in_train, args.suffix, args.labels_train)
    files_test, labels_test = getFiles(args.in_test, args.suffix, args.labels_test)
    print('#train: {}'.format(len(files_train)))
    
    if not os.path.exists('mus.pkl.gz'):
        # TODO task a
        print("Loading descriptors...")
        descriptors = loadRandomDescriptors(files_train, max_descriptors=500000)
        print(f"> Loaded {descriptors.shape[0]} descriptors with dimension {descriptors.shape[1]}.")

        print('> compute dictionary')
        # print("Clustering descriptors to create codebook...")
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

  
    # b) VLAD encoding

    enc_train_file = 'enc_train.pkl.gz'
    enc_test_file = 'enc_test.pkl.gz'
    
    print('> compute VLAD for test')

    print('#test: {}'.format(len(files_test)))
    vlad_filename = 'enc_test.pkl.gz'
    if not os.path.exists(enc_train_file) or args.overwrite:
        enc_train = vlad(files_train, mus, powernorm=True)
        with gzip.open(enc_train_file, 'wb') as fOut:
            cPickle.dump(enc_train, fOut)
    else:
        with gzip.open(enc_train_file, 'rb') as f:
            enc_train = cPickle.load(f)

    if not os.path.exists(enc_test_file) or args.overwrite:
        enc_test = vlad(files_test, mus, powernorm=True)
        with gzip.open(enc_test_file, 'wb') as fOut:
            cPickle.dump(enc_test, fOut)
    else:
        with gzip.open(enc_test_file, 'rb') as f:
            enc_test = cPickle.load(f)


    # # cross-evaluate test encodings
    print('> evaluate')
    evaluate(enc_test, labels_test)

    # d) compute exemplar svms
    print('> compute VLAD for train (for E-SVM)')
    new_enc_test_file = 'new_enc_test.pkl.gz'
    if not os.path.exists(new_enc_test_file) or args.overwrite:
        new_enc_test = esvm(enc_test, enc_train, C=args.C)
        with gzip.open(new_enc_test_file, 'wb') as fOut:
            cPickle.dump(new_enc_test, fOut)
    else:
        with gzip.open(new_enc_test_file, 'rb') as f:
            new_enc_test = cPickle.load(f)

    print('> esvm computation')
    # eval
    evaluate(new_enc_test, labels_test)
    print('> evaluate') 


"""
RUN:

python skeleton.py --in_train ./train --labels_train ./icdar17_labels_train.txt --in_test ./test --labels_test ./icdar17_labels_test.txt

RESULT:

#train: 1182
Loading existing codebook...
Codebook loaded.
> compute VLAD for test
#test: 3600
> evaluate
Top-1 accuracy: 0.8219444444444445 - mAP: 0.6311244263965583
> compute VLAD for train (for E-SVM)
> esvm computation
Top-1 accuracy: 0.8858333333333334 - mAP: 0.7528622835090545
> evaluate


"""

    