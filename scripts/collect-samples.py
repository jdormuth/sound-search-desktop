'''
This file gets called once the user hits the library upload button
It generates the tsne and calculates the 20 closest beats/samples
'''

import sys
samples_root = sys.argv[1]
#file that user submitted
user_file = sys.argv[2]

data_root = './data/sounds/'
sr = 48000
max_length = sr*26 # ignore samples longer than 4 seconds
fixed_length = sr*13 # trim all samples to 250 milliseconds
limit = None # set this to 100 to only load the first 100 samples




sys.path.insert(0, './scripts/utils/')
import numpy as np
from os.path import join
from utils import *
from multiprocessing import Pool

#change to 'caf' for apple loops
files = list(list_all_files(samples_root, ['.wav']))
files.append(user_file)



def load_sample(fn, sr=None,
                max_length=None, fixed_length=None, normalize=True):
    if fn == '': # ignore empty filenames
        return None
    audio, _ = ffmpeg_load_audio(fn, sr, mono=True)
    duration = len(audio)
    if duration == 0: # ignore zero-length samples
        return None
    if max_length and duration >= max_length: # ignore long samples
        return None
    if fixed_length:
        audio.resize(fixed_length)
    max_val = np.abs(audio).max()
    if max_val == 0: # ignore completely silent sounds
        return None
    if normalize:
        audio /= max_val
    return (fn, audio, duration)

def job(fn):
    return load_sample(fn, sr=sr,max_length=max_length, fixed_length=fixed_length)
pool = Pool()
results = pool.map(job, files)


valid = filter(None, results)


filenames = [x[0] for x in valid]
samples = [x[1] for x in valid]
durations = [x[2] for x in valid]
samples = np.asarray(samples)

# np.savetxt(join(data_root, 'filenames.txt'), filenames, fmt='%s')


# np.savetxt(join(data_root, 'durations.txt'), durations, fmt='%i')

# np.save(join(data_root, 'samples.npy'), samples)

#samples to fingerprints
n_fft = 1024
hop_length = n_fft/4
use_logamp = False # boost the brightness of quiet sounds
reduce_rows = 10 # how many frequency bands to average into one
reduce_cols = 1 # how many time steps to average into one
crop_rows = 32 # limit how many frequency bands to use
crop_cols = 32 # limit how many time steps to use
limit = None # set this to 100 to only process 100 samples


from utils import *

from os.path import join
from matplotlib import pyplot as plt
from skimage.measure import block_reduce
from multiprocessing import Pool
import numpy as np
import librosa



# samples = np.load(join(data_root, 'samples.npy'))

window = np.hanning(n_fft)
def job(y):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
    
    amp = np.abs(S)
    
    if reduce_rows > 1 or reduce_cols > 1:
        amp = block_reduce(amp, (reduce_rows, reduce_cols), func=np.mean)
    if amp.shape[1] < crop_cols:
        amp = np.pad(amp, ((0, 0), (0, crop_cols-amp.shape[1])), 'constant')
    amp = amp[:crop_rows, :crop_cols]
    if use_logamp:
        amp = librosa.logamplitude(amp**2)
    amp -= amp.min()
    if amp.max() > 0:
        amp /= amp.max()
    amp = np.flipud(amp) # for visualization, put low frequencies on bottom
    return amp
pool = Pool()

fingerprints = pool.map(job, samples[:limit])
fingerprints = np.asarray(fingerprints).astype(np.float32)

# np.save(join(data_root, 'fingerprints.npy'), fingerprints)


initial_dims = [30]
perplexities = [5]
mode = 'fingerprints'

from matplotlib import pyplot as plt

from utils import *
from os.path import join
from multiprocessing import Pool
import numpy as np
import itertools

import pickle



def save_tsv(data, fn):
    np.savetxt(fn, data, fmt='%.5f', delimiter='\t')
def tsne(data, data_root, prefix, initial_dims=30, perplexity=30):
    
    # mkdir_p(data_root + 'tsne')
    # mkdir_p(data_root + 'plot')
    # figsize = (16,16)
    pointsize = 20
    X_2d = list(bh_tsne(data, initial_dims=initial_dims, perplexity=perplexity, no_dims=2))
    X_2d = normalize(np.array(X_2d))
    top_twenty = get_top_twenty(X_2d)
    X_2d = top_twenty[0]
    
    indexes = top_twenty[1]
    # save_tsv(X_2d, join(data_root, 'tsne/{}.{}.{}.2d.tsv'.format(prefix, initial_dims, perplexity)))
    # plt.figure(figsize=figsize)
    # plt.scatter(X_2d[:,0], X_2d[:,1], edgecolor='', s=pointsize)
    # plt.tight_layout()
    
    # plt.savefig(join(data_root, 'plot/{}.{}.{}.png'.format(prefix, initial_dims, perplexity)))
    # plt.close()
    # X_3d = list(bh_tsne(data, initial_dims=initial_dims, perplexity=perplexity, no_dims=3))
    # X_3d = normalize(np.array(X_3d))
    # x_3d = X_3d[indexes]
    # save_tsv(X_3d, join(data_root, 'tsne/{}.{}.{}.3d.tsv'.format(prefix, initial_dims, perplexity)))
    # plt.figure(figsize=figsize)
    # plt.scatter(X_2d[:,0], X_2d[:,1], edgecolor='', s=pointsize, c=X_3d[indexes])
    # with open(data_root + 'filenames.txt') as f:
    #     content = f.readlines()
    content = filenames
      
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    
        
   
    #content = [x[40:] for x in content]
    # for i, txt in enumerate(content):
    #     plt.annotate(txt, (X_2d[i][0],X_2d[i][1]))
    print "listen"
    sys.stdout.flush()
    import time
    for i in range(len(indexes)):
        # plt.annotate(content[indexes[i]],(X_2d[i][0],X_2d[i][1]))
        #print content[indexes[i]], (X_2d[i][0],X_2d[i][1])
        print content[indexes[i]]
        time.sleep(.1)
        sys.stdout.flush()
    time.sleep(1)
    sys.stdout.flush()
    # plt.tight_layout()
    # # plt.ylim([0,1])
    # # plt.xlim([0,1])
    # plt.savefig(join(data_root, 'plot/{}.{}.{}.png'.format(prefix, initial_dims, perplexity)))
    # plt.close()

#function used to get distance between 2 points
def get_distance(x1,y1,x2,y2):
    return np.square(x1-x2)+np.square(y1-y2)
def get_top_twenty(X):
    distances = []
    sorted_distances = []
    user_track_x = X[X.shape[0]-1][0]
    user_track_y = X[X.shape[0]-1][1]
    for i in range(X.shape[0]-1):
        track_x = X[i][0]
        track_y = X[i][1]
        distances.append(get_distance(track_x,track_y,user_track_x,user_track_y))
        sorted_distances.append(get_distance(track_x,track_y,user_track_x,user_track_y))
    
    sorted_distances.sort()
    
    #print distances
    indexes = []
    for i in range(10):
        #print sorted_distances[i]
        indexes.append(distances.index(sorted_distances[i]))
    indexes.append(X.shape[0]-1)
    return (X[indexes], indexes)

# if mode == 'fingerprints' or mode == 'combined':
#     fingerprints = np.load(join(data_root, 'fingerprints.npy'))
#     fingerprints = fingerprints.reshape(len(fingerprints), -1)
# if mode == 'predicted_labels' or mode == 'combined':
#     predicted_labels = np.load(join(data_root, 'predicted_labels.npy'))
#     predicted_labels -= predicted_labels.min()
#     predicted_labels /= predicted_labels.max()
# if mode == 'predicted_encoding' or mode == 'combined':
#     predicted_encoding = np.load(join(data_root, 'predicted_encoding.npy'))
#     std = predicted_encoding.std(axis=0)
#     predicted_encoding = predicted_encoding[:, std > 0] / std[std > 0]
    
# if mode == 'fingerprints':
#     data = fingerprints
# if mode == 'predicted_labels':
#     data = predicted_labels
# if mode == 'predicted_encoding':
#     data = predicted_encoding
# if mode == 'combined':
#     data = np.hstack((fingerprints, predicted_labels, predicted_encoding))
fingerprints = fingerprints.reshape(len(fingerprints), -1)
data = fingerprints
data = data.astype(np.float64)
def job(params):
    tsne(data, data_root, mode, initial_dims=params[0], perplexity=params[1])

    #print 'initial_dims={}, perplexity={}, {} seconds'.format(params[0], params[1], time() - start)
# params = list(itertools.product(initial_dims, perplexities))
# pool = Pool()
# pool.map(job, params)
tsne(data,data_root, mode, initial_dims = 30, perplexity = 5)

print "finished"
sys.stdout.flush()
quit()


