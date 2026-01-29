from sklearn.datasets import fetch_openml
from tqdm import trange
import numpy as np
import random
import json
import os

random.seed(1)
np.random.seed(1)

NUM_USERS = 20  # should be multiple of 10
NUM_CLASSES = 10
ALPHA = 0.1  # Dirichlet concentration parameter (smaller = more heterogeneous)

# Setup directory for train/test data
train_path = './data/train/mnist_train.json'
test_path = './data/test/mnist_test.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Get MNIST data, normalize, and divide by level
try:
    # Try to load from local file first
    import scipy.io
    mnist_file = './data/mldata/mnist-original.mat'
    if os.path.exists(mnist_file):
        print("Loading MNIST from local file...")
        mnist_raw = scipy.io.loadmat(mnist_file)
        mnist_data_array = mnist_raw['data'].T
        mnist_target_array = mnist_raw['label'][0]
        mnist = type('obj', (object,), {'data': mnist_data_array, 'target': mnist_target_array})()
    else:
        # Fallback to fetch_openml
        print("Downloading MNIST from openml...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        mnist.target = mnist.target.astype(np.int64)
except Exception as e:
    print(f"Error loading MNIST: {e}")
    print("Trying fetch_openml...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    mnist.target = mnist.target.astype(np.int64)

# Normalize data
mu = np.mean(mnist.data.astype(np.float32), 0)
sigma = np.std(mnist.data.astype(np.float32), 0)
mnist.data = (mnist.data.astype(np.float32) - mu) / (sigma + 0.001)

# Separate data by class
mnist_data = []
for i in trange(NUM_CLASSES):
    label_idx = mnist.target == i
    mnist_data.append(mnist.data[label_idx])

print("\nNumber of samples per class:\n", [len(v) for v in mnist_data])

# Shuffle data within each class
for i in range(NUM_CLASSES):
    indices = np.random.permutation(len(mnist_data[i]))
    mnist_data[i] = mnist_data[i][indices]

###### CREATE USER DATA SPLIT USING DIRICHLET DISTRIBUTION #######
# For each user, sample a class distribution from Dirichlet(alpha)
X = [[] for _ in range(NUM_USERS)]
y = [[] for _ in range(NUM_USERS)]

# Sample class distributions for each user from Dirichlet(alpha)
# Each user gets a probability distribution over classes
class_distributions = np.random.dirichlet([ALPHA] * NUM_CLASSES, NUM_USERS)

print("\nClass distributions for first 5 users:")
for i in range(min(5, NUM_USERS)):
    print(f"User {i}: {class_distributions[i]}")

# Calculate total samples per user (can be fixed or variable)
# Here we use a log-normal distribution to simulate varying data sizes
samples_per_user = (np.random.lognormal(4, 2, NUM_USERS).astype(int) + 50) * 5
print(f"\nSamples per user: min={samples_per_user.min()}, max={samples_per_user.max()}, mean={samples_per_user.mean():.1f}")

# Track indices for each class
class_indices = [0] * NUM_CLASSES

# Assign samples to each user based on their class distribution
for user in trange(NUM_USERS):
    num_samples = int(samples_per_user[user])
    
    # Sample class labels according to the user's class distribution
    class_probs = class_distributions[user]
    sampled_classes = np.random.choice(NUM_CLASSES, size=num_samples, p=class_probs)
    
    # Count how many samples of each class this user needs
    unique_classes, class_counts = np.unique(sampled_classes, return_counts=True)
    
    # Assign actual data samples
    for cls, count in zip(unique_classes, class_counts):
        # Get available samples for this class
        cls = int(cls)
        count = int(count)
        available = len(mnist_data[cls]) - class_indices[cls]
        actual_count = min(count, available)
        
        if actual_count > 0:
            start_idx = class_indices[cls]
            end_idx = start_idx + actual_count
            
            # Handle DataFrame vs numpy array
            if hasattr(mnist_data[cls], 'values'):
                X[user].extend(mnist_data[cls].iloc[start_idx:end_idx].values.tolist())
            else:
                X[user].extend(mnist_data[cls][start_idx:end_idx].tolist())
            
            y[user].extend([int(cls)] * actual_count)
            class_indices[cls] = int(end_idx)
    
    # Shuffle user's data
    combined = list(zip(X[user], y[user]))
    random.shuffle(combined)
    X[user][:], y[user][:] = zip(*combined)

print("\nFinal class indices:", class_indices)
print("\nSample distribution per user:")
for i in range(min(5, NUM_USERS)):
    unique_labels, counts = np.unique(y[i], return_counts=True)
    print(f"User {i}: {len(y[i])} samples, labels: {dict(zip(unique_labels, counts))}")

# Create data structure
train_data = {'users': [], 'user_data': {}, 'num_samples': []}
test_data = {'users': [], 'user_data': {}, 'num_samples': []}

# Split train/test (75%/25%)
for i in range(NUM_USERS):
    uname = 'f_{0:05d}'.format(i)
    
    num_samples = len(X[i])
    train_len = int(0.75 * num_samples)
    test_len = num_samples - train_len
    
    train_data['users'].append(uname)
    # Convert numpy types to Python native types for JSON serialization
    train_data['user_data'][uname] = {
        'x': X[i][:train_len], 
        'y': [int(label) for label in y[i][:train_len]]
    }
    train_data['num_samples'].append(int(train_len))
    
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {
        'x': X[i][train_len:], 
        'y': [int(label) for label in y[i][train_len:]]
    }
    test_data['num_samples'].append(int(test_len))

print("\nNum_samples per user:", train_data['num_samples'])
print("Total_samples:", sum(train_data['num_samples'] + test_data['num_samples']))

# Save to JSON files
with open(train_path, 'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)

print("\nFinish Generating Samples with Dirichlet(α={}) distribution".format(ALPHA))
