# estoreric algorithm (cat vs dog, the inefficient way)

import sys
sys.set_int_max_str_digits(10000000)

import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# database from Kaggle, for cats and dogs classification
# 12,499 cats and 1,499 dogs

IMG_SIZE = 32
DATASET_PATH = r"C:\Users\Usuario\Desktop\esoteric\PetImages"
MAX_IMAGES = 10000

def load_and_resize(image_path, size = 32):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((size, size))
        return np.array(img)
    except:
        return None


def flatten_image(img):
    return img.flatten()

def vector_to_integer(vec):
    N = 0
    base = 256

    for val in vec:
        N = N * base + int(val)

    return N

def build_dataset(dataset_path):
    numbers = []
    labels = []
    ids = []
    vectors = []
    images = []

    classes = {"Cat":0, "Dog":1}


    for class_name, label in classes.items():
        folder = Path(dataset_path)/class_name
        print("folder reader:", folder)

        count = 0

        for file in sorted(folder.glob("*.jpg")):

            if count >= MAX_IMAGES:
                break

            img = load_and_resize(file)

            if img is None:
                continue

            vec = flatten_image(img)
            num = vector_to_integer(vec)
            numbers.append(num)
            labels.append(label)
            ids.append((file.name))
            vectors.append(vec)
            images.append(img)

            count += 1
            if count % 50 ==0:
                print(f"Processed {count} images...")
    
    return vectors, numbers, labels, ids, images

vectors, numbers, labels, ids, images = build_dataset(DATASET_PATH)

def generate_permutation(N,p):
    seed = N % (2**32)
    rng = np.random.default_rng(seed)
    perm=rng.permutation(p)
    return perm

def scramble_vector(vec,perm):
    scrambled = vec[perm]
    return scrambled

def scramble_dataset(vectors, numbers):
    scrambled_vectros = []
    p = IMG_SIZE * IMG_SIZE * 3

    for vec, N in zip(vectors, numbers):
        perm = generate_permutation(N, p)
        scrambled = scramble_vector(vec, perm)
        scrambled_vectros.append(scrambled)
    return scrambled_vectros

scrambled_vectors = scramble_dataset(vectors, numbers)

#similarities between scrambled images
def eucledian_distance(a,b):
    return np.linalg.norm(a-b)

def knn_predict(train_vectors, train_labels, test_vec, k=3):
    distances = []

    for vec, label in zip(train_vectors, train_labels):
        dist = eucledian_distance(vec,test_vec)
        distances.append((dist,label))

    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    votes = [label for _, label in neighbors]
    prediction = max(set(votes), key=votes.count)
    return prediction

def train_test_split(images, vectors,labels,test_ratio=0.2):
    n=len(vectors)
    split = int(n*(1-test_ratio))

    train_vectors = vectors[:split]
    train_labels = labels[:split]

    test_vectors = vectors[split:]
    test_labels = labels[split:]
    test_images = images[split:]

    return train_vectors, train_labels, test_vectors, test_labels, test_images


train_vectors, train_labels, test_vectors, test_labels, test_images = train_test_split(images, 
                                                                                       scrambled_vectors, 
                                                                                       labels)
correct = 0

for vec, true_label in zip(test_vectors, test_labels):
    pred = knn_predict(train_vectors, train_labels, vec, k=3)
    if pred == true_label:
        correct += 1

accuracy = correct / len(test_labels)
print(f"Accuracy: {accuracy:.2f}")

def show_predictions(test_images, test_vectors, test_labels, train_vectors, train_labels, k=3, num_examples=3):
    for i in range(num_examples):
        img = test_images[i]
        vec = test_vectors[i]
        true_label = test_labels[i]

        pred = knn_predict(train_vectors, train_labels, vec, k)

        plt.imshow(img)
        plt.title(f"True: {'Cat' if true_label == 0 else 'Dog'}, Pred: {'Cat' if pred == 0 else 'Dog'}")
        plt.axis('off')
        plt.show()


show_predictions(test_images, test_vectors, test_labels, train_vectors, train_labels, num_examples=3)

            

