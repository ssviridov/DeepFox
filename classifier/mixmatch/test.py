from model import StateClassifier
from constants import LABELS
import torch
import numpy as np
from matplotlib import pyplot as plt

clf = StateClassifier()
clf.load_state_dict(torch.load('best_model.pkl', map_location='cpu'))

for i in range(50, 4000):
    image = np.load('../crowdsource_data/unlabeled_data/{}.npz'.format(i))['image']
    image = (image * 255).astype(np.uint8)

    tensor = torch.from_numpy(np.asarray([image]))
    probs = torch.sigmoid(clf(tensor)).detach().numpy()[0]

    idx = np.argsort(-probs)

    for i in idx:
        print(LABELS[i], probs[i])
    print('=' * 100)

    plt.imshow(image)
    plt.pause(10)
