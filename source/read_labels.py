from matplotlib import pyplot as plt
import numpy as np

def retrieve_labels(file, label_indices):
	# Initialize numpy matrix to store the images
	labels = np.zeros(len(label_indices))

	with open(file, "rb") as f:
		# Intialize counters
		i = 0
		label_number = 0

		# Sort label indices
		label_indices.sort()

		# Read first byte
		byte = f.read(1)

		# Find each image in the data file
		for label_index in label_indices:
			# Read in bytes until you arrive at the label
			while byte and (i < (label_index + 8)):
				byte = f.read(1)
				i += 1

			# Store label value in numpy array
			value = int.from_bytes(byte, "big")
			labels[label_number] = value

			# Increment to next label
			label_number += 1

	return labels

labels = retrieve_labels("./../data/train-labels.idx1-ubyte", [2, 0])
print(labels)