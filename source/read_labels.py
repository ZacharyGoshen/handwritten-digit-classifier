import numpy as np

def retrieve_labels(file, label_indices):
	"""
	Takes in a file path and an array of indices

	Returns a NumPy matrix of the one hot encodings of the specified labels
	"""

	# Initialize numpy matrix to store the images
	labels = np.zeros((len(label_indices), 10))

	with open(file, "rb") as f:
		# Intialize counters
		i = 0
		label_number = 0

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
			labels[label_number] = np.zeros(10)
			labels[label_number, value] = 1

			# Increment to next label
			label_number += 1

	return labels