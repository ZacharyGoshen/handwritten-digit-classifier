import numpy as np

def retrieve_image_data(file, image_indices):
	"""
	Takes in a file path and an array of indices

	Returns a NumPy matrix of the specified images
	"""

	# Initialize numpy matrix to store the images
	images = np.zeros((len(image_indices), 784))

	with open(file, "rb") as f:
		# Intialize counters
		i = 0
		pixel_number = 0
		image_number = 0
		byte_ranges = []

		# Calculate the byte index for the beginning and end of each image in the data file
		for image_index in image_indices:
			byte_start = 16 + (28 * 28 * image_index)
			byte_end = byte_start + (28 * 28)
			byte_ranges.append((byte_start, byte_end))

		# Read first byte
		byte = f.read(1)

		# Find each image in the data file
		for (byte_start, byte_end) in byte_ranges:
			# Read in bytes until you arrive at the start of the image
			while byte and (i < byte_start):
				byte = f.read(1)
				i += 1

			# Iterate through each byte of the image
			while byte and (i < byte_end):
				# Convert byte value to integer pixel value
				value = int.from_bytes(byte, "big")

				# Store the normalized pixel value in the NumPy matrix
				images[image_number, pixel_number] = value
				pixel_number += 1

				# Read next byte
				byte = f.read(1)
				i += 1

			# Standardize data
			images[image_number] = images[image_number] - np.mean(images[image_number]) / np.std(images[image_number])

			# Reset column and row counters and move to next image
			pixel_number = 0
			image_number += 1

	return images