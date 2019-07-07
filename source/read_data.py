from matplotlib import pyplot as plt
import numpy as np

def retrieve_image_data(file, image_indices):
	# Initialize numpy matrix to store the images
	images = np.zeros((len(image_indices), 28, 28))

	with open("./../data/train-images.idx3-ubyte", "rb") as f:
		# Intialize counters
		i = 1
		row = 0
		col = 0
		image_number = 0
		byte_ranges = []

		# Sort image indices
		image_indices.sort()

		# Calculate the byte index for the beginning and end of each image in the data file
		for image_index in image_indices:
			byte_start = 16 + (28 * 28 * image_index) - 1
			byte_end = byte_start + (28 * 28) - 1
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

				# Assign each pixel value a location
				if (col < 27):
					col += 1
				else:
					col = 0
					row += 1

				# Store pixel value in numpy matrix
				images[image_number, row, col] = value

				# Read next byte
				byte = f.read(1)
				i += 1

			# Reset column and row counters and move to next image
			col = 0
			row = 0
			image_number += 1

	return images

images = retrieve_image_data("./../data/train-images.idx3-ubyte", [2, 0])
for image in images:
	plt.imshow(image, cmap="gray_r")
	plt.show()