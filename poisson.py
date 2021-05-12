import cv2
import numpy as np 
import scipy.linalg as linalg

OUTSIDE = 0
INSIDE = 1
IN_BOUND = 2
OUT_BOUND = 3

# get whether the point is inside, outside or boundary
def get_location(mask, x, y):
	surroundings = get_surroundings(x, y)

	in_count = 0
	out_count = 0

	for pt in surroundings:
		if pt[0] < 0 or pt[1] < 0 or pt[0] > mask.shape[1] or pt[1] > mask.shape[0]:
			out_count += 1
			continue

		if mask[pt[1], pt[0]] < 200:
			out_count +=1
			continue 

		in_count += 1

	if in_count == 4: return INSIDE
	if out_count == 4: return OUTSIDE

	if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
		return OUT_BOUND

	if mask[y, x] < 200:
		return OUT_BOUND
		
	return IN_BOUND


# get the surrounding pixels(take note of the coordinates, follow standard cartesian, not opencv)
def get_surroundings(x, y):
	return [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)]

# get list of points that is selected by mask
def get_mask_pts(mask):
	pt_array = []
	for row in range(0, mask.shape[0]):
		for col in range(0, mask.shape[1]):
			if mask[row, col] > 200:
				pt_array.append((col, row))

	return pt_array

# compute poisson sparse 
def compute_poisson_matrix(mask, mask_pts):
	N = len(mask_pts)

	mat_list = [[0 for _ in range(N)] for _ in range(N)]
	mat = np.array(mat_list)

	for i, point in enumerate(mask_pts):

		mat[i, i] = 4

		for s_pt in get_surroundings(point[0], point[1]):

			if s_pt not in mask_pts: 
				continue

			j = mask_pts.index(s_pt)
			mat[i, j] = -1

	return mat

# compute laplace
def compute_laplace(src_img, mask_pts, x, y, c):
	top = src_img[y-1, x, c]
	right = src_img[y, x+1, c]
	bottom = src_img[y+1, x, c]
	left = src_img[y, x-1, c]

	surroundings = get_surroundings(x, y)

	return (4 * src_img[y,x, c]) - top - right - bottom - left


# naive fusing
def naive_fuse(src_img, dst_img, mask,pos_x, pos_y):
	mask_pts = get_mask_pts(mask)
	dim = cv2.boundingRect(np.array(mask_pts))
	res_img = dst_img.copy()
	for point in mask_pts:
		res_img[point[1] - dim[1] + pos_y, point[0] - dim[0] + pos_x] = src_img[point[1], point[0]]

	return res_img


# driver function to perform the fusion
def fuse_image(src_img, dst_img, mask, pos_x, pos_y):

	res_img = np.copy(dst_img)
	mask_pts = get_mask_pts(mask)
	mask_len = len(mask_pts)
	dim = cv2.boundingRect(np.array(mask_pts))

	poisson_matrix = compute_poisson_matrix(mask, mask_pts)

	for c in range(0, 3):
		known_matrix = np.zeros(mask_len)

		for i, pt in enumerate(mask_pts):
			known_matrix[i] = compute_laplace(src_img, mask_pts, pt[0], pt[1], c)

			if get_location(mask, pt[0], pt[1]) == IN_BOUND:
				for s_pt in get_surroundings(pt[0], pt[1]):
					if get_location(mask, s_pt[0], s_pt[1]) == OUT_BOUND:
						known_matrix[i] += dst_img[s_pt[1] - dim[1] + pos_y, s_pt[0] - dim[0] + pos_x][c]

		composite_matrix = linalg.solve(poisson_matrix, known_matrix)

		for i, val in enumerate(composite_matrix):
			if val > 255:
				composite_matrix[i] = 255
			if val < 0 :
				composite_matrix[i] = 0

		for i, pt in enumerate(mask_pts):
			res_img[pt[1] - dim[1] + pos_y, pt[0] - dim[0] + pos_x, c] = composite_matrix[i]


	return res_img
	
	
	

