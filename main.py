from poisson import *

def main():
	src_img_1 = cv2.imread("img/test1_src.jpg", cv2.IMREAD_COLOR)
	dst_img_1 = cv2.imread("img/test1_target.jpg", cv2.IMREAD_COLOR)
	mask_1 = cv2.imread("img/test1_mask.jpg", cv2.IMREAD_GRAYSCALE)

	src_img_2 = cv2.imread("img/test2_src.png", cv2.IMREAD_COLOR)
	dst_img_2 = cv2.imread("img/test2_target.png", cv2.IMREAD_COLOR)
	mask_2 = cv2.imread("img/test2_mask.png", cv2.IMREAD_GRAYSCALE)

	cv2.imwrite("img/test_1_naive.jpg", naive_fuse(src_img_1, dst_img_1, mask_1, 23, 120))
	cv2.imwrite("img/test_2_naive.jpg", naive_fuse(src_img_2, dst_img_2, mask_2, 170, 185))
	cv2.imwrite("img/test_1_poisson.jpg", fuse_image(src_img_1, dst_img_1, mask_1, 23, 120))
	cv2.imwrite("img/test_2_poisson.jpg", fuse_image(src_img_2, dst_img_2, mask_2, 170, 185))
	

if __name__ == "__main__":
	main()