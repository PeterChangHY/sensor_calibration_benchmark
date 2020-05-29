import cv2


def rotate_image(mat, angle):
    # angle in degrees
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)
    """
    opencv's getRotationMatrix2D:
    angle *= CV_PI/180;
    double alpha = std::cos(angle)*scale;
    double beta = std::sin(angle_)*scale;
    Matx23d M =
             [ alpha, beta, (1-alpha)*center.x - beta*center.y,]
             [ -beta, alpha, beta*center.x + (1-alpha)*center.y]

  """
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # get the new image size
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # shift to the new image's center
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

    return rotated_mat
