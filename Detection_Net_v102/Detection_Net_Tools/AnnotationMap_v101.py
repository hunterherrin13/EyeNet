import cv2
import numpy as np
import Image_Loader_HelenDataset_v102 as IML_Helen

def draw_circles_on_resized_image(image_path, x_coordinates, y_coordinates, circle_color=(0, 255, 0), circle_radius=1, resize_factor=0.5):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at '{image_path}'")
        return

    # Resize the image
    resized_image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)

    # Resize the coordinates of the circles
    scaled_x_coordinates = [int(x * resize_factor) for x in x_coordinates]
    scaled_y_coordinates = [int(y * resize_factor) for y in y_coordinates]

    # Draw circles on the resized image
    for x, y in zip(scaled_x_coordinates, scaled_y_coordinates):
        cv2.circle(resized_image, (x, y), circle_radius, circle_color, -1)  # -1 fills the circle

    # Show the resized image with circles
    cv2.imshow("Resized Image with Circles", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


draw_circles_on_resized_image(IML_Helen.train_ordered_path[6][0],IML_Helen.train_ordered_annotation[6][0],IML_Helen.train_ordered_annotation[6][1])
