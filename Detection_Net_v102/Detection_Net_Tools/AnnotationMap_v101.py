import cv2
import numpy as np
import Image_Loader_HelenDataset_v102 as IML_Helen

def draw_circles_on_image(image_path, x_coordinates, y_coordinates, circle_color=(0, 255, 0), circle_radius=5):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at '{image_path}'")
        return

    # Ensure the number of x and y coordinates match
    if len(x_coordinates) != len(y_coordinates):
        print("Error: Number of x and y coordinates do not match")
        return

    # Draw circles on the image
    for x, y in zip(x_coordinates, y_coordinates):
        cv2.circle(image, (int(x), int(y)), circle_radius, circle_color, -1)  # -1 fills the circle

    # Show the image with circles
    cv2.imshow("Image with Circles", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


draw_circles_on_image(IML_Helen.train_image_master_list[0][0],IML_Helen.annotations_master_file[0][1],IML_Helen.annotations_master_file[0][2])
