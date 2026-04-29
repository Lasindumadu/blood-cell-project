import cv2

def preprocess_image(image_path):

    img = cv2.imread(image_path)

    # Gaussian Blur
    img = cv2.GaussianBlur(img, (5,5), 0)

    # Convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l,a,b = cv2.split(lab)

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    cl = clahe.apply(l)

    merged = cv2.merge((cl,a,b))

    final = cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2BGR
    )

    return final