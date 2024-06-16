import cv2
import matplotlib.pyplot as plt

# Load the image
imagePath = "image5.jpg"
img = cv2.imread(imagePath)
print(img.shape)

# Convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray_image.shape)

# Load the pre-trained face detection model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Detect faces in the image
faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

# Convert the image from BGR to RGB for Matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image with Matplotlib
plt.figure(figsize=(20, 10))
cv2.imshow('Detected Faces', img)
cv2.waitKey(0)  # Wait for a key press to close the image window
cv2.destroyAllWindows()  