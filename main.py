import numpy as np
import cv2
import os

# --- CONFIGURATION ---
# Define paths for models and images
PROTOTXT = os.path.join("models", "colorization_deploy_v2.prototxt")
MODEL = os.path.join("models", "colorization_release_v2.caffemodel")
POINTS = os.path.join("models", "pts_in_hull.npy")

# Input Image (Change this name to your file)
IMAGE_PATH = os.path.join("images", "einstein.jpg") 
OUTPUT_PATH = "result_colorized.jpg"

def main():
    # 1. Check if model files exist
    if not os.path.exists(MODEL) or not os.path.exists(PROTOTXT) or not os.path.exists(POINTS):
        print("[ERROR] Model files missing! Please run 'download_models.py' first.")
        return

    # 2. Load the Network
    print("[INFO] Loading Deep Learning model...")
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)

    # 3. Setup Layer Cluster Centers
    # The model uses these cluster centers to quantize colors in Lab space.
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # 4. Load and Preprocess Image
    print(f"[INFO] Processing image: {IMAGE_PATH}")
    if not os.path.exists(IMAGE_PATH):
        print(f"[ERROR] Image not found at {IMAGE_PATH}")
        return

    image = cv2.imread(IMAGE_PATH)
    normalized = image.astype("float32") / 255.0
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)

    # Resize the 'L' channel to 224x224 (Network Input Size)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50 # Subtract mean centering

    # 5. Predict Colors (a and b channels)
    print("[INFO] Colorizing...")
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # 6. Post-Processing
    # Resize the predicted 'ab' channels back to the original image size
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    # Grab the 'L' channel from the ORIGINAL image (for sharpness)
    L = cv2.split(lab)[0]
    
    # Merge L (Original) + ab (Predicted)
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # Convert Lab back to BGR
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1) # Ensure values are within 0-1
    
    # Convert to 8-bit integer (0-255)
    colorized = (255 * colorized).astype("uint8")

    # 7. Visualization (Side-by-Side)
    # Resize original image to match height if necessary (usually same)
    original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_gray = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR) # Make it 3 channel for stacking
    
    combined = np.hstack([original_gray, colorized])
    
    # Save and Show
    cv2.imwrite(OUTPUT_PATH, combined)
    print(f"[SUCCESS] Saved colorized image to '{OUTPUT_PATH}'")
    
    cv2.imshow("Original (B&W) vs AI Colorized", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()