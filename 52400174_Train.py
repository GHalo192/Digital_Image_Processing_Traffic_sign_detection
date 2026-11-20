
import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "training")
MODEL_PATH = os.path.join(BASE_DIR, "52400174_TD_Model.pkl")

IMAGE_SIZE = 64

# HOG parameters
HOG_PARAMS = {
    "winSize": (64, 64),
    "blockSize": (16, 16),
    "blockStride": (8, 8),
    "cellSize": (8, 8),
    "nbins": 9
}


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def preprocessImage(image, size=IMAGE_SIZE):

    h, w = image.shape[:2]

    # Preserve aspect ratio
    scale = size / max(h, w)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    interpolation = (
        cv2.INTER_AREA if scale < 1
        else cv2.INTER_LINEAR
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=interpolation
    )

    # Create square canvas
    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    # Convert to grayscale
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    return gray


# ==========================================
# INITIALIZE HOG
# ==========================================

hog = cv2.HOGDescriptor(
    HOG_PARAMS["winSize"],
    HOG_PARAMS["blockSize"],
    HOG_PARAMS["blockStride"],
    HOG_PARAMS["cellSize"],
    HOG_PARAMS["nbins"]
)


# ==========================================
# LOAD DATASET AND EXTRACT HOG FEATURES
# ==========================================

def loadDataset():

    features = []
    labels = []

    class_counts = {}

    if not os.path.isdir(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_PATH}"
        )

    class_names = sorted([
        name for name in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, name))
    ])

    print("\n========== LOADING DATASET ==========\n")

    for class_name in class_names:

        class_path = os.path.join(DATASET_PATH, class_name)

        image_files = sorted([
            f for f in os.listdir(class_path)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            )
        ])

        loaded_count = 0

        for image_file in image_files:

            image_path = os.path.join(class_path, image_file)

            image = cv2.imread(image_path)

            if image is None:
                print(f"[WARNING] Cannot read: {image_path}")
                continue

            # Preprocess image
            processed = preprocessImage(image)

            # Extract HOG features
            hog_features = hog.compute(processed)

            if hog_features is None:
                print(f"[WARNING] HOG failed: {image_path}")
                continue

            features.append(hog_features.flatten())

            # Label = folder name
            labels.append(class_name)

            loaded_count += 1

        class_counts[class_name] = loaded_count

        print(f"{class_name}: {loaded_count} images")

    print("\n=====================================")

    X = np.array(features, dtype=np.float32)
    y = np.array(labels)

    print(f"Total images loaded: {len(X)}")
    print(f"Number of classes: {len(class_counts)}")

    return X, y, class_counts


# ==========================================
# TRAIN SVM MODEL
# ==========================================

def trainModel(X, y):

    print("\n========== TRAINING SVM ==========\n")

    model = SVC(
        kernel="linear",
        C=1.0,
        class_weight="balanced"
    )

    model.fit(X, y)

    print("Training completed!")

    # Training accuracy
    predictions = model.predict(X)

    accuracy = np.mean(predictions == y) * 100

    print(f"Training accuracy: {accuracy:.2f}%")

    return model


# ==========================================
# SAVE MODEL
# ==========================================

def saveModel(model, class_counts):

    model_data = {
        "model": model,
        "class_names": model.classes_.tolist(),
        "image_size": IMAGE_SIZE,
        "hog_params": HOG_PARAMS,
        "class_counts": class_counts
    }

    joblib.dump(model_data, MODEL_PATH)

    print("\n========== MODEL SAVED ==========\n")
    print(f"Model path: {MODEL_PATH}")
    print(f"Classes: {len(model.classes_)}")


# ==========================================
# MAIN
# ==========================================

def main():

    # Load dataset
    X, y, class_counts = loadDataset()

    if len(X) == 0:
        print("[ERROR] No valid images found!")
        return

    if len(np.unique(y)) < 2:
        print("[ERROR] SVM requires at least 2 classes!")
        return

    # Train model
    model = trainModel(X, y)

    # Save model
    saveModel(model, class_counts)

    print("\n========== FINISHED ==========")


if __name__ == "__main__":
    main()
