import os

dataset_path = "training"

total = 0

for class_name in sorted(os.listdir(dataset_path)):
    class_path = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_path):
        continue

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    count = len(images)
    total += count

    print(f"{class_name}: {count} images")

print("-" * 30)
print(f"Total: {total} images")