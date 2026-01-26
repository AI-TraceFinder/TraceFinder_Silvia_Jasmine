import os
import cv2

# Absolute paths (FIXED)
INPUT_DIR = r"C:\Users\JASEMINE\Desktop\TraceFinder_Silvia_Jasmine\scanner_data"
OUTPUT_DIR = r"C:\Users\JASEMINE\Desktop\TraceFinder_Silvia_Jasmine\processed_data"

IMG_SIZE = 224  # Image size for ML models

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📂 Input Directory:", INPUT_DIR)
print("📂 Output Directory:", OUTPUT_DIR)

for scanner in os.listdir(INPUT_DIR):
    scanner_path = os.path.join(INPUT_DIR, scanner)
    output_scanner_path = os.path.join(OUTPUT_DIR, scanner)

    if not os.path.isdir(scanner_path):
        continue

    os.makedirs(output_scanner_path, exist_ok=True)

    print(f"\n🖨️ Processing scanner: {scanner}")

    for img_name in os.listdir(scanner_path):
        if img_name.lower().endswith((".tif", ".tiff", ".jpg", ".png")):
            img_path = os.path.join(scanner_path, img_name)

            try:
                img = cv2.imread(img_path)

                if img is None:
                    print("❌ Skipped (cannot read):", img_name)
                    continue

                # Resize image
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                # Save as JPG
                new_name = os.path.splitext(img_name)[0] + ".jpg"
                save_path = os.path.join(output_scanner_path, new_name)

                cv2.imwrite(save_path, img)
                print("✅ Saved:", save_path)

            except Exception as e:
                print("❌ Error processing:", img_name, e)

print("\n🎉 Preprocessing completed successfully!")
