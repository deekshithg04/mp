import os
from pathlib import Path
from PIL import Image
import hashlib


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"   # <-- change if needed


def md5(file_path):
    """Return MD5 of a file to detect duplicates."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def check_dataset(root_dir):
    print("\n===============================")
    print("📂 DATASET CHECK REPORT")
    print("===============================\n")

    root = Path(root_dir)

    if not root.exists():
        print(f"❌ Dataset directory not found: {root}")
        return

    for crop_dir in sorted(root.iterdir()):
        if not crop_dir.is_dir():
            continue

        print(f"\n🌿 CROP: {crop_dir.name}")
        print("-" * 40)

        class_counts = {}
        duplicate_hashes = {}
        duplicates = []
        corrupted = []
        small_images = []
        total_images = 0

        for class_dir in crop_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            class_counts[class_name] = 0

            for file in class_dir.glob("*"):
                if not file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    print(f"⚠️  Non-image file found: {file}")
                    continue

                # Check corruption
                try:
                    img = Image.open(file)
                    img.verify()
                except Exception:
                    corrupted.append(file)
                    continue

                # Check small image
                img = Image.open(file)
                if img.width < 150 or img.height < 150:
                    small_images.append(file)

                # Count valid image
                class_counts[class_name] += 1
                total_images += 1

                # Check duplicates
                h = md5(file)
                if h in duplicate_hashes:
                    duplicates.append((file, duplicate_hashes[h]))
                else:
                    duplicate_hashes[h] = file

        # REPORT
        print("📌 Class counts:")
        for cls, count in class_counts.items():
            print(f"   • {cls}: {count} images")

        if total_images == 0:
            print("❌ No images in this crop dataset.")
            continue

        # Imbalance Check
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())

        if max_count > min_count * 3:
            print("⚠️ Dataset imbalance detected!")

        if corrupted:
            print("\n❌ CORRUPTED IMAGES:")
            for f in corrupted[:10]:
                print("   ", f)
            if len(corrupted) > 10:
                print(f"   ... +{len(corrupted)-10} more")

        if small_images:
            print("\n⚠️ VERY SMALL IMAGES (<150px):")
            for f in small_images[:10]:
                print("   ", f)
            if len(small_images) > 10:
             print(f"   ... +{len(small_images)-10} more")

        if duplicates:
            print("\n⚠️ DUPLICATES FOUND:")
            for dup, orig in duplicates[:10]:
                print(f"   {dup}  == duplicate of ==>  {orig}")
            if len(duplicates) > 10:
                print(f"   ... +{len(duplicates)-10} more")

        print("\n✔ Crop checked successfully.")

    print("\n===============================")
    print("✅ Dataset scan complete!")
    print("===============================\n")


if __name__ == "__main__":
    check_dataset(DATASET_DIR)