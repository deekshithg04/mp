# src/download_nonleaf.py

from icrawler.builtin import BingImageCrawler

SAVE_DIR = "leafnonleaf/nonleaf"

crawler = BingImageCrawler(storage={'root_dir': SAVE_DIR})

keywords = [
    # Humans (real-world photos, not anatomy)
    "real human face photo",
    "real person selfie",
    "man standing outdoor real photo",
    "woman standing photo real",
    "people group photo real",
    "college students group real",
    "human hand closeup real",
    "person walking street real",
    "farmer real photo",
    "worker outdoor real photo",

    # Clothes / Wearables
    "tshirt closeup real",
    "jeans cloth texture real",
    "hoodie closeup real",
    "shirt closeup real",

    # Tables / Indoor Objects
    "wooden table real photo",
    "kitchen table items real",
    "office desk photo real",
    "books on table real",
    "mobile phone on table real",
    "coffee mug on table real",

    # Backgrounds / Non-living textures
    "plain wall background real",
    "wood texture background real",
    "floor tile background real",
    "concrete background real",
    "soil background real",
]

print("📥 Starting download...\n")

for keyword in keywords:
    print(f"⬇ Downloading: {keyword}")
    crawler.crawl(
        keyword=keyword,
        max_num=25,          # every keyword ~25 images
        min_size=(200, 200), # avoid icons / wallpapers
        file_idx_offset='auto'
    )

print("\n✅ Done! All Non-leaf images saved to:", SAVE_DIR)