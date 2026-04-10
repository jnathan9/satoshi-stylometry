"""Upload processed data to Modal volume for training."""

import modal
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")

vol = modal.Volume.from_name("satoshi-stylometry-data", create_if_missing=True)

def main():
    files = ["train.json", "val.json", "test.json", "golden.json",
             "golden_satoshi.json", "golden_non_satoshi.json"]

    # Remove existing files first
    for f in files:
        try:
            vol.remove_file(f"/{f}")
        except Exception:
            pass

    with vol.batch_upload(force=True) as batch:
        for f in files:
            path = os.path.join(DATA_DIR, f)
            if os.path.exists(path):
                batch.put_file(path, f"/{f}")
                size = os.path.getsize(path) / 1024
                print(f"  Uploaded {f} ({size:.1f} KB)")
            else:
                print(f"  MISSING {f}")

    print("Done! Data uploaded to Modal volume 'satoshi-stylometry-data'")

if __name__ == "__main__":
    main()
