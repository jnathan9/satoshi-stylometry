"""Download trained model from Modal volume to local disk."""
import modal
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

vol = modal.Volume.from_name("satoshi-stylometry-data")

def main():
    # List what's in the volume
    for entry in vol.listdir("/model/best"):
        path = entry.path
        print(f"Downloading {path}...")
        # Read from volume
        data = b""
        for chunk in vol.read_file(path):
            data += chunk
        # Write locally
        local_path = os.path.join(OUTPUT_DIR, os.path.basename(path))
        with open(local_path, "wb") as f:
            f.write(data)
        print(f"  -> {local_path} ({len(data)} bytes)")

    # Also download the evaluation results
    for fname in ["golden_results.json", "evaluation_report.json"]:
        try:
            data = b""
            for chunk in vol.read_file(f"/{fname}"):
                data += chunk
            local_path = os.path.join(os.path.dirname(OUTPUT_DIR), "data", fname)
            with open(local_path, "wb") as f:
                f.write(data)
            print(f"Downloaded {fname}")
        except Exception as e:
            print(f"Could not download {fname}: {e}")

if __name__ == "__main__":
    main()
