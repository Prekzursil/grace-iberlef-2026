"""Download the grace-report volume (ensemble outputs) to FINAL/report_out/."""
import os
import modal

vol = modal.Volume.from_name("grace-report")
os.makedirs("report_out", exist_ok=True)
for entry in vol.iterdir("/"):
    name = entry.path.lstrip("/")
    with open(os.path.join("report_out", name), "wb") as fh:
        for chunk in vol.read_file(entry.path):
            fh.write(chunk)
    print("downloaded", name)
