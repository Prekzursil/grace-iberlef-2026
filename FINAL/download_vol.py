import os, modal
vol = modal.Volume.from_name("grace-out")
os.makedirs("modal_out", exist_ok=True)
n = 0
for e in vol.iterdir("/"):
    p = e.path
    try:
        with open(os.path.join("modal_out", os.path.basename(p)), "wb") as f:
            for chunk in vol.read_file(p):
                f.write(chunk)
        n += 1
    except Exception as ex:
        print("skip", p, str(ex)[:60])
print("downloaded", n, "files")
