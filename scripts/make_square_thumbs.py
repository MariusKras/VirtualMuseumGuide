from PIL import Image
from pathlib import Path

src = Path("assets/raw")
dst = Path("assets/processed")
dst.mkdir(parents=True, exist_ok=True)

for p in src.glob("*.*"):
    im = Image.open(p).convert("RGB")
    size = max(im.size)
    # create white square background
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    # paste original image centered
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    # resize to 400x400
    thumb = canvas.resize((600, 600), Image.LANCZOS)
    thumb.save(dst / p.name, quality=90)
