"""Draw the app icon (a trapped cell between two laser beams) and write celldform/demo/assets/celldform.{png,ico}.

Generated here so there is no third-party artwork to license. Run:  python installer/make_icon.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent.parent / "celldform" / "demo" / "assets"
SIZES = (16, 24, 32, 48, 64, 128, 256)


def draw(size: int = 512) -> Image.Image:
    s = size
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, s - 1, s - 1), radius=int(s * 0.22), fill=(22, 52, 92, 255))
    # two beams meeting at the cell
    for side in (-1, 1):
        x0 = s * (0.04 if side < 0 else 0.96)
        glow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        g = ImageDraw.Draw(glow)
        g.polygon([(x0, s * 0.40), (x0, s * 0.60), (s * 0.5, s * 0.5)], fill=(255, 196, 64, 150))
        img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(s * 0.012)))
    # the cell, stretched along the beam axis, with a nucleus
    d = ImageDraw.Draw(img)
    d.ellipse((s * 0.22, s * 0.31, s * 0.78, s * 0.69), fill=(226, 238, 250, 255), outline=(255, 255, 255, 255), width=max(2, s // 90))
    d.ellipse((s * 0.40, s * 0.42, s * 0.62, s * 0.58), fill=(96, 148, 214, 255))
    d.ellipse((s * 0.45, s * 0.46, s * 0.53, s * 0.52), fill=(190, 218, 248, 255))
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    big = draw(512)
    big.resize((256, 256), Image.LANCZOS).save(OUT / "celldform.png")
    big.save(OUT / "celldform.ico", sizes=[(n, n) for n in SIZES])
    print("wrote", OUT / "celldform.png", "and", OUT / "celldform.ico")


if __name__ == "__main__":
    main()
