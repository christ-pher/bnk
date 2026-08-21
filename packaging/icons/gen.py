#!/usr/bin/env python3
"""Draw the bnk icon set from the logo master.

The icons are generated rather than hand-drawn so that a change to the
logo is one command away from reaching the tray, the executables and the
installer. Outputs are committed: the build must not need Pillow.

Run from anywhere:  python3 packaging/icons/gen.py

After changing the artwork, regenerate the resource objects too — they
carry the icon into the .exe files and are built from bnk.ico:

    packaging/icons/gen-syso.sh
"""
import os
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MASTER = os.path.join(HERE, "bnk-logo.png")

TRAY_DIR = os.path.join(ROOT, "internal", "trayicon", "icons")
APP_ICO = os.path.join(ROOT, "packaging", "windows", "bnk.ico")
SHEET = os.path.join(HERE, "design", "actual-size.png")

# Windows picks the frame closest to the current DPI, so the tray ships
# every size it might ask for rather than one it has to rescale.
TRAY_SIZES = [16, 20, 24, 32, 48, 64]
# The product icon is also shown large: Explorer's extra-large view and
# the Add or Remove Programs list both reach for 256.
APP_SIZES = [16, 20, 24, 32, 48, 64, 128, 256]

# Disconnected has no dot at all. A grey dot and a green one differ only
# in lightness, which is the one difference a 16px icon glimpsed in the
# corner of a screen does not carry; dot-versus-no-dot is a difference in
# shape, which it does. Amber survives as a dot because it differs from
# green in hue rather than lightness, and because the state it marks —
# the service down, or this machine not signed in — is one the user has
# to act on and must not read as a deliberate disconnect.
STATES = {
    "connected":    (46, 204, 113, 255),
    "disconnected": None,
    "attention":    (240, 173, 38, 255),
}

DOT = 0.34  # dot diameter as a fraction of the icon; larger swallows the face


def head():
    """The logo cropped square to the head.

    The master is a head-and-chest portrait. Fitting that whole shape
    into a square icon shrinks the head until the ears and eyes vanish
    at 16px, so the chest is cropped away and the head fills the frame.
    """
    im = Image.open(MASTER).convert("RGBA")
    im = im.crop(im.getbbox())
    w, _ = im.size
    return im.crop((int(w * 0.04), 0, int(w * 0.96), int(w * 0.96)))


def fit(src, sz, pad):
    s = src.copy()
    s.thumbnail((sz - 2 * pad, sz - 2 * pad), Image.LANCZOS)
    c = Image.new("RGBA", (sz, sz), (0, 0, 0, 0))
    c.paste(s, ((sz - s.width) // 2, (sz - s.height) // 2), s)
    return c


def tray(src, sz, colour):
    """One tray icon: the head, with a status dot over the lower jaw.

    A colour of None means no dot, which is how "disconnected" reads.

    The dot carries a white outline because the logo is nearly black and
    the taskbar may be either colour — without it the dot merges into
    one or the other.
    """
    c = fit(src, sz, max(1, round(sz / 16)))
    if colour is None:
        return c
    r = sz * DOT
    ImageDraw.Draw(c).ellipse(
        [sz - r, sz - r, sz - 1, sz - 1],
        fill=colour, outline=(255, 255, 255, 235), width=max(1, round(sz / 18)))
    return c


def app(src, sz):
    """The product icon: the logo alone, with no status to report."""
    return fit(src, sz, max(1, round(sz / 16)))


def write_ico(path, frames):
    """Save one .ico holding every size, each rendered at its own size.

    Letting the encoder downscale a single large frame instead would
    blur the small ones, and the small ones are what the tray shows.
    """
    frames = sorted(frames, key=lambda f: -f.width)
    frames[0].save(path, sizes=[(f.width, f.height) for f in frames],
                   append_images=frames[1:])


def contact_sheet(sets):
    """Draw the icons at 1:1 on both taskbar colours.

    Reviewing icons magnified is how unreadable ones get approved; this
    is the sheet that tells the truth.
    """
    gap, padx, cell = 14, 16, max(TRAY_SIZES)
    colw = sum(s + gap for s in TRAY_SIZES) + padx * 2
    rowh = cell + 14
    w = 110 + colw * 2 + 20
    h = 26 + rowh * len(sets) + 10
    sheet = Image.new("RGBA", (w, h), (120, 120, 120, 255))
    dr = ImageDraw.Draw(sheet)
    for bi, (bg, lab) in enumerate([((32, 32, 32, 255), "dark"),
                                    ((243, 243, 243, 255), "light")]):
        ox = 110 + bi * (colw + 10)
        sheet.paste(Image.new("RGBA", (colw, h - 26), bg), (ox, 26))
        dr.text((ox + 6, 10), lab, fill=(255, 255, 255, 255))
        for si, (name, frames) in enumerate(sets):
            oy = 30 + si * rowh
            x = ox + padx
            for f in frames:
                sheet.paste(f, (x, oy + (cell - f.height) // 2), f)
                x += f.width + gap
            if bi == 0:
                dr.text((8, oy + cell // 2 - 4), name, fill=(255, 255, 255, 255))
    sheet.save(SHEET)


def main():
    src = head()
    os.makedirs(TRAY_DIR, exist_ok=True)

    sets = []
    for name, colour in STATES.items():
        frames = [tray(src, sz, colour) for sz in TRAY_SIZES]
        write_ico(os.path.join(TRAY_DIR, f"{name}.ico"), frames)
        sets.append((name, frames))
        print(f"internal/trayicon/icons/{name}.ico  {TRAY_SIZES}")

    write_ico(APP_ICO, [app(src, sz) for sz in APP_SIZES])
    print(f"packaging/windows/bnk.ico  {APP_SIZES}")

    contact_sheet(sets)
    print("packaging/icons/design/actual-size.png")


if __name__ == "__main__":
    main()
