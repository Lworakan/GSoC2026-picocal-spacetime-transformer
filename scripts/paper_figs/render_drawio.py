import subprocess
import sys
import time
from pathlib import Path

# Renders paper/spacetformer.drawio to PNG and PDF without a drawio desktop install: the
# diagram is handed to the viewer's own JavaScript inside a local page, and the headless
# Chromium that ships with the playwright cache prints it. Kept in the repository because
# the architecture figure has to be regenerated whenever the model changes, and a
# hand-exported image silently goes stale.
#
# The canvas is fixed and generous rather than cropped by re-rendering. The viewer scales
# the drawing to fit its container, so shrinking the page rescales the content instead of
# trimming it, and an iterative crop clips the drawing rather than converging on it. The
# white margin is removed afterwards by setting the PDF's CropBox, which keeps the vectors.

CHROME = Path.home() / '.cache/ms-playwright/chromium-1208/chrome-linux64/chrome'
REPO = Path(__file__).resolve().parents[2]
SRC = REPO / 'paper' / 'spacetformer.drawio'
OUT_PDF = REPO / 'paper' / 'spacetformer.pdf'
OUT_PNG = REPO / 'paper' / 'spacetformer.png'
W, H = 1500, 1150
PAD = 10


def embed_html(xml: str, w: int, h: int) -> str:
    import html
    import json
    cfg = json.dumps({'highlight': '#0000ff', 'nav': False, 'toolbar': '',
                      'edit': None, 'resize': True, 'fit': True, 'xml': xml})
    return ('<!doctype html><meta charset="utf-8">'
            f'<style>@page{{size:{w}px {h}px;margin:0}}html,body{{margin:0;padding:0;'
            f'background:#fff;width:{w}px;height:{h}px;overflow:hidden}}'
            '.mxgraph{max-width:none!important}</style>'
            f'<div class="mxgraph" style="background:#fff;width:{w}px;height:{h}px" '
            f'data-mxgraph="{html.escape(cfg, quote=True)}"></div>'
            '<script src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>')


def shoot(page, flag):
    subprocess.run([str(CHROME), '--headless', '--disable-gpu', '--no-sandbox',
                    '--no-pdf-header-footer', '--hide-scrollbars',
                    f'--window-size={W},{H}', '--virtual-time-budget=20000',
                    flag, page.as_uri()], check=True, capture_output=True)
    time.sleep(0.2)


def main():
    if not CHROME.exists():
        sys.exit(f'headless chromium not found at {CHROME}')
    from PIL import Image, ImageChops
    from pypdf import PdfReader, PdfWriter
    page = REPO / '.scratch' / 'drawio_render.html'
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(embed_html(SRC.read_text(), W, H))

    shoot(page, f'--screenshot={OUT_PNG}')
    im = Image.open(OUT_PNG).convert('RGB')
    box = ImageChops.difference(im, Image.new('RGB', im.size, (255, 255, 255))).getbbox()
    if box is None:
        sys.exit('blank render')
    if box[2] >= im.size[0] - 2 or box[3] >= im.size[1] - 2:
        sys.exit(f'the drawing fills the {W}x{H} canvas; raise W/H in this script')
    left, top = max(box[0] - PAD, 0), max(box[1] - PAD, 0)
    right, bottom = min(box[2] + PAD, im.size[0]), min(box[3] + PAD, im.size[1])
    im.crop((left, top, right, bottom)).save(OUT_PNG)

    shoot(page, f'--print-to-pdf={OUT_PDF}')
    reader = PdfReader(str(OUT_PDF))
    p0 = reader.pages[0]
    ph = float(p0.mediabox.height)
    sx = float(p0.mediabox.width) / im.size[0]        # CSS pixels -> PDF points
    sy = ph / im.size[1]
    p0.cropbox.lower_left = (left * sx, ph - bottom * sy)
    p0.cropbox.upper_right = (right * sx, ph - top * sy)
    writer = PdfWriter()
    writer.add_page(p0)
    with open(OUT_PDF, 'wb') as f:
        writer.write(f)
    print(f'wrote spacetformer.pdf/.png, drawing is '
          f'{right - left}x{bottom - top} px on a {W}x{H} canvas')


if __name__ == '__main__':
    main()
