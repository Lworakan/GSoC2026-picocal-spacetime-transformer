import subprocess
import sys
import time
from pathlib import Path

# Renders paper/spacetformer.drawio to PNG and PDF without a drawio desktop install:
# the diagram is deflate+base64 encoded into a viewer.diagrams.net #R fragment, opened in
# the headless Chromium that ships with the playwright cache, and printed to PDF. Kept in
# the repository because the architecture figure has to be regenerated whenever the model
# changes, and a hand-exported image silently goes stale.

CHROME = Path.home() / '.cache/ms-playwright/chromium-1208/chrome-linux64/chrome'
REPO = Path(__file__).resolve().parents[2]
SRC = REPO / 'paper' / 'spacetformer.drawio'
OUT_PDF = REPO / 'paper' / 'spacetformer.pdf'
OUT_PNG = REPO / 'paper' / 'spacetformer.png'


def embed_html(xml: str, w: int = 1400, h: int = 1260) -> str:
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


def shoot(page, w, h, flag):
    subprocess.run([str(CHROME), '--headless', '--disable-gpu', '--no-sandbox',
                    '--no-pdf-header-footer', '--hide-scrollbars',
                    f'--window-size={w},{h}', '--virtual-time-budget=20000',
                    flag, page.as_uri()], check=True, capture_output=True)
    time.sleep(0.2)


def main():
    if not CHROME.exists():
        sys.exit(f'headless chromium not found at {CHROME}')
    from PIL import Image, ImageChops
    xml = SRC.read_text()
    page = REPO / '.scratch' / 'drawio_render.html'
    page.parent.mkdir(parents=True, exist_ok=True)
    w, h = 1400, 1260
    # first pass at a generous canvas, then crop the page to the drawing's own bounding box
    # so the figure carries no dead margin into the paper
    page.write_text(embed_html(xml, w, h))
    shoot(page, w, h, f'--screenshot={OUT_PNG}')
    im = Image.open(OUT_PNG).convert('RGB')
    box = ImageChops.difference(im, Image.new('RGB', im.size, (255, 255, 255))).getbbox()
    h = min(h, box[3] + 24)
    w = min(w, box[2] + 24)
    page.write_text(embed_html(xml, w, h))
    for target, flag in ((OUT_PDF, f'--print-to-pdf={OUT_PDF}'),
                         (OUT_PNG, f'--screenshot={OUT_PNG}')):
        shoot(page, w, h, flag)
        print('wrote', target.relative_to(REPO), f'{w}x{h}',
              target.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
