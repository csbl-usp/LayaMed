"""Screenshots and an animated demo of the interface, for the README (run with the real model, never a stub).

Start the server with the trained checkpoint first (python server.py), then, in another terminal:
    python -m pip install playwright && python -m playwright install chromium
    python tools/capture_demo.py
Writes to docs/img/: interface.png (light theme), interface-dark.png (dark theme) and, if ffmpeg is installed,
demo.gif (the "Biliary → pancreatic" example typed word by word: cholecystitis turns into pancreatitis when the
lipase appears). On Colab: !apt-get -qq install ffmpeg && python -m playwright install --with-deps chromium.
Options: --url (default http://127.0.0.1:8000/), --example (button label), --chromium (browser executable path).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
VIEWPORT = {"width": 1200, "height": 820}


def type_example(page, url: str, label: str) -> None:
    page.goto(url)
    page.get_by_role("button", name=label).click()
    page.evaluate("window.scrollTo(0, 0)")  # the click scrolls down to the button
    page.wait_for_function("!document.querySelector('.examples button.typing')", timeout=120_000)
    page.wait_for_timeout(1500)  # last answer and bar transitions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/")
    ap.add_argument("--example", default="Biliary → pancreatic")
    ap.add_argument("--chromium", default=None, help="Chromium executable, if Playwright's own is not installed")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chromium)
        for scheme, name in (("light", "interface.png"), ("dark", "interface-dark.png")):
            page = browser.new_page(viewport=VIEWPORT, color_scheme=scheme, device_scale_factor=2)
            type_example(page, args.url, args.example)
            page.screenshot(path=OUT / name)
            page.close()
            print("Screenshot:", OUT / name)

        if not shutil.which("ffmpeg"):
            print("ffmpeg not found: skipping demo.gif")
            browser.close()
            return
        # Frames are screenshots taken while the example is typed (no video recording, so Playwright needs no ffmpeg
        # of its own); the system ffmpeg turns them into a GIF at the measured frame rate
        with tempfile.TemporaryDirectory() as tmp:
            page = browser.new_page(viewport=VIEWPORT, color_scheme="light")
            page.goto(args.url)
            page.wait_for_timeout(500)
            page.get_by_role("button", name=args.example).click()
            page.evaluate("window.scrollTo(0, 0)")
            frames, t0 = 0, time.time()
            while frames < 2 or page.query_selector(".examples button.typing"):
                page.screenshot(path=Path(tmp) / f"{frames:05d}.png")
                frames += 1
            for _ in range(int(2.5 * frames / (time.time() - t0))):  # hold the final state ~2.5 s
                page.screenshot(path=Path(tmp) / f"{frames:05d}.png")
                frames += 1
            fps = frames / (time.time() - t0)
            browser.close()
            gif = OUT / "demo.gif"
            # One pass with a palette: small file and clean colors
            filters = "scale=900:-1:flags=lanczos"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:.2f}",
                            "-i", str(Path(tmp) / "%05d.png"), "-vf",
                            f"{filters},split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer",
                            str(gif)], check=True)
            print(f"Demo: {gif} ({frames} frames, {fps:.1f} fps, {gif.stat().st_size / 1e6:.1f} MB)")

if __name__ == "__main__":
    main()
