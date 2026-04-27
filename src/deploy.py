"""Deploy assets to Jekyll repository."""

import shutil
from pathlib import Path

from src.config import (
    DATA_PROCESSED,
    JEKYLL_BASE,
    JEKYLL_CSS,
    JEKYLL_DATA,
    JEKYLL_JS,
    JEKYLL_PAGE,
    JEKYLL_REPO,
    VIZ_DIR,
)


def deploy_to_jekyll():
    """Deploy all assets to Jekyll repository."""
    if JEKYLL_REPO is None:
        print("ERROR: Variable de entorno 'JEKYLL_REPO' no definida.")
        return
    if not JEKYLL_REPO.exists():
        print(f"⚠ Jekyll repo not found at {JEKYLL_REPO}")
        print("Skipping deployment. Configure JEKYLL_REPO in config.py")
        return

    # Create Jekyll directories if they don't exist
    JEKYLL_DATA.mkdir(parents=True, exist_ok=True)
    JEKYLL_CSS.mkdir(parents=True, exist_ok=True)
    JEKYLL_JS.mkdir(parents=True, exist_ok=True)
    JEKYLL_PAGE.mkdir(parents=True, exist_ok=True)

    # Copy data files
    print(f"Copying data to {JEKYLL_DATA}...")
    for json_file in DATA_PROCESSED.glob("*.json"):
        dest = JEKYLL_DATA / json_file.name
        shutil.copy2(json_file, dest)
        print(f"  {json_file.name}")

    # Copy CSS
    print(f"Copying CSS to {JEKYLL_CSS}...")
    css_file = VIZ_DIR / "assets" / "css" / "regression.css"
    if css_file.exists():
        shutil.copy2(css_file, JEKYLL_CSS / "regression.css")
        print("  regression.css")

    # Copy JavaScript files
    print(f"Copying JavaScript to {JEKYLL_JS}...")
    js_dir = VIZ_DIR / "assets" / "js"
    if js_dir.exists():
        for js_file in js_dir.glob("*.js"):
            shutil.copy2(js_file, JEKYLL_JS / js_file.name)
            print(f"  {js_file.name}")

    # Copy HTML visualization
    print(f"Copying HTML to {JEKYLL_BASE}...")
    html_file = VIZ_DIR / "index.html"
    if html_file.exists():
        viz_html = JEKYLL_BASE / "viz.html"
        shutil.copy2(html_file, viz_html)
        print("  viz.html")

    # Copy Jekyll page
    jekyll_md = Path(__file__).parent.parent / "jekyll" / "comparativa-regresion.md"
    if jekyll_md.exists():
        shutil.copy2(jekyll_md, JEKYLL_PAGE / "comparativa-regresion.md")
        print("  jekyll/comparativa-regresion.md")

    print("✓ Deployment complete!")


if __name__ == "__main__":
    deploy_to_jekyll()
