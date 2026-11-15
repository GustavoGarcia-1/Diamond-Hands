"""Fetch the pinned Apache-licensed Aero Hand model and its referenced meshes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import tempfile
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from g1_receiver.aero import MODEL_DIR, UPSTREAM_REVISION, XML_SHA256, _load_source


BASE_URL = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{UPSTREAM_REVISION}/{MODEL_DIR}"


def fetch_model(destination: Path) -> Path:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aero-download-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        model = stage / MODEL_DIR
        (model / "assets").mkdir(parents=True)

        def fetch(filename: str) -> bytes:
            with urlopen(f"{BASE_URL}/{filename}", timeout=30) as response:
                data = response.read()
            (model / filename).write_bytes(data)
            return data

        mesh_names = set()
        for side, digest in XML_SHA256.items():
            xml = fetch(f"{side}_hand.xml")
            if sha256(xml).hexdigest() != digest:
                raise ValueError(f"unverified Aero XML for {side} hand")
            mesh_names.update(e.attrib["file"] for e in ET.fromstring(xml).findall("./asset/mesh"))
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(fetch, ["LICENSE", "README.md", *(f"assets/{name}" for name in sorted(mesh_names))]))
        _load_source(stage)
        if destination.name == MODEL_DIR:
            model.rename(destination)
            return destination
        stage.rename(destination)
        return destination / MODEL_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="new output directory for the verified source assets")
    args = parser.parse_args()
    print(fetch_model(args.destination))


if __name__ == "__main__":
    main()
