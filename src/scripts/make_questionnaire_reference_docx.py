#!/usr/bin/env python
"""Generate a Pandoc reference.docx for questionnaire distribution.

Style goals:
- A4 portrait
- body font 10.5pt
- single line spacing
- narrowed but printable margins
"""

from __future__ import annotations

import argparse
import io
import subprocess
import zipfile
from typing import Optional
from xml.etree import ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
ET.register_namespace("w", W_NS)
ET.register_namespace("r", R_NS)


def w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def ensure_child(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def build_default_reference_docx_bytes(pandoc_exe: str) -> bytes:
    proc = subprocess.run(
        [pandoc_exe, "--print-default-data-file", "reference.docx"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout


def patch_document_xml(xml_bytes: bytes) -> bytes:
    tree = ET.ElementTree(ET.fromstring(xml_bytes))
    root = tree.getroot()

    body = root.find(w("body"))
    if body is None:
        raise RuntimeError("word/document.xml: w:body not found")

    sect_pr = body.find(w("sectPr"))
    if sect_pr is None:
        sect_pr = ET.SubElement(body, w("sectPr"))

    # A4 portrait
    pg_sz = ensure_child(sect_pr, w("pgSz"))
    pg_sz.set(w("w"), "11906")
    pg_sz.set(w("h"), "16838")

    # Narrow but practical print margins (~19mm)
    pg_mar = ensure_child(sect_pr, w("pgMar"))
    for side in ("top", "right", "bottom", "left"):
        pg_mar.set(w(side), "1080")
    pg_mar.set(w("header"), "720")
    pg_mar.set(w("footer"), "720")
    pg_mar.set(w("gutter"), "0")

    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue()


def patch_styles_xml(xml_bytes: bytes) -> bytes:
    tree = ET.ElementTree(ET.fromstring(xml_bytes))
    root = tree.getroot()

    doc_defaults = ensure_child(root, w("docDefaults"))

    rpr_default = ensure_child(doc_defaults, w("rPrDefault"))
    rpr = ensure_child(rpr_default, w("rPr"))

    # Japanese-friendly Windows default pairing.
    rfonts = ensure_child(rpr, w("rFonts"))
    rfonts.set(w("ascii"), "Calibri")
    rfonts.set(w("hAnsi"), "Calibri")
    rfonts.set(w("eastAsia"), "Meiryo")
    rfonts.set(w("cs"), "Calibri")

    # 10.5pt = 21 half-points.
    sz = ensure_child(rpr, w("sz"))
    sz.set(w("val"), "21")
    sz_cs = ensure_child(rpr, w("szCs"))
    sz_cs.set(w("val"), "21")

    ppr_default = ensure_child(doc_defaults, w("pPrDefault"))
    ppr = ensure_child(ppr_default, w("pPr"))

    spacing = ensure_child(ppr, w("spacing"))
    spacing.set(w("before"), "0")
    spacing.set(w("after"), "0")
    spacing.set(w("line"), "240")  # single spacing
    spacing.set(w("lineRule"), "auto")

    def patch_normal_style(style: ET.Element) -> None:
        ppr_style = ensure_child(style, w("pPr"))
        spacing_style = ensure_child(ppr_style, w("spacing"))
        spacing_style.set(w("before"), "0")
        spacing_style.set(w("after"), "0")
        spacing_style.set(w("line"), "240")
        spacing_style.set(w("lineRule"), "auto")

        rpr_style = ensure_child(style, w("rPr"))
        rfonts_style = ensure_child(rpr_style, w("rFonts"))
        rfonts_style.set(w("ascii"), "Calibri")
        rfonts_style.set(w("hAnsi"), "Calibri")
        rfonts_style.set(w("eastAsia"), "Meiryo")
        rfonts_style.set(w("cs"), "Calibri")

        sz_style = ensure_child(rpr_style, w("sz"))
        sz_style.set(w("val"), "21")
        sz_cs_style = ensure_child(rpr_style, w("szCs"))
        sz_cs_style.set(w("val"), "21")

    def patch_heading_style(
        style: ET.Element, *, bold: bool = True, line: Optional[str] = "240"
    ) -> None:
        ppr_style = ensure_child(style, w("pPr"))
        if line:
            spacing_style = ensure_child(ppr_style, w("spacing"))
            spacing_style.set(w("before"), "0")
            spacing_style.set(w("after"), "0")
            spacing_style.set(w("line"), line)
            spacing_style.set(w("lineRule"), "auto")

        rpr_style = ensure_child(style, w("rPr"))
        rfonts_style = ensure_child(rpr_style, w("rFonts"))
        rfonts_style.set(w("ascii"), "Calibri")
        rfonts_style.set(w("hAnsi"), "Calibri")
        rfonts_style.set(w("eastAsia"), "Meiryo")
        rfonts_style.set(w("cs"), "Calibri")
        sz_style = ensure_child(rpr_style, w("sz"))
        sz_style.set(w("val"), "21")
        sz_cs_style = ensure_child(rpr_style, w("szCs"))
        sz_cs_style.set(w("val"), "21")
        if bold:
            b = ensure_child(rpr_style, w("b"))
            b.set(w("val"), "1")

    for style in root.findall(w("style")):
        style_id = style.get(w("styleId"))
        if style_id == "Normal":
            patch_normal_style(style)
        if style_id in {"Heading1", "Heading2", "Heading3"}:
            patch_heading_style(style)

    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue()


def write_reference_docx(pandoc_exe: str, output_path: str) -> None:
    base = build_default_reference_docx_bytes(pandoc_exe)
    zin = zipfile.ZipFile(io.BytesIO(base), "r")

    out_buf = io.BytesIO()
    zout = zipfile.ZipFile(out_buf, "w", compression=zipfile.ZIP_DEFLATED)

    for info in zin.infolist():
        data = zin.read(info.filename)
        if info.filename == "word/document.xml":
            data = patch_document_xml(data)
        elif info.filename == "word/styles.xml":
            data = patch_styles_xml(data)
        zout.writestr(info, data)

    zin.close()
    zout.close()

    with open(output_path, "wb") as f:
        f.write(out_buf.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Pandoc reference.docx for questionnaire distribution "
            "(A4, 10.5pt, single spacing, narrowed margins)."
        )
    )
    parser.add_argument("--pandoc", required=True, help="Path to pandoc executable")
    parser.add_argument("--output", required=True, help="Output reference.docx path")
    args = parser.parse_args()
    write_reference_docx(args.pandoc, args.output)


if __name__ == "__main__":
    main()
