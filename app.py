import hashlib
import sys
import uuid
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

try:
    import pymupdf as fitz
except Exception:
    import fitz

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "承诺书.pdf"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

SIG_BOX: Dict[str, Any] = {
    "pageIndex": 2,
    "x": 0.66,
    "y": 0.32,
    "w": 0.18,
    "h": 0.08,
}

st.set_page_config(page_title="第十二届慈善健康行安全承诺书", layout="wide")


def read_pdf_bytes(path: Path) -> bytes:
    return path.read_bytes()


@st.cache_data(
    show_spinner=False,
    hash_funcs={bytes: lambda b: hashlib.sha256(b).hexdigest()},
)
def render_pdf_to_png_pages(pdf_bytes: bytes, zoom: float = 2.0) -> List[bytes]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(zoom, zoom)
    pages: List[bytes] = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pages.append(pix.tobytes("png"))
    doc.close()
    return pages


def pdf_viewer(pdf_bytes: bytes, zoom: float = 2.0):
    pages = render_pdf_to_png_pages(pdf_bytes, zoom=zoom)
    if not pages:
        st.warning("PDF 无法渲染。")
        return
    for i, img_bytes in enumerate(pages):
        st.image(img_bytes, use_container_width=True)
        if i < len(pages) - 1:
            st.divider()


def is_blank_canvas(rgba_array: Optional[np.ndarray]) -> bool:
    if rgba_array is None:
        return True
    if rgba_array.ndim != 3 or rgba_array.shape[2] < 4:
        return True
    alpha = rgba_array[:, :, 3]
    return int(alpha.max()) == 0


def canvas_to_png_bytes(rgba_array: np.ndarray) -> bytes:
    img = Image.fromarray(rgba_array.astype("uint8"), mode="RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def overlay_signature(pdf_bytes: bytes, sig_png_bytes: bytes, sig_box: Dict[str, Any]) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    page_index = int(sig_box["pageIndex"])
    if page_index < 0 or page_index >= doc.page_count:
        doc.close()
        raise ValueError(f"pageIndex={page_index} 超出范围：PDF 共 {doc.page_count} 页")

    page = doc[page_index]
    r = page.rect

    x0 = float(sig_box["x"]) * r.width
    y0 = float(sig_box["y"]) * r.height
    x1 = x0 + float(sig_box["w"]) * r.width
    y1 = y0 + float(sig_box["h"]) * r.height
    rect = fitz.Rect(x0, y0, x1, y1)

    page.insert_image(rect, stream=sig_png_bytes, keep_proportion=True)

    out = doc.tobytes()
    doc.close()
    return out


def save_signed_pdf(pdf_bytes: bytes) -> str:
    doc_id = uuid.uuid4().hex[:12]
    out_path = DATA_DIR / f"{doc_id}.pdf"
    out_path.write_bytes(pdf_bytes)
    return doc_id


def load_signed_pdf(doc_id: str) -> Optional[bytes]:
    p = DATA_DIR / f"{doc_id}.pdf"
    if p.exists():
        return p.read_bytes()
    return None


try:
    original_pdf = read_pdf_bytes(PDF_PATH)
except FileNotFoundError:
    st.error(f"找不到 PDF：{PDF_PATH}")
    st.stop()
    sys.exit(1)

st.title("第十二届慈善健康行安全承诺书")

doc_param: Union[str, List[str], None] = st.query_params.get("doc")
if isinstance(doc_param, list):
    doc_param = doc_param[0] if doc_param else None

shared_pdf = load_signed_pdf(doc_param) if doc_param else None

if shared_pdf:
    colA, colB = st.columns([1.2, 1])
    with colA:
        st.subheader("预览")
        pdf_viewer(shared_pdf, zoom=2.0)
    with colB:
        st.subheader("下载")
        st.download_button(
            label="下载已签署 PDF",
            data=shared_pdf,
            file_name=f"{doc_param}_signed.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    st.stop()

col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("预览")
    preview_pdf = st.session_state.get("signed_pdf", original_pdf)
    pdf_viewer(preview_pdf, zoom=2.0)

with col_right:
    st.subheader("签名")

    if "show_pad" not in st.session_state:
        st.session_state.show_pad = False

    if st.button("签名", use_container_width=True):
        st.session_state.show_pad = True

    if st.session_state.show_pad:
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=3,
            stroke_color="#000000",
            background_color="rgba(255, 255, 255, 0)",
            height=220,
            width=520,
            drawing_mode="freedraw",
            key="sig_canvas",
        )

        c1, c2 = st.columns(2)

        with c1:
            if st.button("确认并生成已签署 PDF", use_container_width=True):
                if is_blank_canvas(canvas_result.image_data):
                    st.warning("签名为空。")
                else:
                    try:
                        sig_png = canvas_to_png_bytes(canvas_result.image_data)
                        signed_pdf = overlay_signature(original_pdf, sig_png, SIG_BOX)
                    except Exception as e:
                        st.error(f"生成失败：{e}")
                        st.stop()

                    st.session_state["signed_pdf"] = signed_pdf
                    doc_id = save_signed_pdf(signed_pdf)
                    st.session_state["doc_id"] = doc_id
                    st.query_params["doc"] = doc_id
                    st.success("已生成。")

        with c2:
            if st.button("关闭", use_container_width=True):
                st.session_state.show_pad = False

    st.divider()

    signed = st.session_state.get("signed_pdf")
    doc_id = st.session_state.get("doc_id")

    if signed:
        st.download_button(
            label="下载已签署 PDF",
            data=signed,
            file_name="signed.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    if doc_id:
        st.code(f"?doc={doc_id}", language="text")
