import argparse
import io
import json
import logging
import multiprocessing
import os
import pickle
import signal
import time
from typing import Optional

import qoi
import fitz
import numpy as np
import polars as pl
import psutil
import tesserocr
from PIL import Image
from tesserocr import PyTessBaseAPI
from tesserocr import get_languages
from tqdm import tqdm

from GLAM.common import get_bytes_per_pixel, truncate_with_ellipsis, pixmap_to_image, pixmap_to_ndarray

# logging.basicConfig(filename='dln.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger_stream_handler = logging.StreamHandler()
logger.addHandler(logger_stream_handler)
logger.setLevel(logging.DEBUG)

INVALID_UNICODE = chr(0xFFFD)

tessdata_prefix = os.environ.get("TESSDATA_PREFIX", "/usr/share/tesseract/tessdata")
tesseract_languages_required = ["eng", "deu", "fra", "jpn"]
api = PyTessBaseAPI(path=tessdata_prefix, lang="+".join(tesseract_languages_required))


CLASSES_MAP = {
    1: "Caption",
    2: "Footnote",
    3: "Formula",
    4: "List-item",
    5: "Page-footer",
    6: "Page-header",
    7: "Picture",
    8: "Section-header",
    9: "Table",
    10: "Text",
    11: "Title",
}


def worker_init():
    # global api
    # api = PyTessBaseAPI(path=tessdata_prefix, lang="+".join(tesseract_languages_required))
    pass


def pdf_extract(
        pdf_filepath: str,
        scale: float = 1,
) -> list[tuple[np.ndarray, int, int, dict]]:
    """Returns a list of tuples (image_webp, width, height, page_dict)"""
    doc = fitz.Document(pdf_filepath)
    result: list[tuple[np.ndarray, int, int, dict]] = []

    for page_i in range(doc.page_count):
        page: fitz.Page = doc.load_page(page_i)
        page_dict = fitz.utils.get_text(page=page, option="dict", clip=page.rect, flags=fitz.TEXT_PRESERVE_IMAGES)

        # Filter out empty image blocks
        page_dict["blocks"] = [
            block
            for block in page_dict["blocks"]
            if not (block["type"] == 1 and len(block["image"]) == 0)
        ]
        # Filter out empty spans and resolve invalid unicode
        for block in page_dict["blocks"]:
            if block["type"] == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]

                        if INVALID_UNICODE in text:
                            page_pixmap = fitz.utils.get_pixmap(
                                page=page,
                                matrix=fitz.Matrix(5, 5),  # 360 dpi
                                clip=span["bbox"],
                                colorspace=fitz.csRGB,
                                alpha=False,
                            )
                            if page_pixmap.samples_ptr != 0:
                                bpp = get_bytes_per_pixel(page_pixmap.colorspace, page_pixmap.alpha)
                                api.SetImageBytes(
                                    imagedata=page_pixmap.samples,
                                    width=page_pixmap.w,
                                    height=page_pixmap.h,
                                    bytes_per_pixel=bpp,
                                    bytes_per_line=page_pixmap.stride,
                                )
                                api.SetPageSegMode(tesserocr.PSM.RAW_LINE)
                                api.Recognize()
                                ocr_text = api.GetUTF8Text().rstrip()

                                ls = " " * (len(text) - len(text.lstrip()))
                                rs = " " * (len(text) - len(text.rstrip()))
                                old_text, text = text, ls + ocr_text + rs
                                span["text_ocr"] = text
                                # logger.debug(f"Replaced {old_text!r} with {text!r}")

                    # Use list comprehension to filter empty spans
                    line["spans"] = [span for span in line["spans"] if not span["text"].strip() == ""]
            elif block["type"] == 1:
                # QOI
                block_image: Image.Image = Image.open(io.BytesIO(block["image"]))
                block_image: np.ndarray = np.array(block_image)  # Makes a copy
                if block_image.ndim in (3, 4):
                    image_qoi = qoi.encode(block_image)
                else:
                    image_qoi = None

                # WebP
                try:
                    block_image = Image.open(io.BytesIO(block["image"]))
                    block_webp_buffer = io.BytesIO()
                    block_image.save(block_webp_buffer, format="WEBP", lossless=True, quality=100, method=1)
                    image_webp = block_webp_buffer.getvalue()
                except OSError as e:
                    print(f"Failed to open image block {truncate_with_ellipsis(str(block), 128)} in pdf {pdf_filepath}: {e}")
                    image_webp = None

                # Select the best image format
                smallest_image = block["image"]
                if image_qoi is not None and len(image_qoi) < len(smallest_image):
                    smallest_image = image_qoi
                if image_webp is not None and len(image_webp) < len(smallest_image):
                    smallest_image = image_webp
                block["image"] = smallest_image
            else:
                raise ValueError(f"Unknown block type {block['type']} in pdf {pdf_filepath}")

        if scale != 0:
            page_pixmap = fitz.utils.get_pixmap(
                page=page,
                matrix=fitz.Matrix(scale, scale),
                colorspace=fitz.csRGB,
                alpha=False,
            )
            assert page_pixmap.samples_ptr != 0
            image = pixmap_to_ndarray(page_pixmap)
        else:
            image = None

        result.append((image, page.rect.width, page.rect.height, page_dict))

    return result


def process(
        pdf_filepath: str,
        scale: float,
        split_name: str,
        row: dict,
):
    try:
        return pdf_extract(pdf_filepath, scale), (pdf_filepath, scale, split_name, row)
    except Exception as e:
        print(f"pdf_ser failed: {e}")
        return None


def main():
    # signal.signal(signal.SIGINT, lambda sig, frame: exit(sig))

    parser = argparse.ArgumentParser("DocLayNet dataset preparation. Using paper proposed dataset splits.")
    parser.add_argument("--dataset-path", type=str, default="/home/i/dataset/DocLayNet/raw/DocLayNet",
                        help="Directory for the raw dataset (default: %(default)s)")
    parser.add_argument("--image-scale", type=float, default=1,
                        help="Set scaling factor for an image. A scale of 1 is 72 dpi. (default: %(default)s)")
    args = parser.parse_args()

    print("Processing DocLayNet dataset")
    split_names = ["train", "test", "val"]

    num_processes = psutil.cpu_count(logical=False)
    logger.debug(f"Using {num_processes} processes.")
    tasks_in_pool = 0
    max_tasks_in_pool = 100 + num_processes

    pbar = tqdm(desc=f"Processing...", smoothing=0.001)

    with multiprocessing.Pool(num_processes, initializer=worker_init) as pool:
        def callback(result):
            nonlocal tasks_in_pool
            tasks_in_pool -= 1
            pbar.update(1)

            if result is None:
                return

            example, (orig_pdf_filepath, scale, split_name, row) = result
            assert len(example) == 1, f"Expected 1 page, got {len(result)} pages"
            image, width, height, page_dict = example[0]

            id_file = os.path.join(args.dataset_path, "DATA", split_name, "by-id", str(row["id"]))
            os.makedirs(id_file, exist_ok=True)
            pdf_filepath = os.path.join(id_file, "page.pdf")
            row_filepath = os.path.join(id_file, "row.json")
            webp_filepath = os.path.join(id_file, "image.webp")
            qoi_filepath = os.path.join(id_file, "image.qoi")
            page_dict_filepath = os.path.join(id_file, "page_dict.pkl")
            annotations_filepath = os.path.join(id_file, "annotations.json")

            # Convert annotations to original coordinates
            scale_x = width / row["width"]
            scale_y = height / row["height"]
            annotations = []
            for ann_id in image_id_to_annotations_index.get(row["id"], []):
                ann = split_coco['annotations'][ann_id]
                for b in range(0, len(ann['bbox']), 2):
                    ann['bbox'][b] *= scale_x
                    ann['bbox'][b + 1] *= scale_y
                for seg in ann['segmentation']:
                    for s in range(0, len(seg), 2):
                        seg[s] *= scale_x
                        seg[s + 1] *= scale_y
                annotations.append(ann)

            with open(annotations_filepath, "w", encoding="utf-8") as f:
                json.dump(annotations, f)

            if image is not None:
                # Save image as QOI
                _ = qoi.write(qoi_filepath, image)

                # Save image as WebP
                image = Image.fromarray(image)
                image.save(webp_filepath, format="WEBP", lossless=True, quality=100, method=1)

            if page_dict is not None:
                with open(page_dict_filepath, "wb") as f:
                    pickle.dump(page_dict, f)

            # Save row
            row["width"] = width
            row["height"] = height
            with open(row_filepath, "w", encoding="utf-8") as f:
                json.dump(row, f)

            # Hard link PDF from pdf_filepath to orig_pdf_filepath
            try:
                os.unlink(pdf_filepath)
            except FileNotFoundError:
                pass
            os.link(orig_pdf_filepath, pdf_filepath)

        def my_error_callback(e):
            nonlocal tasks_in_pool
            tasks_in_pool -= 1
            pbar.update(1)
            # logger.exception(e)

        for split_name in split_names:
            coco_filepath = os.path.join(args.dataset_path, "COCO", f"{split_name}.json")
            with open(coco_filepath, "r", encoding="utf-8") as f:
                split_coco = json.load(f)

            image_id_to_annotations_index = {}
            for i, ann in enumerate(split_coco['annotations']):
                image_id_to_annotations_index.setdefault(ann['image_id'], []).append(i)

            pbar.reset(total=len(split_coco["images"]))

            for row in split_coco["images"]:
                page_hash = row["file_name"][:-4]
                id_file = os.path.join(args.dataset_path, "DATA", split_name, "by-id", str(row["id"]))
                row_filepath = os.path.join(id_file, "row.json")
                pdf_filepath = os.path.join(args.dataset_path, "PDF", page_hash + ".pdf")

                # Skip if already processed
                if os.path.exists(row_filepath):
                    pbar.update(1)
                    continue

                while tasks_in_pool >= max_tasks_in_pool:
                    time.sleep(0.1)

                tasks_in_pool += 1
                pool.apply_async(process, args=(pdf_filepath, args.image_scale, split_name, row), callback=callback, error_callback=my_error_callback)
                # callback(process(orig_pdf_filepath, args.image_scale, split_name, row))

            while tasks_in_pool > 0:
                pbar.refresh()
                print("Tasks in pool:", tasks_in_pool)
                print("Waiting for following tasks:")
                # print(pool._cache)
                print(pool._taskqueue)
                time.sleep(1)

        print("Finished processing DocLayNet dataset")


if __name__ == '__main__':
    main()
