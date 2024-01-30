import argparse
import logging
import os
import signal
from typing import Optional, Callable
from urllib.parse import urlparse
from zipfile import ZipFile

import requests
from tqdm import tqdm


def download(url: str, output_filepath: Optional[str] = None) -> str:
    if not output_filepath:
        parsed_url = urlparse(url)
        output_filepath = os.path.basename(parsed_url.path)

    if os.path.exists(output_filepath):
        return output_filepath

    tmp_output_filepath = output_filepath + ".download"

    if os.path.exists(tmp_output_filepath):
        resume_byte_pos = os.path.getsize(tmp_output_filepath)
        headers = {'Range': f'bytes={resume_byte_pos}-'} if resume_byte_pos else {}
    else:
        resume_byte_pos = 0
        headers = {}

    with requests.get(url, stream=True, headers=headers) as response:
        if response.status_code == 206:
            pass
        elif response.status_code == 200:
            resume_byte_pos = 0
        else:
            raise Exception(f"Unexpected response status code: {response.status_code}")

        total_size_in_bytes = int(response.headers.get('content-length', 0)) + resume_byte_pos
        block_size = 4 << 20  # 4 MiB
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, initial=resume_byte_pos)

        # Write or append to the file
        mode = 'ab' if resume_byte_pos else 'wb'
        with open(tmp_output_filepath, mode) as file:
            for data in response.iter_content(block_size):
                file.write(data)
                progress_bar.update(len(data))

        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            raise Exception("Something went wrong during the download")

    os.rename(tmp_output_filepath, output_filepath)

    return output_filepath


def extract_zip(
        zip_filepath: str,
        extract_path: Optional[str] = None,
        filename_predicate: Optional[Callable[[str], bool]] = None
) -> str:
    extract_path = extract_path if extract_path else os.getcwd()

    with ZipFile(zip_filepath, 'r') as zip_ref:
        file_list = [file for file in zip_ref.namelist() if filename_predicate(file)] if filename_predicate else zip_ref.namelist()
        with tqdm(total=len(file_list), desc="Extracting files") as progress_bar:
            for file in file_list:
                zip_ref.extract(member=file, path=extract_path)
                progress_bar.update(1)

    return extract_path


def download_extract_cached(
        url: str,
        *,
        output_filepath: Optional[str] = None,
        extract_path: Optional[str] = None,
        filename_predicate: Optional[Callable[[str], bool]] = None,
        extracted_marker_filepath: Optional[str] = None
) -> str:
    extract_path = extract_path if extract_path else os.getcwd()
    if extracted_marker_filepath and os.path.exists(extracted_marker_filepath):
        return extract_path
    output_filepath = download(url, output_filepath=output_filepath)
    extract_zip(output_filepath, extract_path=extract_path, filename_predicate=filename_predicate)
    if extracted_marker_filepath:
        with open(extracted_marker_filepath, "w") as f:
            f.write("")
    return extract_path


def main():
    parser = argparse.ArgumentParser("DocLayNet dataset downloader and extractor")
    parser.add_argument("--download-path", type=str, default="/home/i/dataset/DocLayNet/raw",
                        help="Directory for the raw dataset (default: %(default)s)")
    parser.add_argument("--extract-path", type=str, default="/home/i/dataset/DocLayNet/raw/DocLayNet",
                        help="Directory for the processed dataset (default: %(default)s)")
    args = parser.parse_args()

    os.makedirs(args.extract_path, exist_ok=True)
    download_extract_cached(
        "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip",
        output_filepath=os.path.join(args.download_path, "DocLayNet_core.zip"),
        extract_path=args.extract_path,
        # filename_predicate=lambda filename: (filename.startswith("COCO/")),
        filename_predicate=lambda filename: (filename.startswith("COCO/")
                                             and not os.path.exists(os.path.join(args.extract_path, filename))),
        extracted_marker_filepath=os.path.join(args.extract_path, ".core_extracted.txt"),
    )
    download_extract_cached(
        "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_extra.zip",
        output_filepath=os.path.join(args.download_path, "DocLayNet_extra.zip"),
        extract_path=args.extract_path,
        # filename_predicate=lambda filename: (filename.startswith("PDF/")),
        filename_predicate=lambda filename: (filename.startswith("PDF/")
                                             and not os.path.exists(os.path.join(args.extract_path, filename))),
        extracted_marker_filepath=os.path.join(args.extract_path, ".extra_extracted.txt"),
    )


if __name__ == '__main__':
    main()
