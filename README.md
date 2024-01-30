# GLAM

Graph-based Layout Analysis Model (GLAM) is a deep learning model for document layout analysis.

Unofficial implementation in PyTorch of "A Graphical Approach to Document Layout Analysis" [[arXiv]](https://arxiv.org/abs/2308.02051).

## Introduction

The Graph-based Layout Analysis Model (GLAM) is a novel deep learning model designed for advanced document layout analysis. This repository contains an unofficial PyTorch implementation of the model as described in the paper "A Graphical Approach to Document Layout Analysis". You can find the original paper [here](https://arxiv.org/abs/2308.02051).

Retrieval Augmented Generation (RAG) tasks represent a significant advancement in the field of large language models, focusing on enhancing model performance by integrating external knowledge sources. However, a fundamental challenge in these tasks arises from the processing of PDF files. Unlike standard text documents, PDFs are composed of positioned font glyphs that often lack labels, making them inherently unstructured. Traditional methods like image embedding or OCR can extract content, but they fall short in organizing it into meaningful structures, such as differentiating titles from paragraphs, and tables from figures. This is where GLAM comes into play. It bridges this gap by converting the unstructured content of PDFs into structured data, enabling the efficient use of such information in RAG tasks. With GLAM, the barrier of transforming complex PDF content into an organized format suitable for large language models is effectively removed, paving the way for more sophisticated and informed data retrieval and generation processes.

## Prerequisites

- Python 3.6+
- pip
- Optional: Tesseract or EasyOCR
- Optional: Git with [Git LFS](https://git-lfs.github.com/) support

### Ubuntu/Debian

```shell
apt-get update -q -y
apt-get install -q -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu tesseract-ocr-fra tesseract-ocr-jpn
python -m pip install -q -U -r requirements.txt
TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
```

## Dataset preparation

Download and extract DocLayNet dataset:

```shell
python dln_download_and_extract.py --download-path /home/i/dataset/DocLayNet/raw --extract-path /home/i/dataset/DocLayNet/raw/DocLayNet
```

Make own DocLayNet-v1.1, [free from bugs](https://huggingface.co/datasets/ds4sd/DocLayNet-v1.1/discussions/1), parsing spans with unlabelled glyphs with Tesseract:

```shell
python dln_parse_pdf.py --dataset-path /home/i/dataset/DocLayNet/raw/DocLayNet --image-scale 1
```

Make training examples:

```shell
python dln_glam_prepare.py --dataset-path /home/i/dataset/DocLayNet/raw/DocLayNet/DATA --output-path /home/i/dataset/DocLayNet/glam
```

## Training

Some paths are hardcoded in `dln_glam_train.py`. Please, change them before training.

```shell
python dln_glam_train.py
```

## Evaluation

Please, change paths in `dln_glam_evaluate.py` before evaluation.

```shell
python dln_glam_inference.py
```

## Features

- Simple architecture.
- Fast. With batch size of 128 examples it takes 00:11:35 for training on 507 batches and 00:02:17 for validation on 48 batches on CPU per 1 epoch.

## Limitations

- No reading order prediction, though it is not objective of this model, and dataset does not contain such information.

## TODO

- Implement mAP@IoU\[0.5:0.05:0.95] metric because there is no way to compare with other models yet.
- Implement input features normalization.
- Implement text and image features.
- Batching in inference. Currently, only one page is processed at a time.
- W&B integration for training.
- Some text spans in PDF contains unlabelled font glyphs. Currently, whole span is passed to OCR. It is faster to OCR font glyphs separately and then merge them into spans.

## Alternatives

* [Kensho Extract](https://kensho.com/extract) (GLAM author's SaaS closed-source implementation)
* [Unstructured](https://github.com/Unstructured-IO/unstructured)

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.

## Acknowledgements

- Jilin Wang, Michael Krumdick, Baojia Tong, Hamima Halim, Maxim Sokolov, Vadym Barda, Delphine Vendryes, and Chris Tanner. "A Graphical Approach to Document Layout Analysis". 2023. arXiv: [2308.02051](https://arxiv.org/abs/2308.02051)
