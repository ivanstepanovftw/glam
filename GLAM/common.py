from scipy.spatial import KDTree
import logging
import math
import re
import time
from collections import namedtuple
from dataclasses import dataclass
from typing import NamedTuple, Sequence, MutableSequence, Optional, Generator, Iterable, Any, Mapping

import cv2
import easyocr
import numpy as np
import tesserocr
import torch
from tesserocr import PyTessBaseAPI

import fitz  # PyMuPDF
from PIL import Image, ImageDraw


INVALID_UNICODE = chr(0xFFFD)
EasyocrTextResult = namedtuple("EasyocrTextResult", ["bbox", "text", "confidence"])
MuPDFTextTraceChar = namedtuple("MuPDFTextTraceChar", ["unicode", "glyph", "origin", "bbox"])
logger = logging.getLogger(__name__)
HI_RES_MATRIX = fitz.Matrix(5, 5)
# HI_RES_MATRIX = fitz.Matrix(1, 1)  # enough for tesseract in english


def truncate_with_ellipsis(s: str, max_length=128):
    return (s[:max_length - 3] + '...') if len(s) > max_length else s


def get_color_mode(colorspace, alpha: bool, unmultiply: bool = False) -> str:
    if colorspace is None:
        return "L"
    if colorspace.n == 1:
        return "LA" if alpha else "L"
    if colorspace.n == 3:
        return "RGBa" if (alpha and unmultiply) else "RGBA" if alpha else "RGB"
    return "CMYK"


def get_bytes_per_pixel(colorspace, alpha: bool) -> int:
    if colorspace is None:
        return 1
    if colorspace.n == 1:
        return 1 + int(alpha)
    if colorspace.n == 3:
        return 3 + int(alpha)
    return 4


def pixmap_to_image(pixmap: fitz.Pixmap, unmultiply: bool = False, mode: Optional[str] = None) -> Image.Image:
    """
    Makes a view of the pixmap samples as a PIL image, copies if colorspace is odd.
    """
    mode = mode or get_color_mode(pixmap.colorspace, pixmap.alpha, unmultiply)
    image = Image.frombuffer(mode, (int(pixmap.w), int(pixmap.h)), pixmap.samples, "raw", mode, 0, 1)
    return image


def pixmap_to_ndarray(pixmap: fitz.Pixmap) -> np.ndarray:
    """
    Makes a view of the pixmap samples as a numpy array.
    """
    bpp = get_bytes_per_pixel(pixmap.colorspace, pixmap.alpha)
    image_data = np.frombuffer(pixmap.samples, dtype=np.uint8)
    if bpp == 1:
        image_data = image_data.reshape((pixmap.height, pixmap.stride))
    else:
        image_data = image_data.reshape((pixmap.height, pixmap.stride // bpp, bpp))
    # image_data = image_data[:, :pixmap.width, ...]
    return image_data


def image_to_ndarray(image: Image.Image) -> np.ndarray:
    """
    Makes a view of the image samples as a numpy array.
    """
    image_data = np.frombuffer(image.tobytes(), dtype=np.uint8)
    image_data = image_data.reshape((image.height, image.width, 4))
    return image_data


# Frozen is used to implement __hash__ and __eq__
@dataclass(frozen=True)
class Node:
    bbox_min_x: float
    bbox_min_y: float
    bbox_max_x: float
    bbox_max_y: float

    def centroid(self) -> tuple[float, float]:
        return (
            (self.bbox_min_x + self.bbox_max_x) / 2,
            (self.bbox_min_y + self.bbox_max_y) / 2,
        )


@dataclass(frozen=True)
class TextNode(Node):
    """
    Text node in the GLAM graph.
    """
    text: str
    font_name: str
    font_size: float
    font_color_r: float  # 0.0-1.0
    font_color_g: float  # 0.0-1.0
    font_color_b: float  # 0.0-1.0

    @classmethod
    def from_span(
            cls,
            span: dict,  # span from PyMuPDF page.get_text("dict")["blocks"][...]["lines"][...]["spans"][...]
            *,
            text: Optional[str] = None,
    ) -> "TextNode":
        return cls(
            bbox_min_x=span["bbox"][0],
            bbox_min_y=span["bbox"][1],
            bbox_max_x=span["bbox"][2],
            bbox_max_y=span["bbox"][3],
            text=text or span["text"],
            font_name=span["font"],
            font_size=span["size"],
            font_color_r=((span["color"]) & 0xFF) / 255,  # TODO: check if correct order
            font_color_g=((span["color"] >> 8) & 0xFF) / 255,
            font_color_b=((span["color"] >> 16) & 0xFF) / 255,
        )


@dataclass(frozen=True)
class ImageNode(Node):
    """
    Image node in the GLAM graph.
    """
    image: Optional[Image] = None

    @classmethod
    def from_page_block(
            cls,
            page_block: Mapping[str, Any],  # page_block from PyMuPDF page.get_text("dict")["blocks"][...]
    ) -> "ImageNode":
        return cls(
            bbox_min_x=page_block["bbox"][0],
            bbox_min_y=page_block["bbox"][1],
            bbox_max_x=page_block["bbox"][2],
            bbox_max_y=page_block["bbox"][3],
            image=None  # TODO: implement
        )


class Edge(NamedTuple):
    """
    Edge in the GLAM graph.
    """
    node_index_1: int
    node_index_2: int
    centroid_distance: float  # pixels
    centroid_angle: float  # radians
    ordered_hint: int  # 0-1

    @classmethod
    def from_node_pair(
            cls,
            node_1: Node,
            node_2: Node,
            node_index_1: int,
            node_index_2: int,
            ordered_hint: int,
    ) -> "Edge":
        node_1_centroid = node_1.centroid()
        node_2_centroid = node_2.centroid()
        delta_x = node_2_centroid[0] - node_1_centroid[0]
        delta_y = node_2_centroid[1] - node_1_centroid[1]
        centroid_distance = math.hypot(delta_x, delta_y)
        centroid_angle = math.atan2(delta_y, delta_x)

        return cls(
            node_index_1=node_index_1,
            node_index_2=node_index_2,
            centroid_distance=centroid_distance,
            centroid_angle=centroid_angle,
            ordered_hint=ordered_hint,
        )

    def copy_inverted(self) -> "Edge":
        return Edge(
            node_index_1=self.node_index_2,
            node_index_2=self.node_index_1,
            centroid_distance=self.centroid_distance,
            centroid_angle=self.centroid_angle + math.pi,
            ordered_hint=self.ordered_hint,
        )


class PageNodes(list[Node]):
    features_len = 14
    re_text_spaces = re.compile(r"\s+")

    def to_node_features(self) -> torch.Tensor:
        node_list = [
            [
                # Node - type
                int(isinstance(node, TextNode)),
                int(isinstance(node, ImageNode)),
                # Node
                node.bbox_min_x,
                node.bbox_min_y,
                node.bbox_max_x,
                node.bbox_max_y,
                # TextNode
                node.font_color_r if isinstance(node, TextNode) else 0,
                node.font_color_g if isinstance(node, TextNode) else 0,
                node.font_color_b if isinstance(node, TextNode) else 0,
                node.font_size if isinstance(node, TextNode) else 0,
                "bold" in node.font_name.lower() if isinstance(node, TextNode) else 0,
                "italic" in node.font_name.lower() if isinstance(node, TextNode) else 0,
                len(node.text) if isinstance(node, TextNode) else 0,
                len(self.re_text_spaces.split(node.text)) if isinstance(node, TextNode) else 0,
                # ImageNode - nothing
            ]
            for node in self
        ]
        return torch.tensor(node_list, dtype=torch.float32)


def is_angle_in_range(angle, range):
    start, end = range
    if start > end:  # This handles the wrap-around case
        return start <= angle or angle < end
    else:
        return start <= angle < end


class PageEdges(list[Edge]):
    features_len = 3

    def to_edge_index(self) -> torch.Tensor:
        edge_list = [
            [edge.node_index_1, edge.node_index_2]
            for edge in self
        ]
        return torch.tensor(edge_list, dtype=torch.int64)

    def to_edge_features(self) -> torch.Tensor:
        return torch.tensor(
            [
                [
                    edge.centroid_distance,
                    edge.centroid_angle,
                    edge.ordered_hint,
                ]
                for edge in self
            ],
            dtype=torch.float32,
        )

    @classmethod
    def from_page_nodes_as_complete_graph(cls, page_nodes) -> "PageEdges":
        page_edges = cls()
        for i in range(len(page_nodes)):
            for j in range(len(page_nodes)):
                if i == j:
                    continue
                edge = Edge.from_node_pair(
                    node_1=page_nodes[i],
                    node_2=page_nodes[j],
                    node_index_1=i,
                    node_index_2=j,
                    ordered_hint=int(i + 1 == j),
                )
                page_edges.extend([edge, edge.copy_inverted()])
        return page_edges

    @classmethod
    def from_page_nodes_by_top_closest(cls, page_nodes, always_has_next=True, k=10 + 1) -> "PageEdges":
        page_edges = cls()
        centroids = np.array([node.centroid() for node in page_nodes])
        k = min(len(centroids), k)
        tree = KDTree(centroids)

        for i, centroid in enumerate(centroids):
            if always_has_next and i + 1 < len(page_nodes):
                edge = Edge.from_node_pair(
                    node_1=page_nodes[i],
                    node_2=page_nodes[i + 1],
                    node_index_1=i,
                    node_index_2=i + 1,
                    ordered_hint=int(True),
                )
                page_edges.extend([edge, edge.copy_inverted()])

            distances, indices = tree.query(centroid, k=k)
            for j, distance in zip(indices, distances):
                if i == j or always_has_next and i + 1 == j:
                    continue

                delta_x = centroids[j][0] - centroid[0]
                delta_y = centroids[j][1] - centroid[1]
                centroid_angle = math.atan2(delta_y, delta_x)

                edge = Edge(
                    node_index_1=i,
                    node_index_2=j,
                    centroid_distance=distance,
                    centroid_angle=centroid_angle,
                    ordered_hint=int(i + 1 == j),
                )
                page_edges.extend([edge, edge.copy_inverted()])

        return page_edges

    @classmethod
    def from_page_nodes_by_directions(
            cls,
            page_nodes,
            always_has_next=True,
            # Left, Up, Right, Down (because in both PIL and MuPDF coordinate system Y axis is inverted)
            directions=((3 * math.pi / 4, -3 * math.pi / 4), (-3 * math.pi / 4, -1 * math.pi / 4),
                        (-1 * math.pi / 4, 1 * math.pi / 4), (1 * math.pi / 4, 3 * math.pi / 4)),
            k=10 + 1
    ) -> "PageEdges":
        page_edges = cls()
        centroids = np.array([node.centroid() for node in page_nodes])
        k = min(len(centroids), k)
        tree = KDTree(centroids)

        for i, centroid in enumerate(centroids):
            if always_has_next and i + 1 < len(page_nodes):
                edge = Edge.from_node_pair(
                    node_1=page_nodes[i],
                    node_2=page_nodes[i + 1],
                    node_index_1=i,
                    node_index_2=i + 1,
                    ordered_hint=int(True),
                )
                page_edges.extend([
                    edge,
                    edge.copy_inverted()
                ])

            closest_nodes = {dir_range: None for dir_range in directions}
            distances, indices = tree.query(centroid, k=k)

            for j, distance in zip(indices, distances):
                if i == j:
                    continue

                delta_x = centroids[j][0] - centroid[0]
                delta_y = centroids[j][1] - centroid[1]
                centroid_angle = math.atan2(delta_y, delta_x)  # -pi..pi

                # Check each direction
                for dir_range in directions:
                    if is_angle_in_range(centroid_angle, dir_range):
                        if closest_nodes[dir_range] is None or distance < closest_nodes[dir_range][1]:
                            closest_nodes[dir_range] = (j, distance, centroid_angle)

            # Create edges for each direction
            for dir_range, node_info in closest_nodes.items():
            # if True:
            #     dir_range = directions[1]
                node_info = closest_nodes[dir_range]

                if node_info is None:
                    continue

                j, distance, centroid_angle = node_info
                if always_has_next and i + 1 == j:
                    continue

                edge = Edge(
                    node_index_1=i,
                    node_index_2=j,
                    centroid_distance=distance,
                    centroid_angle=centroid_angle,
                    ordered_hint=int(i + 1 == j),
                )
                page_edges.extend([
                    edge,
                    edge.copy_inverted(),
                ])

        return page_edges
