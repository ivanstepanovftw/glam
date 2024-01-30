import argparse
import io
import json
import logging
import multiprocessing
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import Mapping, Optional, Any, Iterable

import networkx as nx
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torch_geometric
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, box
from torch_geometric.data import Data
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from GLAM.common import PageNodes, PageEdges, TextNode, ImageNode


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler())


CLASSES_MAP = {
    0: "Unknown",
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


def iou(polygon1: Polygon, polygon2: Polygon) -> float:
    """Calculate intersection over union between two polygons. May return NaN if polygons are invalid."""
    intersection = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    iou = intersection / union
    return iou


# # @lru_cache(maxsize=1)
# def compare_segmentations(segmentations1, segmentations2):
#     """Function to compare segmentations and return a similarity score"""
#     if not segmentations1 or not segmentations2:
#         return 0  # No segmentations to compare
#
#     max_similarity = 0
#     for seg1 in segmentations1:
#         polygon1 = Polygon([(seg1[i], seg1[i + 1]) for i in range(0, len(seg1), 2)])
#         for seg2 in segmentations2:
#             polygon2 = Polygon([(seg2[i], seg2[i + 1]) for i in range(0, len(seg2), 2)])
#             max_similarity = max(max_similarity, compare_polygons(polygon1, polygon2))
#
#     return max_similarity


class DLNDataset(torch_geometric.data.Dataset):
    index_to_example_filename: Mapping[int, str]

    def __init__(self, root, split_name, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

        self.split_name = split_name
        image_id_to_example_filename = {
            int(filename.split(".")[0]): filename
            for filename in os.listdir(os.path.join(self.root, split_name))
        }
        self.index_to_example_filename = {
            i: image_id_to_example_filename[image_id]
            for i, image_id in enumerate(sorted(image_id_to_example_filename))
        }

    # @property
    # def raw_file_names(self):
    #     return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return tuple(self.index_to_example_filename.values())
    #
    # def download(self):
    #     # Download to `self.raw_dir`.
    #     path = download_url(url, self.raw_dir)
    #     ...
    #
    # def process(self):
    #     idx = 0
    #     for raw_path in self.raw_paths:
    #         # Read data from `raw_path`.
    #         data = Data(...)
    #
    #         if self.pre_filter is not None and not self.pre_filter(data):
    #             continue
    #
    #         if self.pre_transform is not None:
    #             data = self.pre_transform(data)
    #
    #         torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
    #         idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get_filepath(self, idx):
        example_filename = self.index_to_example_filename[idx]
        return os.path.join(self.root, self.split_name, example_filename)

    def get(self, idx):
        example = torch.load(self.get_filepath(idx))
        # assert example.node_features.isfinite().all()
        # assert example.edge_features.isfinite().all()
        return example


def process_row(dataset_dir, output_dir, split_name, image_id) -> Optional[Data]:
    id_dir = os.path.join(dataset_dir, split_name, "by-id", str(image_id))
    pdf_filepath = os.path.join(id_dir, "page.pdf")
    image_filepath = os.path.join(id_dir, "image.webp")
    row_filepath = os.path.join(id_dir, "row.json")
    annotations_filepath = os.path.join(id_dir, "annotations.json")
    page_dict_filepath = os.path.join(id_dir, "page_dict.pkl")

    with open(row_filepath, "rb") as f:
        row = json.load(f)

    with open(annotations_filepath, "rb") as f:
        annotations = json.load(f)

    with open(page_dict_filepath, "rb") as f:
        page_dict = pickle.load(f)

    # with open(image_filepath, "rb") as f:
    #     image = Image.open(f).convert("L").convert("RGBA")

    nodes = PageNodes()
    for block in page_dict["blocks"]:
        if block["type"] == 0:
            for line in block["lines"]:
                for span in line["spans"]:
                    nodes.append(TextNode.from_span(span))
        elif block["type"] == 1:
            try:
                nodes.append(ImageNode.from_page_block(block))
            except ValueError as e:
                logger.warning(f"{split_name}/{image_id}: Could not parse image block {block}: {e}")
        else:
            raise ValueError(f"{split_name}/{image_id}: Unknown block type {block['type']}")

    if len(nodes) <= 0:
        logger.warning(f"{split_name}/{image_id}: Skipping: No nodes found")
        return

    if len(nodes) == 1:
        logger.warning(f"{split_name}/{image_id}: Skipping: Only one node found, cannot make edges")
        return

    if len(nodes) > 1024:
        logger.warning(f"{split_name}/{image_id}: Skipping: Too many nodes ({len(nodes)}), slow to process")
        return

    edges = PageEdges.from_page_nodes_by_directions(nodes, k=31)
    # edges = PageEdges.from_page_nodes_by_top_closest(nodes, k=4+1)

    segmentations: list[Polygon] = [
        Polygon([
            (segmentation[i1], segmentation[i1 + 1])
            for i1 in range(0, len(segmentation), 2)
        ])
        for annotation in annotations
        for segmentation in annotation["segmentation"]
    ]

    # Calculate probabilities for each class for each node. This is a target for the node classification model.
    node_probs = torch.zeros(len(nodes), len(CLASSES_MAP), dtype=torch.float32)
    node_segmentations: dict[int, list[int]] = {}  # Mapping from node index to a list of segmentation indices
    uncovered_segmentations = set(range(len(segmentations)))  # Set of uncovered segmentations for cleaning dataset
    for node_i, node in enumerate(nodes):
        # Get bounding box
        node_bbox = box(node.bbox_min_x, node.bbox_min_y, node.bbox_max_x, node.bbox_max_y)
        if node_bbox.area <= 0:
            logger.warning(f"{split_name}/{image_id}: Node {node} has zero area.")
            continue

        # Iterate over the segmentations
        segmentation_i = 0
        for annotation in annotations:
            for _ in annotation["segmentation"]:
                overlap_area = segmentations[segmentation_i].intersection(node_bbox).area
                if overlap_area > 0:
                    # Weighted votes by the proportion of overlap
                    node_probs[node_i][annotation["category_id"]] += overlap_area / node_bbox.area
                    node_segmentations.setdefault(node_i, []).append(segmentation_i)
                    uncovered_segmentations.discard(segmentation_i)
                segmentation_i += 1

    # Normalize node_probs by the number of segmentations for each node
    node_probs /= torch.tensor([len(node_segmentations.get(node_i, [None])) for node_i in range(len(nodes))], dtype=torch.float32).unsqueeze(1)
    # if node_probs.max() < 0.95:
    #     logger.debug(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11")
    #     logger.debug(f"node_probs: {node_probs.min():.2f}, {node_probs.max():.2f}, {node_probs.mean():.2f}, {node_probs.std():.2f}")

    unlabelled_nodes = [node_i for node_i, node in enumerate(nodes) if node_i not in node_segmentations]

    # # Debug draw
    # if unlabelled_nodes or uncovered_segmentations:
    #     logger.warning(f"pdf_filepath: {pdf_filepath}, unlabelled_nodes: {unlabelled_nodes}, uncovered_segmentations: {uncovered_segmentations}")
    #     # Render page
    #     #image = Image.open(io.BytesIO(data["image"]["bytes"])).convert("L").convert("RGBA")
    #     overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    #     draw = ImageDraw.Draw(overlay)
    #
    #     # Render segmentation
    #     for annotation in annotations:
    #         segmentation = annotation["segmentation"][0]
    #         segmentation_polygon = Polygon((
    #             (segmentation[i], segmentation[i + 1])
    #             for i in range(0, len(segmentation), 2)
    #         ))
    #         draw.polygon(segmentation_polygon.exterior.coords, fill=(200, 200, 200, 200), outline="magenta", width=4)
    #         # area = segmentation_polygon.intersection(node_bbox).area
    #         # print("area:", area, "node_bbox.area:", node_bbox.area, "area / node_bbox.area:", area / node_bbox.area)
    #
    #     image = Image.alpha_composite(image, overlay)
    #     draw = ImageDraw.Draw(image)
    #
    #     # Render bad nodes
    #     for node_i in nodes:
    #         outline = "green" if isinstance(node_i, TextNode) else "blue"
    #         draw.rectangle((node_i.bbox_min_x, node_i.bbox_min_y, node_i.bbox_max_x, node_i.bbox_max_y), outline=outline, width=1)
    #     for node_i in unlabelled_nodes:
    #         outline = "red" if isinstance(node_i, TextNode) else "orange"
    #         draw.rectangle((node_i.bbox_min_x, node_i.bbox_min_y, node_i.bbox_max_x, node_i.bbox_max_y), outline=outline, width=1)
    #
    #     image.show()
    #     print("showing", len(unlabelled_nodes) / len(nodes))
    #     # breakpoint()
    #     # time.sleep(3)

    if len(unlabelled_nodes) >= 5:
        logger.warning(f"{split_name}/{image_id}: Skipping: Too many # unlabelled nodes ({len(unlabelled_nodes)}, {len(unlabelled_nodes) / len(nodes):.2f}%)")
        return

    if len(unlabelled_nodes) >= 3 and len(unlabelled_nodes) / len(nodes) > 0.05:
        logger.warning(f"{split_name}/{image_id}: Skipping: Too many % unlabelled nodes ({len(unlabelled_nodes)}, {len(unlabelled_nodes) / len(nodes):.2f}%)")
        return

    node_features = nodes.to_node_features()
    edge_index = edges.to_edge_index().t()
    edge_features = edges.to_edge_features()

    # Calculate probability of same segmentation for each edge. This is a target for the edge classification model.
    edge_probs = torch.zeros(edge_index.shape[1])
    edge_connections: dict[int, list[tuple[int, int]]] = {}  # Mapping from edge index to a list of (segmentation_i1, segmentation_i2) tuples

    def precompute_iou(polygons):
        n = len(polygons)
        ious = np.zeros((n, n))
        for i in range(n):
            ious[i, i] = 1
            for j in range(i + 1, n):
                ious[i, j] = ious[j, i] = iou(polygons[i], polygons[j])
        return ious

    segmentation_ious = precompute_iou(segmentations)

    # Iterate over the edges to label them
    for k in range(edge_index.shape[1]):
        src_node_i = edge_index[0, k].item()
        dst_node_i = edge_index[1, k].item()

        # Get segmentations for both nodes
        src_node_segmentations = node_segmentations.get(src_node_i, [])
        dst_node_segmentations = node_segmentations.get(dst_node_i, [])

        # Calculate similarity between segmentations
        for src_node_segmentation in src_node_segmentations:
            for dst_node_segmentation in dst_node_segmentations:
                segmentation_iou = segmentation_ious[src_node_segmentation, dst_node_segmentation].item()
                if segmentation_iou > 0:
                    edge_probs[k] += segmentation_iou
                    edge_connections.setdefault(k, []).append((src_node_segmentation, dst_node_segmentation))

    # Normalize edge_probs
    edge_probs /= torch.tensor([len(edge_connections.get(k, [None])) for k in range(edge_index.shape[1])], dtype=torch.float32)
    # logger.debug(f"edge_probs: {edge_probs.min():.2f}, {edge_probs.max():.2f}, {edge_probs.mean():.2f}, {edge_probs.std():.2f}")

    example = Data(
        split_name=split_name,  # metadata
        image_id=image_id,      # metadata
        node_features=node_features,  # input
        edge_index=edge_index,        # input
        edge_features=edge_features,  # input
        node_probs=node_probs,  # target
        edge_probs=edge_probs,  # target
    )

    #########################################
    # Calculate mAP@IoU[0.5:0.95:0.05]
    #########################################
    # def single_image_evaluation(bbox_preds, class_preds, bbox_gts, class_gts, iou_threshold):
    #     """Evaluate a single image for precision and recall at a specific IoU threshold."""
    #     tp = 0
    #     fp = 0
    #     gt_used = [False] * len(bbox_gts)
    #     for bbox_pred, class_pred in zip(bbox_preds, class_preds):
    #         matched = False
    #         for i, (bbox_gt, class_gt) in enumerate(zip(bbox_gts, class_gts)):
    #             if gt_used[i] or class_pred != class_gt:
    #                 continue
    #             if iou(bbox_pred, bbox_gt) >= iou_threshold:
    #                 gt_used[i] = True
    #                 tp += 1
    #                 matched = True
    #                 break
    #         if not matched:
    #             fp += 1
    #
    #     fn = sum(1 for used in gt_used if not used)
    #     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    #     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    #     return precision, recall
    #
    # def mean_average_precision(bbox_preds, class_preds, bbox_gts, class_gts, iou_thresholds: Iterable[float] = np.arange(0.5, 1, 0.05)):
    #     """Calculate the mean Average Precision over a range of IoU thresholds."""
    #     average_precisions = []
    #     for iou_threshold in iou_thresholds:
    #         precisions = []
    #         recalls = []
    #         # for preds, gts in zip(predictions, ground_truths):
    #         for bbox_pred, class_pred, bbox_gt, class_gt in zip(bbox_preds, class_preds, bbox_gts, class_gts):
    #             precision, recall = single_image_evaluation(bbox_preds, class_preds, bbox_gts, class_gts, iou_threshold)
    #             precisions.append(precision)
    #             recalls.append(recall)
    #         ap = np.mean(precisions)  # Average precision for this IoU threshold
    #         average_precisions.append(ap)
    #     return np.mean(average_precisions)  # Mean over all IoU thresholds

    # edge_prob_threshold = 0.5
    # graph = nx.Graph()
    # for k in range(example.edge_index.shape[1]):
    #     src_node_i = example.edge_index[0, k].item()
    #     dst_node_i = example.edge_index[1, k].item()
    #     edge_prob = example.edge_probs[k].item()
    #
    #     if edge_prob >= edge_prob_threshold:
    #         graph.add_edge(src_node_i, dst_node_i, weight=edge_prob)
    #     else:
    #         graph.add_node(src_node_i)
    #         graph.add_node(dst_node_i)
    #
    # clusters: list[set[int]] = list(nx.connected_components(graph))
    # # cluster_min_spanning_boxes: list[Polygon] = [
    # #     Polygon([
    # #         (min(nodes[node_i].bbox_min_x for node_i in cluster), min(nodes[node_i].bbox_min_y for node_i in cluster)),
    # #         (max(nodes[node_i].bbox_max_x for node_i in cluster), min(nodes[node_i].bbox_min_y for node_i in cluster)),
    # #         (max(nodes[node_i].bbox_max_x for node_i in cluster), max(nodes[node_i].bbox_max_y for node_i in cluster)),
    # #         (min(nodes[node_i].bbox_min_x for node_i in cluster), max(nodes[node_i].bbox_max_y for node_i in cluster)),
    # #     ])
    # #     for cluster in clusters
    # # ]
    # cluster_classes: list[int] = torch.stack([example.node_probs[torch.tensor(list(cluster))].sum(dim=0) for cluster in clusters]).argmax(dim=1).tolist()
    #
    # node_class_accuracy = 0
    # for cluster, cluster_class in zip(clusters, cluster_classes):
    #     for node_index in cluster:
    #         node_class = example.node_probs[node_index].argmax().item()
    #         if node_class == cluster_class:
    #             node_class_accuracy += 1
    # node_class_accuracy /= len(nodes)
    # logger.debug(f"{split_name}/{image_id}: node_class_accuracy: {node_class_accuracy:.2f}, len(unlabelled_nodes): {len(unlabelled_nodes)}, len(nodes): {len(nodes)}, len(uncovered_segmentations): {len(uncovered_segmentations)}, len(annotation): {len(annotations)}, len(segmentations): {len(segmentations)}, len(clusters): {len(clusters)}")

    # # Render page
    # if node_class_accuracy < 0.98:
    #     draw = ImageDraw.Draw(image)
    #
    #     # for annotation in annotations:
    #     #     for segmentation in annotation["segmentation"]:
    #     #         draw.polygon(segmentation, outline=(0, 0, 255), width=6)
    #
    #     for cluster, cluster_class in zip(clusters, cluster_classes):
    #         cluster_bbox = (
    #             min(example.node_features[node_i][2] for node_i in cluster),
    #             min(example.node_features[node_i][3] for node_i in cluster),
    #             max(example.node_features[node_i][4] for node_i in cluster),
    #             max(example.node_features[node_i][5] for node_i in cluster),
    #         )
    #         draw.rectangle(cluster_bbox, outline=(0, 255, 0), width=3)
    #         draw.text(cluster_bbox[:2], CLASSES_MAP[cluster_class], fill=(0, 0, 0))
    #
    #     # for k, node_features in zip(range(example.node_features.size(0)), example.node_features):
    #     #     node_bbox = (node_features[2], node_features[3], node_features[4], node_features[5])
    #     #     draw.rectangle(node_bbox, outline=(255, 0, 0), width=1)
    #     #
    #     # for annotation in annotations:
    #     #     for segmentation in annotation["segmentation"]:
    #     #         draw.text(segmentation, CLASSES_MAP[annotation["category_id"]], fill=(0, 0, 0))
    #
    #     logger.debug(f"{split_name}/{image_id}: node_class_accuracy: {node_class_accuracy}")
    #     image.show(title=f"{split_name}/{image_id}")
    #     breakpoint()

    return example


def process(dataset_dir, output_dir, split_name, image_id) -> tuple[Optional[Data], Any]:
    return process_row(dataset_dir, output_dir, split_name, image_id), (dataset_dir, output_dir, split_name, image_id)


def main():
    parser = argparse.ArgumentParser("DocLayNet dataset")
    parser.add_argument("--dataset-path", type=str, default="/home/i/dataset/DocLayNet/raw/DocLayNet/DATA",
                        help="Directory for the raw dataset (default: %(default)s)")
    parser.add_argument("--output-path", type=str, default="/home/i/dataset/DocLayNet/glam",
                        help="Directory for the processed dataset (default: %(default)s)")
    args = parser.parse_args()

    split_names = ["train", "test", "val"]
    # split_names = ["val"]
    split_image_ids = {}

    for split_name in split_names:
        image_ids = os.listdir(os.path.join(args.dataset_path, split_name, "by-id"))
        image_ids = [int(x.split(".")[0]) for x in image_ids]
        image_ids = sorted(image_ids)
        split_image_ids[split_name] = image_ids

    num_processes = psutil.cpu_count(logical=False)
    # num_processes = 1
    logger.debug(f"Using {num_processes} processes.")
    tasks_in_pool = 0
    max_tasks_in_pool = 100 + num_processes

    pbar = tqdm(desc=f"Processing...", total=sum(len(image_ids) for image_ids in split_image_ids.values()), smoothing=0.001, position=0, leave=False)

    with logging_redirect_tqdm(), multiprocessing.Pool(num_processes) as pool:
        def callback(result):
            nonlocal tasks_in_pool
            tasks_in_pool -= 1
            pbar.update(1)

            example, context = result
            dataset_dir, output_path, split_name, image_id = context

            if not example:
                return

            assert example.node_features.isfinite().all()
            assert example.edge_features.isfinite().all()

            example_filepath = os.path.join(output_path, f'{image_id}.pt')
            torch.save(example, example_filepath)

        def my_error_callback(e):
            nonlocal tasks_in_pool
            tasks_in_pool -= 1
            pbar.update(1)
            # logger.exception(e)

        for split_name in split_names:
            output_path = os.path.join(args.output_path, split_name)
            os.makedirs(output_path, exist_ok=True)

            image_ids = split_image_ids[split_name]
            for image_id in image_ids:
                example_filepath = os.path.join(output_path, f'{image_id}.pt')
                if os.path.exists(example_filepath):
                    pbar.update(1)
                    # pbar.total -= 1
                    continue

                while tasks_in_pool >= max_tasks_in_pool:
                    time.sleep(0.1)

                tasks_in_pool += 1
                pool.apply_async(process, args=(args.dataset_path, output_path, split_name, image_id), callback=callback, error_callback=my_error_callback)
                # callback(process(args.dataset_path, output_path, split_name, image_id))

        while tasks_in_pool > 0:
            pbar.refresh()
            print("Tasks in pool:", tasks_in_pool)
            print("Waiting for following tasks:")
            # print(pool._cache)
            print(pool._taskqueue)
            time.sleep(1)

        pool.close()
        pool.join()

    pbar.refresh()
    pbar.close()

    print("Done.")


if __name__ == '__main__':
    main()
