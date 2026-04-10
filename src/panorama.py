import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class PanoramaResult:
    combo_name: str
    output_path: str
    elapsed_seconds: float
    num_keypoints_img1: int
    num_keypoints_img2: int
    num_good_matches: int
    num_inliers: int
    inlier_ratio: float
    quality_note: str


def _create_detector(detector_name: str):
    name = detector_name.upper()
    if name == "ORB":
        return cv2.ORB_create(nfeatures=4000)
    if name == "SIFT":
        return cv2.SIFT_create()
    raise ValueError(f"Detector invalido: {detector_name}")


def _create_matcher(detector_name: str, matcher_name: str):
    d = detector_name.upper()
    m = matcher_name.upper()

    if m == "BF":
        if d == "SIFT":
            return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    if m == "FLANN":
        if d == "SIFT":
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)

        index_params = dict(
            algorithm=6,
            table_number=6,
            key_size=12,
            multi_probe_level=1,
        )
        search_params = dict(checks=50)
        return cv2.FlannBasedMatcher(index_params, search_params)

    raise ValueError(f"Matcher invalido: {matcher_name}")


def _ratio_test(matches, ratio: float = 0.75):
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def _warp_and_blend(img1: np.ndarray, img2: np.ndarray, H: np.ndarray):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    warped_corners_img2 = cv2.perspectiveTransform(corners_img2, H)
    all_corners = np.concatenate((corners_img1, warped_corners_img2), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = [-x_min, -y_min]
    T = np.array(
        [[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]],
        dtype=np.float64,
    )

    stitched = cv2.warpPerspective(img2, T.dot(H), (x_max - x_min, y_max - y_min))
    stitched[translation[1] : h1 + translation[1], translation[0] : w1 + translation[0]] = img1
    return stitched


def _quality_from_metrics(inlier_ratio: float, num_inliers: int) -> str:
    score = inlier_ratio * 100.0 + 0.05 * num_inliers
    if score >= 70:
        return "Alta"
    if score >= 45:
        return "Media"
    return "Baixa"


def stitch_images(
    img1: np.ndarray,
    img2: np.ndarray,
    detector_name: str,
    matcher_name: str,
    output_dir: str,
) -> PanoramaResult:
    detector = _create_detector(detector_name)
    matcher = _create_matcher(detector_name, matcher_name)

    start = time.perf_counter()

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        raise RuntimeError("Nao foi possivel detectar pontos suficientes nas imagens.")

    if detector_name.upper() == "ORB" and matcher_name.upper() == "FLANN":
        des1 = np.uint8(des1)
        des2 = np.uint8(des2)

    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = _ratio_test(matches, ratio=0.75)

    if len(good_matches) < 4:
        raise RuntimeError("Nao ha correspondencias suficientes para calcular homografia.")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        raise RuntimeError("Falha ao estimar homografia.")

    panorama = _warp_and_blend(img1, img2, H)

    elapsed = time.perf_counter() - start

    combo = f"{detector_name.upper()}_{matcher_name.upper()}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"panorama_{combo}.jpg")
    cv2.imwrite(output_path, panorama)

    num_inliers = int(mask.ravel().sum())
    inlier_ratio = num_inliers / max(len(good_matches), 1)
    quality_note = _quality_from_metrics(inlier_ratio, num_inliers)

    return PanoramaResult(
        combo_name=combo,
        output_path=output_path,
        elapsed_seconds=elapsed,
        num_keypoints_img1=len(kp1),
        num_keypoints_img2=len(kp2),
        num_good_matches=len(good_matches),
        num_inliers=num_inliers,
        inlier_ratio=inlier_ratio,
        quality_note=quality_note,
    )


def run_all_combinations(
    img1_path: str,
    img2_path: str,
    output_dir: str,
) -> List[PanoramaResult]:
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None:
        raise FileNotFoundError(f"Imagem 1 nao encontrada: {img1_path}")
    if img2 is None:
        raise FileNotFoundError(f"Imagem 2 nao encontrada: {img2_path}")

    combinations: List[Tuple[str, str]] = [
        ("ORB", "BF"),
        ("ORB", "FLANN"),
        ("SIFT", "BF"),
        ("SIFT", "FLANN"),
    ]

    results: List[PanoramaResult] = []
    for detector, matcher in combinations:
        result = stitch_images(
            img1=img1,
            img2=img2,
            detector_name=detector,
            matcher_name=matcher,
            output_dir=output_dir,
        )
        results.append(result)

    return results


def format_results_table(results: List[PanoramaResult]) -> str:
    header = (
        "Combinacao           | Tempo (s) | KP Img1 | KP Img2 | Matches | Inliers | Inlier Ratio\n"
        "---------------------+-----------+---------+---------+---------+---------+-------------"
    )
    rows = []
    for r in results:
        rows.append(
            f"{r.combo_name:<21}| {r.elapsed_seconds:>9.4f} | {r.num_keypoints_img1:>7} | {r.num_keypoints_img2:>7} | "
            f"{r.num_good_matches:>7} | {r.num_inliers:>7} | {r.inlier_ratio:>11.4f}"
        )
    return header + "\n" + "\n".join(rows)
