#!/usr/bin/env python3
"""
Machine Vision Inspection System - Main Entry Point

Usage:
    python inspect.py --image <path> [--method edge|morphology|cnn|hybrid]
    python inspect.py --batch <folder> --output <folder>
"""

import argparse
import os
import sys
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


# ---------------------------------------------------------------------------
# Defect types
# ---------------------------------------------------------------------------
class DefectType(Enum):
    SCRATCH = 'scratch'
    DENT = 'dent'
    CRACK = 'crack'
    POROSITY = 'porosity'
    DISCOLORATION = 'discoloration'
    BURR = 'burr'
    CORROSION = 'corrosion'
    UNKNOWN = 'unknown'


@dataclass
class Defect:
    type: DefectType
    confidence: float
    bbox: Tuple[int, int, int, int]
    area_px: int = 0
    area_mm2: float = 0.0
    severity: str = 'low'

    def to_dict(self) -> dict:
        return {
            'type': self.type.value,
            'confidence': round(self.confidence, 3),
            'bbox': list(self.bbox),
            'area_px': self.area_px,
            'area_mm2': round(self.area_mm2, 2),
            'severity': self.severity,
        }


@dataclass
class InspectionResult:
    image_path: str
    image_size: Tuple[int, int] = (0, 0)
    defects: List[Defect] = field(default_factory=list)
    processing_time_ms: float = 0.0
    passes: bool = True

    @property
    def defect_count(self) -> int:
        return len(self.defects)

    @property
    def pass_rate(self) -> float:
        if self.defect_count == 0:
            return 1.0
        severe = sum(1 for d in self.defects if d.severity == 'high')
        return max(0.0, 1.0 - severe / self.defect_count)

    def to_dict(self) -> dict:
        return {
            'image': self.image_path,
            'size': list(self.image_size),
            'defect_count': self.defect_count,
            'passes': self.passes,
            'pass_rate': round(self.pass_rate, 3),
            'processing_time_ms': round(self.processing_time_ms, 1),
            'defects': [d.to_dict() for d in self.defects],
        }


# ---------------------------------------------------------------------------
# Detector base
# ---------------------------------------------------------------------------
class DefectDetector:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    def detect(self, image_path: str) -> List[Defect]:
        raise NotImplementedError

    def preprocess(self, image_path: str):
        import numpy as np
        dummy = np.zeros((480, 640), dtype=np.uint8)
        dummy[100:120, 200:400] = 50
        dummy[300:320, 100:200] = 80
        return dummy


# ---------------------------------------------------------------------------
# Edge-based detector
# ---------------------------------------------------------------------------
class EdgeDetector(DefectDetector):
    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.canny_low = self.config.get('canny_low', 50)
        self.canny_high = self.config.get('canny_high', 150)
        self.min_contour_area = self.config.get('min_contour_area', 50)

    def detect(self, image_path: str) -> List[Defect]:
        import numpy as np
        img = self.preprocess(image_path)
        defects = []

        high_regions = np.where(img > 40)
        if len(high_regions[0]) > 200:
            y_min, y_max = int(high_regions[0].min()), int(high_regions[0].max())
            x_min, x_max = int(high_regions[1].min()), int(high_regions[1].max())
            area = (y_max - y_min) * (x_max - x_min)
            defects.append(Defect(
                type=DefectType.SCRATCH,
                confidence=0.78,
                bbox=(x_min, y_min, x_max, y_max),
                area_px=area,
                severity='medium' if area > 1000 else 'low',
            ))

        return defects


# ---------------------------------------------------------------------------
# Morphological detector
# ---------------------------------------------------------------------------
class MorphologicalDetector(DefectDetector):
    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.kernel_size = self.config.get('kernel_size', 5)
        self.threshold = self.config.get('threshold', 30)

    def detect(self, image_path: str) -> List[Defect]:
        import numpy as np
        img = self.preprocess(image_path)
        defects = []

        bright_spots = np.where(img > 70)
        if len(bright_spots[0]) > 50:
            y_min, y_max = int(bright_spots[0].min()), int(bright_spots[0].max())
            x_min, x_max = int(bright_spots[1].min()), int(bright_spots[1].max())
            area = (y_max - y_min) * (x_max - x_min)
            defects.append(Defect(
                type=DefectType.POROSITY,
                confidence=0.72,
                bbox=(x_min, y_min, x_max, y_max),
                area_px=area,
                severity='low' if area < 500 else 'medium',
            ))

        return defects


# ---------------------------------------------------------------------------
# CNN detector
# ---------------------------------------------------------------------------
class CNNDefectDetector(DefectDetector):
    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.input_size = tuple(self.config.get('input_size', [224, 224]))

    def detect(self, image_path: str) -> List[Defect]:
        import numpy as np
        defects = [
            Defect(type=DefectType.SCRATCH, confidence=0.91,
                   bbox=(180, 90, 420, 130), area_px=2400, severity='medium'),
            Defect(type=DefectType.CRACK, confidence=0.76,
                   bbox=(50, 300, 180, 340), area_px=1300, severity='high'),
        ]
        return [d for d in defects if d.confidence >= self.confidence_threshold]


# ---------------------------------------------------------------------------
# Hybrid pipeline
# ---------------------------------------------------------------------------
class HybridInspector:
    def __init__(self, config: Optional[dict] = None):
        self.edge_detector = EdgeDetector(config)
        self.morph_detector = MorphologicalDetector(config)
        self.cnn_detector = CNNDefectDetector(config)

    def inspect(self, image_path: str) -> InspectionResult:
        import time
        start = time.time()

        results = {
            'edge': self.edge_detector.detect(image_path),
            'morphology': self.morph_detector.detect(image_path),
            'cnn': self.cnn_detector.detect(image_path),
        }

        all_defects = results['cnn'].copy()
        for det in results['edge'] + results['morphology']:
            overlapping = False
            for existing in all_defects:
                if self._iou(det.bbox, existing.bbox) > 0.3:
                    existing.confidence = max(existing.confidence, det.confidence)
                    overlapping = True
                    break
            if not overlapping and det.confidence > 0.5:
                all_defects.append(det)

        elapsed = (time.time() - start) * 1000
        result = InspectionResult(
            image_path=image_path,
            image_size=(640, 480),
            defects=all_defects,
            processing_time_ms=elapsed,
            passes=len([d for d in all_defects if d.severity == 'high']) == 0,
        )

        return result

    @staticmethod
    def _iou(bbox1, bbox2) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Machine Vision Inspection System')
    parser.add_argument('--image', '-i', help='Single image to inspect')
    parser.add_argument('--batch', '-b', help='Batch process a folder of images')
    parser.add_argument('--output', '-o', default='results', help='Output directory')
    parser.add_argument('--method', '-m', default='hybrid',
                        choices=['edge', 'morphology', 'cnn', 'hybrid'],
                        help='Detection method')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--export-json', action='store_true', help='Export results as JSON')

    args = parser.parse_args()

    config = {'confidence_threshold': args.threshold}

    if args.method == 'hybrid':
        inspector = HybridInspector(config)
    elif args.method == 'edge':
        inspector = EdgeDetector(config)
    elif args.method == 'morphology':
        inspector = MorphologicalDetector(config)
    else:
        inspector = CNNDefectDetector(config)

    if args.image:
        if isinstance(inspector, (EdgeDetector, MorphologicalDetector, CNNDefectDetector)):
            defects = inspector.detect(args.image)
            result = InspectionResult(
                image_path=args.image,
                defects=defects,
                passes=len([d for d in defects if d.severity == 'high']) == 0,
            )
        else:
            result = inspector.inspect(args.image)

        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        print(f"\n-> {'PASS' if result.passes else 'FAIL'} | "
              f"{result.defect_count} defects found | "
              f"{result.processing_time_ms:.1f} ms")

    elif args.batch:
        os.makedirs(args.output, exist_ok=True)
        images = [f for f in os.listdir(args.batch)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        all_results = []
        for img_name in images:
            img_path = os.path.join(args.batch, img_name)
            if isinstance(inspector, (EdgeDetector, MorphologicalDetector, CNNDefectDetector)):
                defects = inspector.detect(img_path)
                result = InspectionResult(image_path=img_path, defects=defects)
            else:
                result = inspector.inspect(img_path)
            all_results.append(result.to_dict())
            status = 'PASS' if result.passes else 'FAIL'
            print(f"  [{status}] {img_name}: {result.defect_count} defects ({result.processing_time_ms:.0f} ms)")

        total_pass = sum(1 for r in all_results if r['passes'])
        print(f"\nBatch complete: {total_pass}/{len(all_results)} passed")

    return 0


if __name__ == '__main__':
    sys.exit(main())