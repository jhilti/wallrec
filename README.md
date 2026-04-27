## Usage

```python

.venv/bin/python hold_detection_workflow.py IMG_1637.jpeg --backend classic --target-count 92
.venv/bin/python hold_detection_workflow.py IMG_1637.jpeg --backend sam-vit-b --checkpoint sam_vit_b_01ec64.pth --target-count 92
.venv/bin/python hold_detection_workflow.py IMG_1637.jpeg --backend mobile-sam --checkpoint mobile_sam.pt --target-count 92
.venv/bin/python hold_detection_workflow.py IMG_1637.jpeg --backend efficient-sam --checkpoint efficient_sam_vitt.pt --target-count 92
.venv/bin/python hold_detection_workflow.py IMG_1637.jpeg --backend fastsam --yolo-model FastSAM-s.pt --target-count 92
.venv/bin/python hold_detection_workflow.py IMG_1637.jpeg --backend yolo-seg --yolo-model path/to/hold-best.pt --target-count 92

```