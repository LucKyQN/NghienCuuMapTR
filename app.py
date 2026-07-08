import os
import sys
import copy
import base64
import io

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspace/MapTR')
os.chdir('/workspace/MapTR')

from flask import Flask, request, render_template_string
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets.pipelines import Compose
import projects.mmdet3d_plugin  # noqa

CONFIG_PATH = 'projects/configs/maptr/maptr_tiny_r50_24e.py'
CHECKPOINT_PATH = 'ckpts/maptr_tiny_r50_24e.pth'
SAMPLE_IDX = 0
SCORE_THR = 0.4

COLOR_BG = '#0A0E12'
COLOR_PANEL = '#121820'
COLOR_BORDER = '#26313D'
COLOR_TEXT = '#E8EEF3'
COLOR_TEXT_DIM = '#7E8FA0'
CLASS_COLOR = {0: '#F2A65A', 1: '#4C9EF1', 2: '#34D399'}
CLASS_NAME = {0: 'divider', 1: 'ped_crossing', 2: 'boundary'}

UPLOAD_DIR = 'custom_images_web'
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)

print('Đang load model, đợi chút...')
cfg = Config.fromfile(CONFIG_PATH)
cfg.model.pretrained = None
cfg.model.train_cfg = None
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, CHECKPOINT_PATH, map_location='cpu')
model.CLASSES = ['divider', 'ped_crossing', 'boundary']
model = model.cuda()
model = MMDataParallel(model, device_ids=[0])
model.eval()

dataset = build_dataset(cfg.data.test)
ref_info = dataset.data_infos[SAMPLE_IDX]
CAM_ORDER = list(ref_info['cams'].keys())
test_pipeline = Compose(cfg.data.test.pipeline)
print('Model đã sẵn sàng. Thứ tự 6 camera:', CAM_ORDER)

CAM_GRID_AREA = {
    'CAM_FRONT_LEFT': 'fl', 'CAM_FRONT': 'f', 'CAM_FRONT_RIGHT': 'fr',
    'CAM_BACK_LEFT': 'bl', 'CAM_BACK': 'b', 'CAM_BACK_RIGHT': 'br',
}
CAM_LABEL_VI = {
    'CAM_FRONT_LEFT': 'Trước · Trái', 'CAM_FRONT': 'Trước', 'CAM_FRONT_RIGHT': 'Trước · Phải',
    'CAM_BACK_LEFT': 'Sau · Trái', 'CAM_BACK': 'Sau', 'CAM_BACK_RIGHT': 'Sau · Phải',
}


PAGE = """
<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>MapTR — HD Map Inference</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0A0E12;
    --panel: #121820;
    --panel-2: #1A222C;
    --border: #26313D;
    --text: #E8EEF3;
    --text-dim: #7E8FA0;
    --accent: #2DD4BF;
    --accent-dim: #1B8F82;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
    padding: 32px 24px 80px;
  }
  .wrap { max-width: 980px; margin: 0 auto; }
  header {
    display: flex; align-items: baseline; justify-content: space-between;
    border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 32px;
  }
  h1 {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700; font-size: 28px; letter-spacing: 0.5px;
    margin: 0; text-transform: uppercase;
  }
  h1 span { color: var(--accent); }
  .subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; color: var(--text-dim);
  }
  .status-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent); margin-right: 6px;
    box-shadow: 0 0 8px var(--accent);
  }
  .section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 1.5px;
    margin-bottom: 14px;
  }

  .cam-rig {
    display: grid;
    grid-template-columns: 1fr 140px 1fr;
    grid-template-areas:
      "fl f fr"
      ".  car ."
      "bl b br";
    gap: 14px;
    align-items: center;
    margin-bottom: 28px;
  }
  .cam-slot[data-area="fl"] { grid-area: fl; }
  .cam-slot[data-area="f"]  { grid-area: f; }
  .cam-slot[data-area="fr"] { grid-area: fr; }
  .cam-slot[data-area="bl"] { grid-area: bl; }
  .cam-slot[data-area="b"]  { grid-area: b; }
  .cam-slot[data-area="br"] { grid-area: br; }

  .cam-box {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
    transition: border-color 0.15s;
  }
  .cam-box:hover { border-color: var(--accent-dim); }
  .cam-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: var(--accent);
    letter-spacing: 0.5px; margin-bottom: 2px;
  }
  .cam-label-vi { font-size: 12px; color: var(--text-dim); margin-bottom: 10px; }
  .cam-box input[type=file] {
    width: 100%; font-size: 11px; color: var(--text-dim);
  }
  .cam-box input[type=file]::file-selector-button {
    background: var(--panel-2); color: var(--text); border: 1px solid var(--border);
    border-radius: 4px; padding: 5px 10px; font-size: 11px; cursor: pointer;
    font-family: 'Inter', sans-serif;
  }

  .car-svg { grid-area: car; display: flex; align-items: center; justify-content: center; }

  .run-btn-wrap { text-align: center; margin: 8px 0 36px; }
  button.run {
    background: var(--accent); color: #06201C; border: none;
    font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 18px;
    letter-spacing: 1px; text-transform: uppercase;
    padding: 12px 40px; border-radius: 6px; cursor: pointer;
  }
  button.run:hover { background: #3EE6D0; }

  .result-grid {
    display: grid; grid-template-columns: 260px 1fr; gap: 24px;
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 20px;
  }
  .thumb-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
  }
  .thumb-item { text-align: center; }
  .thumb-item img {
    width: 100%; height: 60px; object-fit: cover; border-radius: 4px;
    border: 1px solid var(--border);
  }
  .thumb-item .lbl {
    font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--text-dim);
    margin-top: 3px;
  }
  .map-result img { width: 100%; border-radius: 6px; }

  footer.hint {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-dim);
    margin-top: 40px; border-top: 1px solid var(--border); padding-top: 16px;
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>HD Map <span>&middot;</span> Inference</h1>
      <div class="subtitle">MapTR &middot; mAP 0.4998 &middot; nuScenes trainval</div>
    </div>
    <div class="subtitle"><span class="status-dot"></span>MODEL READY</div>
  </header>

  <div class="section-label">01 &mdash; Sensor Array (6 camera, đúng vị trí trên xe)</div>

  <form method=post enctype=multipart/form-data>
    <div class="cam-rig">
      {% for cam in cam_order %}
      <div class="cam-slot" data-area="{{ cam_area[cam] }}">
        <div class="cam-box">
          <div class="cam-name">{{ cam }}</div>
          <div class="cam-label-vi">{{ cam_label[cam] }}</div>
          <input type=file name="{{ cam }}" accept="image/*" required>
        </div>
      </div>
      {% endfor %}
      <div class="car-svg">
        <svg width="70" height="120" viewBox="0 0 70 120">
          <rect x="12" y="8" width="46" height="104" rx="14" fill="none" stroke="#2DD4BF" stroke-width="2"/>
          <rect x="20" y="18" width="30" height="24" rx="4" fill="#1A222C" stroke="#26313D"/>
          <line x1="35" y1="8" x2="35" y2="112" stroke="#26313D" stroke-width="1" stroke-dasharray="3 3"/>
          <circle cx="35" cy="60" r="3" fill="#2DD4BF"/>
        </svg>
      </div>
    </div>

    <div class="run-btn-wrap">
      <button type=submit class="run">Chạy Inference</button>
    </div>
  </form>

  {% if result_ready %}
  <div class="section-label">02 &mdash; Kết quả</div>
  <div class="result-grid">
    <div>
      <div class="section-label" style="margin-bottom:8px;">Ảnh đầu vào</div>
      <div class="thumb-grid">
        {% for cam in cam_order %}
        <div class="thumb-item">
          <img src="data:image/jpeg;base64,{{ thumbs[cam] }}">
          <div class="lbl">{{ cam }}</div>
        </div>
        {% endfor %}
      </div>
    </div>
    <div class="map-result">
      <div class="section-label" style="margin-bottom:8px;">Vectorized Map (BEV, score &gt; {{ score_thr }})</div>
      <img src="data:image/png;base64,{{ map_b64 }}">
    </div>
  </div>
  {% endif %}

  <footer class="hint">
    Calibration lấy từ sample #{{ sample_idx }} trong nuscenes_infos_temporal_val.pkl &mdash;
    kết quả chỉ chính xác nếu ảnh khớp vị trí/góc camera của sample gốc.
  </footer>
</div>
</body>
</html>
"""


def img_to_b64(path, resize=None):
    from PIL import Image
    im = Image.open(path).convert('RGB')
    if resize:
        im.thumbnail(resize)
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=70)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def run_inference(img_paths_by_cam):
    custom_info = copy.deepcopy(ref_info)
    for cam, path in img_paths_by_cam.items():
        custom_info['cams'][cam]['data_path'] = os.path.abspath(path)

    input_dict = dataset.get_data_info(SAMPLE_IDX)
    input_dict['img_filename'] = [custom_info['cams'][cam]['data_path'] for cam in CAM_ORDER]

    example = test_pipeline(input_dict)
    data = dict(
        img=[example['img'][0].data.unsqueeze(0).cuda()],
        img_metas=[[example['img_metas'][0].data]],
    )
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    pred = result[0]['pts_bbox']
    pts_3d = pred['pts_3d'].numpy() if torch.is_tensor(pred['pts_3d']) else np.array(pred['pts_3d'])
    labels = pred['labels_3d'].numpy() if torch.is_tensor(pred['labels_3d']) else np.array(pred['labels_3d'])
    scores = pred['scores_3d'].numpy() if torch.is_tensor(pred['scores_3d']) else np.array(pred['scores_3d'])

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(COLOR_PANEL)
    ax.set_facecolor(COLOR_BG)
    for spine in ax.spines.values():
        spine.set_color(COLOR_BORDER)
    ax.tick_params(colors=COLOR_TEXT_DIM, labelsize=9)
    ax.grid(color=COLOR_BORDER, linewidth=0.6, alpha=0.6)

    for pts, label, score in zip(pts_3d, labels, scores):
        if score < SCORE_THR:
            continue
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], color=CLASS_COLOR.get(int(label), 'gray'),
                label=CLASS_NAME.get(int(label), str(label)), linewidth=2.2)

    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(), by_label.keys(), facecolor=COLOR_PANEL,
              edgecolor=COLOR_BORDER, labelcolor=COLOR_TEXT, fontsize=10)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-30, 30)
    ax.set_aspect('equal')
    ax.set_title('MapTR prediction', color=COLOR_TEXT, fontsize=13, fontfamily='monospace')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=COLOR_PANEL)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


@app.route('/', methods=['GET', 'POST'])
def index():
    result_ready = False
    map_b64 = None
    thumbs = {}
    if request.method == 'POST':
        img_paths = {}
        for cam in CAM_ORDER:
            f = request.files.get(cam)
            if f and f.filename:
                path = os.path.join(UPLOAD_DIR, f'{cam}.jpg')
                f.save(path)
                img_paths[cam] = path
        if len(img_paths) == len(CAM_ORDER):
            map_b64 = run_inference(img_paths)
            thumbs = {cam: img_to_b64(img_paths[cam], resize=(200, 200)) for cam in CAM_ORDER}
            result_ready = True

    return render_template_string(
        PAGE, cam_order=CAM_ORDER, cam_area=CAM_GRID_AREA, cam_label=CAM_LABEL_VI,
        result_ready=result_ready, map_b64=map_b64, thumbs=thumbs,
        score_thr=SCORE_THR, sample_idx=SAMPLE_IDX,
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
