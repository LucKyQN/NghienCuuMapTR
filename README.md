# MapTR Deployment

Repo triển khai MapTR (baseline) cho bài toán HD Map Estimation, train/test trên full nuScenes trainval dataset. Đạt **mAP = 0.4998**, khớp với số liệu chính thức trong paper/repo gốc (mAP 50.0).


---

## Mục lục

1. [Tổng quan hạ tầng](#1-tổng-quan-hạ-tầng)
2. [Kết quả đạt được](#2-kết-quả-đạt-được)
3. [Cài đặt môi trường](#3-cài-đặt-môi-trường)
4. [Chuẩn bị dataset](#4-chuẩn-bị-dataset)
5. [Các bug đã fix trong code gốc](#5-các-bug-đã-fix-trong-code-gốc-quan-trọng---không-được-revert)
6. [Chạy training / test](#6-chạy-training--test)
7. [Demo web app (Flask 6-camera inference)](#7-demo-web-app-flask-6-camera-inference)
8. [Giới hạn kỹ thuật cần biết](#8-giới-hạn-kỹ-thuật-cần-biết)
9. [File quan trọng — không được xóa](#9-file-quan-trọng--không-được-xóa)
10. [Troubleshooting thường gặp](#10-troubleshooting-thường-gặp)
11. [Roadmap tiếp theo](#11-roadmap-tiếp-theo)

---

## 1. Tổng quan hạ tầng

Setup gồm 2 server nội bộ, thông LAN, mount chéo qua `sshfs`:

| | Server 1 (Compute) | Server 2 (Storage) |
|---|---|---|
| IP | `..........` | `..........` |
| SSH user | `quangnam` | `......` |
| GPU | A100 | Không |
| Vai trò | Chạy training/inference | Lưu trữ dataset (không cài Docker/env) |
| Data thật | Không (chỉ mount qua sshfs) | `/data/maptr/nuscenes/` (~402GB, full trainval) |

**Sơ đồ mount:**
```
Server 1: ~/maptr/data/nuscenes
              └─ symlink →
Server 1: ~/data_server2/nuscenes   (sshfs mount)
              └─ thật sự nằm ở →
Server 2: /data/maptr/nuscenes/
```

**Docker:**
- Base image: `nvcr.io/nvidia/pytorch:22.12-py3`
- Container đang chạy: `maptr_container`
- Image đã commit (backup state): `maptr_server:v3`, `maptr_server:v4` ← **dùng v4**, đã có sẵn `app.py` + notebook
- Bind mount: `~/maptr` (host, server 1) → `/workspace/MapTR` (trong container)

> ⚠️ **Lưu ý dung lượng đĩa dễ gây hoảng:** `du -sh` trên thư mục `data` sẽ hiện ảo ~412G vì lệnh `du` đi xuyên qua sshfs mount point và cộng luôn dung lượng thật bên server 204. Dùng `du -x` để xem dung lượng thật local (chỉ ~10GB — data mini cũ). Disk local server 1 hiện dùng ~521G/878G (63%).

---

## 2. Kết quả đạt được

Train/test trên **full nuScenes trainval** (850 scenes, 6019 val samples):

| Metric | Giá trị |
|---|---|
| mAP | **0.4998** |
| divider | 0.5208 |
| ped_crossing | 0.4537 |
| boundary | 0.5248 |

So sánh: test trên nuScenes-mini (81 samples) trước đó chỉ ra 0.188 — **thấp do thiếu data, không phải lỗi code**.

---

## 3. Cài đặt môi trường

### 3.1. Yêu cầu
- Docker + NVIDIA Container Toolkit (để container thấy GPU)
- Truy cập SSH vào cả 2 server
- Image gốc: `nvcr.io/nvidia/pytorch:22.12-py3`

### 3.2. Pull / build container

```bash
# Nếu dùng lại image đã build sẵn (khuyến nghị — đỡ mất công build lại mmcv-full/CUDA ops):
docker load -i maptr_server_v4.tar   # nếu image được export dạng tar
# hoặc pull từ registry nội bộ nếu có

docker run --gpus all -it --name maptr_container \
  -v ~/maptr:/workspace/MapTR \
  maptr_server:v4 bash
```

### 3.3. Nếu phải build từ đầu (không khuyến khích, tốn nhiều công)

Môi trường này khó dựng vì:
- Base image dùng custom PyTorch build của NGC → `mmcv-full` **phải compile from source**, không dùng bản pip prebuilt được
- Custom CUDA ops `bev_pool`, `bev_pool_v2` phải build tay
- Một loạt lỗi tương thích: đường dẫn module `numba` đổi, `EfficientNet` bị đăng ký trùng, hàm transformer trả về sai số lượng giá trị

→ **Khuyến nghị**: dùng lại `maptr_server:v4` đã build sẵn thay vì build lại từ đầu.

---

## 4. Chuẩn bị dataset

Dataset: **full nuScenes trainval** (~402GB), lưu ở server 2 (`10.70.39.204:/data/maptr/nuscenes/`).

### 4.1. Tải dataset (đã làm, chỉ cần biết vị trí)
Data được tải bằng `aria2c` về server 2, **không cần tải lại**.

### 4.2. Mount dataset vào server 1

```bash
# Trên server 1 (10.70.39.39):
sshfs vnpt@10.70.39.204:/data/maptr/nuscenes ~/data_server2/nuscenes

# Symlink để container thấy đúng path mong đợi:
ln -s ~/data_server2/nuscenes ~/maptr/data/nuscenes
```

Kiểm tra mount thành công:
```bash
ls ~/maptr/data/nuscenes
# Phải thấy: samples/ sweeps/ maps/ v1.0-trainval/ ...
```

---

## 5. Các bug đã fix trong code gốc (QUAN TRỌNG — KHÔNG ĐƯỢC REVERT)

Repo MapTR gốc có nhiều lỗi khi chạy trên full trainval + môi trường NGC container. Các file sau **đã được sửa thủ công**, không được ghi đè lại bản gốc từ upstream:

| # | File | Fix |
|---|---|---|
| 1 | `tools/maptrv2/custom_nusc_map_converter.py` | Bỏ ghép cứng chuỗi `-trainval` |
| 2 | `projects/mmdet3d_plugin/maptr/modules/transformer.py` | Fix `ret_dict['bev']` |
| 3 | `projects/mmdet3d_plugin/maptr/dense_heads/maptr_head.py` | Fix unpack 5 giá trị trả về |
| 4 | `tools/test.py` | Uncomment `MMDataParallel` cho single GPU |
| 5 | `mmdetection3d/mmdet3d/datasets/pipelines/data_augment_utils.py` | Fix đường dẫn `numba.core.errors` |
| 6 | `projects/mmdet3d_plugin/bevformer/modules/encoder.py` | Fix `points_in_boxes_batch` |
| 7 | `mmdetection3d/mmdet3d/ops/__init__.py` | Thêm import `bev_pool`, `bev_pool_v2` |
| 8 | `tools/maptrv2/custom_nusc_map_converter.py` (hàm `union_centerline`, dòng ~743) | Bọc try/except quanh `nx.all_simple_paths` |

Fix #8 chi tiết (bug đã biết của MapTR repo trên full trainval, tác giả gốc chưa fix chính thức):

```python
try:
    paths = nx.all_simple_paths(pts_G, root, leaves)
except nx.NodeNotFound:
    continue
```

> Nếu pull code mới từ upstream MapTR, **phải áp lại 8 fix này** trước khi chạy trên full trainval.

---

## 6. Chạy training / test

```bash
cd /workspace/MapTR

# Test/inference với checkpoint đã train:
python tools/test.py \
  projects/configs/maptr/maptr_tiny_r50_24e.py \
  ckpts/maptr_tiny_r50_24e.pth \
  --eval chamfer
```

Kết quả mong đợi: mAP ~0.4998 (xem mục 2).

---

## 7. Demo web app (Flask 6-camera inference)

**File**: `app.py` (Flask) và `maptr_6cam_inference.ipynb` — cả 2 nằm trong `/workspace/MapTR/`.

**Chức năng**: Upload 6 ảnh camera (bố trí đúng vị trí thật quanh xe: front-left/front/front-right phía trên, back-left/back/back-right phía dưới) → chạy MapTR inference → hiển thị vectorized map song song với ảnh gốc.

### 7.1. Bật demo — làm đúng thứ tự

```bash
# Bước 1: SSH/VS Code vào server 1 (10.70.39.39)

# Bước 2: Check GPU trong container TRƯỚC KHI chạy app (hay lỗi sau khi restart máy)
sudo docker exec -it maptr_container bash
nvidia-smi
```

Nếu gặp lỗi `Failed to initialize NVML`:
```bash
exit
sudo docker restart maptr_container
sudo docker exec -it maptr_container bash
```

```bash
# Bước 3: Chạy Flask app
cd /workspace/MapTR
python app.py
# Đợi đến khi thấy dòng: "Model đã sẵn sàng..."
```

```bash
# Bước 4: Mở tab terminal MỚI (không đụng tab đang chạy app.py), lấy IP container
sudo docker inspect maptr_container | grep IPAddress
# IP hiện tại: 172.17.0.13 (thường không đổi giữa các lần restart)
```

```bash
# Bước 5: Trên máy local (CMD Windows), tunnel THẲNG VÀO IP CONTAINER
ssh -L 5004:172.17.0.13:5000 quangnam@10.70.39.39
# Để yên cửa sổ này, không đóng
```

```
# Bước 6: Mở trình duyệt
http://127.0.0.1:5004
```

### ⚠️ Lỗi hay gặp nhất: đừng dùng VS Code auto port-forward

**KHÔNG** dùng tab "Ports" của VS Code để auto-forward `127.0.0.1:5000` — nó sẽ **không hoạt động** vì Flask chạy bên trong container, không map port trực tiếp ra host. Bắt buộc phải SSH tunnel thẳng vào IP nội bộ của container như bước 5.

### 7.2. Tắt demo

1. Đóng tab trình duyệt
2. Đóng cửa sổ CMD đang tunnel (`Ctrl+C`)
3. Tab đang chạy Flask: `Ctrl+C`
4. `exit` khỏi container
5. Đóng VS Code

> Container `maptr_container` **để nguyên**, không cần `docker stop` giữa các lần demo — chỉ cần lặp lại bước 2-6 ở lần sau.

---

## 8. Giới hạn kỹ thuật cần biết

MapTR bắt buộc cần **calibration** (intrinsic/extrinsic từng camera) để chuyển ảnh 2D → BEV.

**Trạng thái hiện tại của demo**: app lấy calib cố định từ 1 sample có sẵn (`SAMPLE_IDX = 0` trong `nuscenes_infos_temporal_val.pkl`), chỉ thay phần ảnh input.

**Đã test và xác nhận**: đưa 6 ảnh phố Boston rời rạc (khác thời điểm/vị trí, không đồng bộ) vào → kết quả ra gần giống hệt các lần test trước dù nội dung ảnh khác hoàn toàn.

**Kết luận**: khi calib không khớp với ảnh thật, model không thực sự "đọc" nội dung ảnh mới — kết quả chủ yếu phản ánh cấu trúc hình học cố định từ calib.

**Yêu cầu để inference đúng nghĩa trên ảnh thực tế**: cần calibration thật (intrinsic + extrinsic) của đúng bộ 6 camera chụp **đồng thời trên cùng 1 xe, cùng 1 thời điểm** — không phải 6 ảnh rời rạc dù đúng tên vị trí camera.

---

## 9. File quan trọng — không được xóa

- `~/maptr/` toàn bộ (code MapTR + 8 fix ở mục 5)
- `ckpts/maptr_tiny_r50_24e.pth` — checkpoint đã train
- Docker image `maptr_server:v3`, `maptr_server:v4`
- `~/maptr/app.py`, `~/maptr/maptr_6cam_inference.ipynb`

---

## 10. Troubleshooting thường gặp

| Triệu chứng | Nguyên nhân | Cách fix |
|---|---|---|
| `Failed to initialize NVML` khi chạy `nvidia-smi` trong container | Container mất kết nối GPU sau khi restart máy/VS Code | `docker restart maptr_container` rồi exec lại |
| `du -sh` báo dung lượng data cực lớn (400GB+) trên server 1 | `du` đi xuyên qua sshfs mount, cộng luôn dung lượng thật bên server 2 | Dùng `du -x` để xem dung lượng thật local |
| Web demo không load được ở `127.0.0.1:5000` | Dùng nhầm VS Code auto port-forward thay vì SSH tunnel thủ công | Tunnel thẳng vào IP container theo mục 7.1 bước 5 |
| Kết quả inference giống hệt nhau dù đổi ảnh input | Calibration cố định, không khớp ảnh thật (xem mục 8) | Cần bộ calib thật đồng bộ 6 camera, hỏi mentor xác nhận yêu cầu |
| Lỗi `networkx.exception.NodeNotFound` khi convert data full trainval | Bug đã biết của MapTR repo, chưa có fix chính thức từ tác giả | Đã fix bằng try/except, xem mục 5 fix #8 |

---

4. **MapTracker** (ECCV 2024 Oral) — https://github.com/woodfrog/maptracker — memory-based tracking, kiến trúc khác biệt nhất

**Việc cần làm trước khi merge nhiều local map thành 1 global map**: xác nhận với mentor có cần bộ 6 ảnh calibration thật (đồng bộ, cùng xe, cùng thời điểm) hay demo hiện tại đã đủ đáp ứng yêu cầu.
