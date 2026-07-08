Đây là toàn bộ nội dung bạn chỉ cần copy và paste trực tiếp vào file `README.md`. Mình đã định dạng sẵn bằng Markdown chuẩn để nó hiển thị đẹp nhất trên GitHub.

```markdown
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

---

## 1. Tổng quan hạ tầng

Setup gồm 2 server nội bộ, thông LAN, mount chéo qua `sshfs`:

| | Server 1 (Compute) | Server 2 (Storage) |
|---|---|---|
| IP | `<IP_SERVER_1>` | `<IP_SERVER_2>` |
| SSH user | `<USERNAME_1>` | `<USERNAME_2>` |
| GPU | A100 | Không |
| Vai trò | Chạy training/inference | Lưu trữ dataset |
| Data thật | Không (chỉ mount qua sshfs) | `<PATH_DATASET_SERVER_2>` |

**Sơ đồ mount:**

```

Server 1: ~/maptr/data/nuscenes
└─ symlink →
Server 1: ~/data_server2/nuscenes  (sshfs mount)
└─ thật sự nằm ở →
Server 2: <PATH_DATASET_SERVER_2>

```

**Docker:**
- Base image: `nvcr.io/nvidia/pytorch:22.12-py3`
- Container đang chạy: `maptr_container`
- Image đã commit (backup state): `maptr_server:v4`
- Bind mount: `~/maptr` (host, server 1) → `/workspace/MapTR` (trong container)

> ⚠️ **Lưu ý dung lượng đĩa:** Dùng lệnh `du -x` để xem dung lượng thực tế trên local server 1 thay vì dùng `du -sh` (lệnh này sẽ tính luôn cả dung lượng mount point từ server 2 gây hiểu lầm).

---

## 2. Kết quả đạt được

Train/test trên **full nuScenes trainval** (850 scenes, 6019 val samples):

| Metric | Giá trị |
|---|---|
| mAP | **0.4998** |
| divider | 0.5208 |
| ped_crossing | 0.4537 |
| boundary | 0.5248 |

---

## 3. Cài đặt môi trường

### 3.1. Yêu cầu
- Docker + NVIDIA Container Toolkit
- Truy cập SSH vào cả 2 server

### 3.2. Pull / build container

```bash
# Load image từ file tar hoặc pull từ registry nội bộ
docker load -i maptr_server_v4.tar

docker run --gpus all -it --name maptr_container \
  -v ~/maptr:/workspace/MapTR \
  maptr_server:v4 bash

```

### 3.3. Nếu phải build từ đầu

**Khuyến nghị**: sử dụng lại image `maptr_server:v4` đã build sẵn các thư viện đặc thù (mmcv-full, CUDA ops) thay vì tự build lại do xung đột phiên bản phức tạp giữa NGC PyTorch và mmdet3d.

---

## 4. Chuẩn bị dataset

Dataset: **full nuScenes trainval**, lưu ở server 2 (`<IP_SERVER_2>:<PATH_DATASET_SERVER_2>`).

### 4.1. Mount dataset vào server 1

```bash
# Trên server 1:
sshfs <USERNAME_2>@<IP_SERVER_2>:<PATH_DATASET_SERVER_2> ~/data_server2/nuscenes

# Symlink để container thấy đúng path:
ln -s ~/data_server2/nuscenes ~/maptr/data/nuscenes

```

---

## 5. Các bug đã fix trong code gốc (QUAN TRỌNG)

Repo MapTR gốc có nhiều lỗi khi chạy trên full trainval. Các file sau **đã được sửa thủ công**, không được ghi đè lại:

| # | File | Fix |
| --- | --- | --- |
| 1 | `tools/maptrv2/custom_nusc_map_converter.py` | Bỏ ghép cứng chuỗi `-trainval` |
| 2 | `projects/mmdet3d_plugin/maptr/modules/transformer.py` | Fix `ret_dict['bev']` |
| 3 | `projects/mmdet3d_plugin/maptr/dense_heads/maptr_head.py` | Fix unpack 5 giá trị trả về |
| 4 | `tools/test.py` | Uncomment `MMDataParallel` |
| 5 | `mmdetection3d/mmdet3d/datasets/pipelines/data_augment_utils.py` | Fix đường dẫn `numba` |
| 6 | `projects/mmdet3d_plugin/bevformer/modules/encoder.py` | Fix `points_in_boxes_batch` |
| 7 | `mmdetection3d/mmdet3d/ops/__init__.py` | Thêm import `bev_pool`, `bev_pool_v2` |
| 8 | `tools/maptrv2/custom_nusc_map_converter.py` | Thêm try/except `nx.all_simple_paths` |

---

## 6. Chạy training / test

```bash
cd /workspace/MapTR
python tools/test.py \
  projects/configs/maptr/maptr_tiny_r50_24e.py \
  ckpts/maptr_tiny_r50_24e.pth \
  --eval chamfer

```

---

## 7. Demo web app (Flask 6-camera inference)

### 7.1. Bật demo

1. **Kiểm tra GPU**: `sudo docker exec -it maptr_container nvidia-smi` (nếu lỗi, `restart` container).
2. **Chạy Flask**: `python app.py` (trong container).
3. **Tunnel từ local**:
```bash
ssh -L 5004:<IP_CONTAINER>:5000 <USERNAME_1>@<IP_SERVER_1>

```


4. **Truy cập**: `http://127.0.0.1:5004`

> ⚠️ **Lưu ý**: Tuyệt đối không dùng VS Code auto port-forward, bắt buộc dùng SSH tunnel thủ công vào IP của container.

---

## 8. Giới hạn kỹ thuật cần biết

Model cần calibration (intrinsic/extrinsic) để chuyển đổi 2D-BEV chính xác. Demo hiện tại sử dụng calibration từ một mẫu cố định. Để inference ảnh thực tế ngoài tập dữ liệu, cần bộ calibration đồng bộ theo đúng ảnh chụp.

---

## 10. Troubleshooting thường gặp

| Triệu chứng | Cách fix |
| --- | --- |
| `Failed to initialize NVML` | `docker restart maptr_container` |
| Web demo không load được | Kiểm tra lại SSH tunnel thủ công, không dùng VS Code port |
| Lỗi `NodeNotFound` (converter) | Đảm bảo đã áp dụng fix #8 trong mục 5 |

```

```
