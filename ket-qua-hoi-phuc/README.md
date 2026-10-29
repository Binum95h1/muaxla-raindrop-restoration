# Mã chạy và kết quả

Xem [README chính](../README.md) để tảiZIPđầyđủ chứa checkpoint và hướng dẫn chạy.

- `Phuc_hoi_anh.py`: công cụ inference một ảnh/thư mục, không huấn luyện.
- `runtime.py`, `models.py`, `restormer_arch.py`, `data_utils.py`, `reference_split.json`: mã/phânchia tham chiếu của Hybrid gốc.
- `Bao_cao_ket_qua.md`, `*_per_image.csv`, `ket_qua_tong_hop.csv`: kết quả đã xác minh.
- `So_sanh_Hybrid.png`: minh họaInput/Hybrid/GT, cropthu nhỏ.
- `Thong_tin_model.json`, `Restormer_LICENSE.txt`: nguồn checkpoint và giấy phép mã kế thừa.

Checkpoint `best.pt` được tải trongZIPRelease, không nằm trong Git tree. Để chạy thư mục mã trực tiếp, chépbest.ptcùngPhuc_hoi_anh.py; mã hỗ trợ cả layoutphẳng ởđây vàlayoutHybrid_goc/Ma_nguontrongZIP.
