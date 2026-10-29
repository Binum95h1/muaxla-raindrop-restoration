# Kết quả phục hồi ảnh mưa Muaxla

## Bản được giữ

Hybrid gốc là checkpoint được giữ để sử dụng. Những nhánh thử bổ sung chưa đạt mục tiêu giảm thêm ít nhất5% lỗi vùng giọt nước cùng các điều kiện bảo toàn nền, biên và ảnh sạch. Không tuyên bố đã phục hồi hoàn toàn, không tuyên bố phương pháp mới hay tái lập nguyên bản một bài báo.

## Lượt video cuối —363418312

Huấn luyện đủ2000updates,4000samplesTRAIN,54scene/108windows/260patch. Thời gian29phút26giây. Chấm native18ảnh thuộc9scene validation đã dùng lặp lại, cộng9clean identity windows; không phải fullvalidation900 hay test độc lập.

| Chỉ số | Hybrid gốc | Video reconstruction | Repeat-center |
|---|---:|---:|---:|
| ROI MAE ↓ |0.028461868771|0.028462076818|0.028462083855|
| PSNR dB ↑ |36.197153845|36.197151910|36.197150139|
| ROI edge MAE ↓ |0.011281176030|0.011281145064|0.011281149151|
| Background MAE ↓ |0.003009211550|0.003009198609|0.003009198330|
| Clean identity MAE ↓ |0.000409602802|0.000409571622|0.000409567473|

Video ROI xấu hơn Hybrid0.000730968%; hơnrepeat0.000024723%, quá nhỏ để coi là cải thiện hữu ích. Không đạt giảm5%, vì vậy dừng nhánh video và giữ Hybrid. Repeat là cùng mạng/trọng số với centralRAW lặp5lần, không phải mạng singleframe huấn luyện riêng. Ảnh đầu ra gần như giữ nguyên Hybrid.

Checkpoint cuối/25AdamWstates counter2000/history/native semantic SHA và code đã được kiểm tra; Hybrid nguồn không đổi. Không tiếp tục tăngsteps hay đổi ngưỡng để chọn kết quả đẹp hơn.

## Giới hạn và bằng chứng

Ảnh minh họa là crop384thu nhỏ256, không phải đầu ra toàn kích thước. Mờ/vệt còn tồn tại; chưa có kiểm chứng chốngnhấpnháy toànvideo hoặc đánh giá test cuối mới. Công cụ Hybrid hiện xử lý từng ảnh/thư mục ảnh.

ZIP Release gồm modelHybrid,13báo cáo lịch sử,18preview6cột và số liệu xác minh. Các báo cáo lịch sử giữ các kết quả không đạt để đối chiếu; đường dẫn máy cá nhân đã chuẩn hóa. BaCSV trong thư mục mã là kết quả nguồn/video/repeat của cùng18ảnh ở lượt cuối.
