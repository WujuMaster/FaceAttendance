# FaceRecogTerminal
Just a small Terminal-menu project to identify faces, using face-recognition library from Python

HƯỚNG DẪN SỬ DỤNG

Bước 1: (tùy chọn) Kích hoạt môi trường ảo của python trong terminal:

.venv/scripts/activate


Bước 2: Cài đặt các thư viện cần thiết:

pip install -r requirements.txt


Bước 3: Thử chạy file demo.py:

python demo.py
  
![image](https://user-images.githubusercontent.com/55906223/140608888-90d73710-2f37-4ae7-be5e-00a3c7a661ae.png)


Chạy thành công sẽ hiện ra 3 cửa sổ của opencv, terminal sẽ in ra kết quả so sánh tương đồng và độ chênh lệch giữa 3 ảnh khuôn mặt. Nếu terminal báo thiếu thư viện nào thì cài thêm thư viện đó.


Bước 4: Chạy file main.py:

python main.py

Terminal sẽ in ra menu lựa chọn, lựa chọn 1 là lưu 1 khuôn mặt mới vào database bằng cách nhập tên ảnh (đuôi .jpg) - lưu trong folder Pictures, không cần nhập đuôi; lựa chọn 2 là nhận diện bằng camera, nếu có data trong database sẽ hiện tên người dùng, nếu không sẽ hiện "unknown"; lựa chọn 3 là thoát chương trình.

