# ninorobot2 – Codex working rules

## Khi code xong

Sau mọi thay đổi code, phải giải thích bằng tiếng Việt:

1. Luồng: block/input -> xử lý -> output.
2. File nào đã sửa và mỗi file chịu trách nhiệm gì.
3. Giải thích rõ function, class, callback, topic ROS, message, parameter,
   công thức, frame toạ độ mới hoặc đã sửa.
4. Nêu command build, chạy và kiểm tra copy-paste được.
5. Kết luận rõ: runtime đã kiểm chứng hay CHƯA KIỂM CHỨNG RUNTIME.

## Comment trong code

- Viết docstring hoặc comment cho mọi ROS node, callback, publisher,
  subscriber, parameter group, thuật toán, công thức và dữ liệu không tầm thường.
- Với ROS phải ghi rõ topic input/output, type message, frame toạ độ, đơn vị.
- Với prediction/costmap/TF phải giải thích giả định và ý nghĩa từng biến.
- Comment phải giải thích "vì sao", không chỉ nhắc lại cú pháp hiển nhiên.
- Khi sửa hành vi, cập nhật comment cũ để không bị sai.

## Quy tắc ninorobot2

- Trước khi sửa ROS/social-navigation, đọc `social_navigation/RUN_SCENARIOS.txt`.
- Không tự động thay đổi pipeline đang chạy ngoài phạm vi user yêu cầu.
- C++ thay đổi thì build package liên quan; thay đổi ROS runtime phải có lệnh kiểm tra topic/TF/RViz.
