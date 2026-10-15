# Quy tắc Kỹ thuật Hệ thống Interview Agent (SE373 - Đề tài 05)

1. **Công thức trục**: `agent = model + harness`. Không huấn luyện lại model; mọi độ tin cậy phải được kiểm soát bằng code ở tầng Harness.
2. **Cấm tự động ra quyết định tuyển dụng**: Agent chỉ đề xuất (`recommend_level`). Quyết định cuối (`advance` / `reject` / `hold`) bắt buộc chờ người có vai trò `recruiter` phê duyệt qua HITL.
3. **Bảo vệ dữ liệu cá nhân & Chống thiên lệch**: Bắt buộc chạy `redact_pii` che 11 thuộc tính được bảo vệ trước khi đưa CV hoặc transcript vào LLM.
4. **Ranh giới bảo mật tri thức**: `rubrics`, `expected_signals` và `hidden_tests` là tài liệu mật, tuyệt đối chỉ nạp cho `grader` và `coding-assessment`, không bao giờ trả về cho `interviewer` hoặc ứng viên.
5. **Structured Output & Verification**: Cấm dùng regex/substring để đoán ý định từ prose output. Mọi giao tiếp dữ liệu phải qua Pydantic Schema và kiểm tra trích dẫn chuỗi con có thật (`is_exact_substring_quote`).
