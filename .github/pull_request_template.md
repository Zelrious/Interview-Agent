## 📌 Liên kết Issue & Đặc tả Task
- **Closes:** Closes #
- **Phase / Milestone:** 
- **Use Case chính:** [ ] RAG-Buoi06 | [ ] UC-01 | [ ] UC-02 | [ ] UC-03 | [ ] UC-04 | [ ] UC-05 | [ ] UC-06 | [ ] UC-09 | [ ] Khác: 
- **Mảng Kiến thức tác động (K1-K8):**
  - [ ] K1: Model & Structured Output
  - [ ] K2: Tools & Skills
  - [ ] K3: RAG Knowledge Harness
  - [ ] K4: Memory & State Management
  - [ ] K5: Multi-step Workflow & HITL
  - [ ] K6: Multi-agent & MCP
  - [ ] K7: Verification & Evaluation
  - [ ] K8: Security & Observability

---

## 🛠️ Tầng Harness Lõi & Thay đổi Kỹ thuật
- **Lớp Harness được thêm / chỉnh sửa:** (ví dụ: `retrieval`, `verification`, `permission`, `workflow`, `memory`, ...)
- **Mô tả vắn tắt giải pháp:**
  - 

---

## ✅ Checklist Kiểm duyệt Chất lượng & Tuân thủ Quy tắc (Bắt buộc tích đủ 100%)
- [ ] **Bảo mật Secret:** Đã chạy `gitleaks` / `pre-commit`, tuyệt đối KHÔNG commit API Key, token, mật khẩu hoặc file `.env`.
- [ ] **Bảo vệ PII & Chống thiên lệch:** 100% dữ liệu JD/CV/Transcript là hư cấu (synthetic); bắt buộc chạy qua `redact_pii` che đủ 11 thuộc tính được bảo vệ trước khi truyền vào LLM.
- [ ] **Ranh giới Bảo mật Tri thức:** Tài liệu mật (`rubrics`, `expected_signals`, `hidden_tests`) tuyệt đối chỉ nạp cho `grader` / `coding-assessment`, không bao giờ trả về cho `interviewer` hoặc ứng viên.
- [ ] **Hạn chế Quyền hạn AI (HITL):** Agent chỉ đề xuất (`recommend_level`), quyết định tuyển dụng cuối cùng (`advance`/`reject`/`hold`) bắt buộc chờ Recruiter phê duyệt.
- [ ] **Structured Output Chuẩn mực:** Tuyệt đối không dùng regex/substring để đoán kiểu dữ liệu từ prose output; 100% giao tiếp dữ liệu có cấu trúc qua Pydantic Schema (`ConfigDict(extra="forbid")`).
- [ ] **Kiểm thử & Linter:** Toàn bộ lệnh kiểm tra đã chạy thành công trên máy local:
  - `pytest -v` (100% PASSED)
  - `ruff check src evals tests` (0 cảnh báo, 0 lỗi)

---

## 🤖 Khai báo Minh bạch Sử dụng AI (AI Transparency Statement)
- **Công cụ AI hỗ trợ:** (ví dụ: GitHub Copilot, Gemini CLI, Claude 3.5 Sonnet, Cursor, ChatGPT...)
- **Phần việc AI hỗ trợ:** (ví dụ: gợi ý cú pháp typing, viết boilerplate Pydantic schema, sinh dữ liệu test...)
- **Cam kết của lập trình viên:** *Tôi xác nhận đã đọc, hiểu và tự tay rà soát từng dòng code trong PR này. Tôi chịu trách nhiệm giải thích chi tiết kiến trúc và trả lời mọi câu hỏi vấn đáp trực tiếp với giảng viên môn học SE373.*
