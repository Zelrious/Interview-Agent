# Chính Sách Bảo Mật Dữ Liệu & Chống Thiên Lệch Tuyển Dụng

Tài liệu quy định các nguyên tắc bắt buộc về bảo mật thông tin cá nhân, thời hạn lưu trữ dữ liệu và trách nhiệm của hệ thống AI trong quy trình phỏng vấn (Đề tài 05 - Interview Agent, SE373).

---

## 1. Bảo Vệ Thuộc Tính Cá Nhân & Chống Thiên Lệch

Để bảo đảm tính công bằng tuyệt đối trong đánh giá năng lực, mọi quy trình AI phải tuân thủ nghiêm ngặt việc che giấu thông tin nhận dạng cá nhân (PII) trước khi đưa dữ liệu vào các mô hình ngôn ngữ lớn (LLM).

### 1.1. Danh Sách 11 Thuộc Tính Được Bảo Vệ (Protected Attributes)

Tuyệt đối **cấm** sử dụng hoặc căn cứ vào 11 thuộc tính được bảo vệ dưới đây để chấm điểm hoặc đưa ra bất kỳ nhận xét, đề xuất nào liên quan đến ứng viên. Mọi thông tin này phải được loại bỏ khỏi prompt trước khi gọi LLM:

| STT | Thuộc Tính Được Bảo Vệ | Mã Placeholder | Quy Định Ràng Buộc |
|-----|------------------------|----------------|-------------------|
| 1 | **Chủng tộc / Sắc tộc** | `[REDACTED_RACE]` | Cấm thiên lệch dựa trên chủng tộc hoặc nguồn gốc sắc tộc |
| 2 | **Dân tộc** | `[REDACTED_ETHNICITY]` | Không phân biệt dân tộc thiểu số hay đa số |
| 3 | **Giới tính** | `[REDACTED_GENDER]` | Không suy diễn hay phân biệt đối xử theo giới tính |
| 4 | **Tôn giáo / Tín ngưỡng** | `[REDACTED_RELIGION]` | Tự do tín ngưỡng, cấm đưa vào căn cứ phỏng vấn |
| 5 | **Tuổi tác / Ngày sinh** | `[REDACTED_AGE_DOB]` | Cấm phân biệt tuổi tác (ageism) |
| 6 | **Tình trạng khuyết tật** | `[REDACTED_DISABILITY]` | Cấm tất cả hình thức phân biệt đối xử do khuyết tật |
| 7 | **Tình trạng hôn nhân / Thai sản** | `[REDACTED_MARITAL]` | Không hỏi hay đánh giá về hôn nhân, con cái, thai kỳ |
| 8 | **Quê quán / Hộ khẩu / Nơi sinh** | `[REDACTED_HOMETOWN]` | Không phân biệt vùng miền, hộ khẩu, địa phương |
| 9 | **Xu hướng tính dục** | `[REDACTED_ORIENTATION]` | Cấm bất kỳ phân biệt đối xử dựa trên xu hướng tính dục |
| 10 | **Quan điểm chính trị** | `[REDACTED_POLITICS]` | Không căn cứ vào lập trường hay quan điểm chính trị |
| 11 | **Tình trạng kinh tế / Gia cảnh** | `[REDACTED_SOCIOECONOMIC]` | Cấm phân biệt dựa trên hoàn cảnh gia đình hay điều kiện tài chính |

> **Nguyên tắc bổ sung**: Mọi CV và transcript phải được ẩn danh hóa (`anonymize`) trước khi lưu trữ lâu dài. AI chỉ được phép thấy `candidate_code` (mã ẩn danh) trong suốt quá trình đánh giá.

---

## 2. Quy Tắc Che Ẩn Thông Tin Định Danh Cá Nhân (PII Redaction)

### 2.1. Các Trường PII Bắt Buộc Che Trước Khi Đưa Vào LLM

Trước khi đưa bất kỳ văn bản nào (CV, transcript, ghi chú) vào LLM, hệ thống **bắt buộc** phải chạy hàm `redact_pii()` để che các trường sau:

| Loại PII | Mã Placeholder Thay Thế | Phương Pháp Nhận Dạng |
|----------|--------------------------|------------------------|
| Họ và tên đầy đủ | `[REDACTED_NAME]` | NER (Named Entity Recognition) + regex pattern |
| Địa chỉ Email | `[REDACTED_EMAIL]` | Regex: `[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}` |
| Số điện thoại | `[REDACTED_PHONE]` | Regex: số VN (09x, 08x, +84) và quốc tế |
| Số CCCD / CMND / Hộ chiếu | `[REDACTED_ID_NUMBER]` | Regex số định danh 9-12 chữ số |
| Địa chỉ cư trú | `[REDACTED_ADDRESS]` | NER location + từ khóa (đường, phường, quận, tỉnh) |
| Link mạng xã hội | `[REDACTED_SOCIAL_URL]` | Regex URL chứa linkedin, facebook, github cá nhân |

### 2.2. Hàm `redact_pii` — Giao Kèo Triển Khai

```python
# Giao kèo (contract) bắt buộc — chi tiết triển khai tại src/harness/pii_redaction.py
def redact_pii(text: str, candidate_code: str) -> str:
    """
    Nhận text thô và trả về text đã được che tất cả PII.
    Đảm bảo:
    - Không có tên, email, SĐT, CCCD, địa chỉ, link MXH còn sót.
    - Thay thế bằng placeholder chuẩn để downstream pipeline nhận biết.
    - Ghi log audit trail: số lượng token bị redact, thời gian, candidate_code.
    """
    ...
```

### 2.3. Kiểm Chứng Tự Động (Redaction Verification)

Sau mỗi lần `redact_pii()`, hệ thống phải chạy `verify_redaction(output_text)` để xác nhận không còn pattern PII sót. Nếu tỉ lệ redaction < 100%, pipeline **phải dừng** và báo lỗi trước khi gọi LLM.

---

## 3. Chính Sách Lưu Trữ Dữ Liệu (Data Retention)

### 3.1. Thời Hạn Lưu Trữ Theo Loại Dữ Liệu

| Loại Dữ Liệu | Thời Hạn Lưu Trữ Tối Đa | Mục Đích & Quy Trình Tiêu Hủy |
|--------------|--------------------------|-------------------------------|
| Hồ sơ ứng viên (CV) | **90 ngày** | Phục vụ đối sánh năng lực JD-CV. Tự động xóa sau 90 ngày kể từ ngày nạp. |
| Biên bản phỏng vấn (Transcript) | **90 ngày** | Dùng cho Grader chấm điểm và Recruiter phúc khảo. Xóa vĩnh viễn sau 90 ngày. |
| Bản ghi âm / Âm thanh (Audio) | **30 ngày** | Phục vụ chuyển văn bản (STT) và kiểm tra chất lượng âm thanh. Tiêu hủy sau 30 ngày. |
| Bộ nhớ dài hạn ứng viên (Long-Term Memory) | **180 ngày** | Hỗ trợ ứng viên theo dõi lộ trình tiến bộ qua các phiên luyện tập (UC-09). Xóa sau 180 ngày. |
| Nhật ký quyết định tuyển dụng (HumanDecisionRecord) | **365 ngày** | Phục vụ audit và giải trình tuân thủ. Xóa sau 1 năm hoặc theo yêu cầu pháp lý. |

### 3.2. Cơ Chế Xóa Tự Động

- **Retention Job**: Một cron job chạy hàng ngày lúc 02:00 UTC quét và xóa mọi bản ghi vượt quá thời hạn lưu trữ.
- **Soft Delete → Hard Delete**: Dữ liệu được đánh dấu `deleted_at` (soft delete) trước, sau 7 ngày tiến hành xóa vật lý (hard delete) và xóa khỏi backup.
- **Audit Log**: Mọi sự kiện xóa phải được ghi vào `DataDeletionAuditLog` với timestamp, loại dữ liệu, và `candidate_code`.

---

## 4. Quyền Được Quên (Right to be Forgotten)

### 4.1. Yêu Cầu Xóa Tức Thì Từ Ứng Viên

Khi ứng viên yêu cầu xóa dữ liệu cá nhân, hệ thống phải thực hiện **trong vòng 72 giờ**:

1. **Xóa tức thì** toàn bộ dữ liệu ứng viên khỏi:
   - Database chính (CV, Transcript, Audio metadata)
   - Vector store / RAG index (Long-Term Memory)
   - Session store và cache (Redis)
   - Backup (đánh dấu để xóa trong lần backup rotation tiếp theo)

2. **Xác nhận bằng văn bản** (email) gửi đến ứng viên với danh sách các loại dữ liệu đã xóa và timestamp xóa.

3. **Ngoại lệ hợp pháp**: Dữ liệu bắt buộc giữ lại theo yêu cầu pháp luật (ví dụ: hồ sơ thuế, quyết định tuyển dụng theo luật lao động) sẽ được giữ trong kho lưu trữ cách ly (`legal_hold_vault`) và thông báo rõ ràng cho ứng viên.

### 4.2. Xóa Long-Term Memory (Agent Memory)

Ứng viên có quyền yêu cầu xóa **toàn bộ** lịch sử hội thoại và bộ nhớ dài hạn trong hệ thống Agent. Khi nhận yêu cầu:

- Xóa tất cả memory fragments trong `LongTermMemoryStore` gắn với `candidate_code`.
- Xóa embedding vectors tương ứng trong vector database.
- Đặt lại `candidate_profile` về trạng thái mới (blank slate).
- Ghi log `MemoryDeletionRecord` để audit.

---

## 5. Ranh Giới Quyết Định Tuyển Dụng (Human-In-The-Loop)

### 5.1. Nguyên Tắc Phê Duyệt Cuối Cùng

- **Vai trò của Agent**: Hệ thống chỉ tổng hợp dẫn chứng và đề xuất mức độ phù hợp (`recommend_level`: `strong`, `moderate`, `weak`, `insufficient`).
- **Quyền quyết định cuối cùng**: Quyết định tuyển dụng chính thức (`advance` / `reject` / `hold`) bắt buộc phải do con người mang vai trò `recruiter` hoặc trưởng bộ phận chuyên môn phê duyệt thông qua giao diện HITL.
- **Trách nhiệm giải trình**: Mọi quyết định của Recruiter — dù đồng thuận hay khác biệt với đề xuất của AI — đều phải được ghi vào `HumanDecisionRecord` kèm lý do cụ thể và tem thời gian ISO 8601.

### 5.2. Cấm Tự Động Hóa Quyết Định Tuyển Dụng

Tuyệt đối **nghiêm cấm** bất kỳ đoạn code hoặc workflow nào tự động gửi thư mời nhận việc, thư từ chối, hoặc thay đổi trạng thái ứng viên (`advance`/`reject`/`hold`) mà không có bước phê duyệt của con người.

Mọi vi phạm nguyên tắc này được coi là lỗi nghiêm trọng (Critical Bug P0) và phải được báo cáo ngay lập tức.
