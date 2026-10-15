# Chính Sách Bảo Mật Dữ Liệu & Chống Thiên Lệch Tuyển Dụng

Tài liệu quy định các nguyên tắc bắt buộc về bảo mật thông tin cá nhân, thời hạn lưu trữ dữ liệu và trách nhiệm của hệ thống AI trong quy trình phỏng vấn (Đề tài 05 - Interview Agent).

## 1. Bảo Vệ Dữ Liệu Cá Nhân & Chống Thiên Lệch

Để bảo đảm tính công bằng tuyệt đối trong đánh giá năng lực, mọi quy trình AI phải tuân thủ nghiêm ngặt việc che giấu thông tin nhận dạng cá nhân (PII) trước khi đưa dữ liệu vào các mô hình ngôn ngữ lớn (LLM).

### 1.1. Danh Sách 11 Thuộc Tính Được Bảo Vệ Bị Cấm Dùng Khi Đánh Giá

Tuyệt đối cấm sử dụng hoặc căn cứ vào 11 thuộc tính được bảo vệ sau đây để chấm điểm hoặc đưa ra bất kỳ nhận xét, đề xuất nào liên quan đến ứng viên:

| STT | Thuộc Tính Được Bảo Vệ | Mã Placeholder Che Giấu | Quy Định Ràng Buộc |
|---|---|---|---|
| 1 | Họ và tên | `[REDACTED_NAME]` | Bắt buộc ẩn danh, chỉ định danh bằng `candidate_code` |
| 2 | Giới tính | `[REDACTED_GENDER]` | Không suy diễn hay phân biệt đối xử theo giới tính |
| 3 | Tuổi / Ngày sinh | `[REDACTED_AGE_DOB]` | Cấm phân biệt tuổi tác (ageism) |
| 4 | Quê quán / Nơi sinh | `[REDACTED_HOMETOWN]` | Không phân biệt vùng miền |
| 5 | Dân tộc | `[REDACTED_ETHNICITY]` | Cấm thiên lệch chủng tộc, sắc tộc |
| 6 | Tôn giáo | `[REDACTED_RELIGION]` | Tự do tín ngưỡng, cấm đưa vào căn cứ phỏng vấn |
| 7 | Tình trạng hôn nhân | `[REDACTED_MARITAL]` | Không hỏi hay đánh giá về hôn nhân, con cái |
| 8 | Ngoại hình / Ảnh | `[REDACTED_APPEARANCE]` | Không đánh giá chiều cao, cân nặng, nhan sắc |
| 9 | Số điện thoại | `[REDACTED_PHONE]` | Che số điện thoại cá nhân |
| 10 | Địa chỉ Email | `[REDACTED_EMAIL]` | Che email liên lạc cá nhân |
| 11 | Địa chỉ cư trú | `[REDACTED_ADDRESS]` | Che địa chỉ nhà riêng, nơi thường trú/tạm trú |

## 2. Quy Định Về Thời Hạn Lưu Trữ Dữ Liệu

Tất cả các tài liệu và thông tin thu thập trong quá trình phỏng vấn phải được tự động thu hồi hoặc xóa bỏ vĩnh viễn theo đúng hạn mức lưu trữ quy định dưới đây:

| Loại Dữ Liệu | Thời Hạn Lưu Trữ Tối Đa | Mục Đích Sử Dụng & Quy Trình Tiêu Hủy |
|---|---|---|
| Hồ sơ ứng viên (CV) | 90 ngày | Phục vụ đối sánh năng lực JD-CV. Tự động xóa sau 90 ngày. |
| Biên bản phỏng vấn (Transcript) | 90 ngày | Dùng cho Grader chấm điểm và Recruiter phúc khảo. Hết 90 ngày xóa vĩnh viễn. |
| Bản ghi âm / Âm thanh (Audio) | 30 ngày | Phục vụ chuyển văn bản (STT) và kiểm tra chất lượng. Tiêu hủy sau 30 ngày. |
| Bộ nhớ dài hạn ứng viên (Memory) | 180 ngày | Hỗ trợ ứng viên theo dõi lộ trình tiến bộ qua các phiên luyện tập (UC-09). Xóa sau 180 ngày. |

## 3. Ranh Giới Quyết Định Tuyển Dụng (Human-In-The-Loop)

Hệ thống AI đóng vai trò trợ lý hỗ trợ phỏng vấn và cung cấp góc nhìn khách quan dựa trên dữ liệu. AI tuyệt đối **không được phép** tự động ra quyết định tuyển dụng.

### 3.1. Nguyên Tắc Phê Duyệt Cuối Cùng

- **Vai trò của Agent**: Hệ thống chỉ tổng hợp dẫn chứng và đề xuất mức độ phù hợp (`recommend_level`: `strong`, `moderate`, `weak`, `insufficient`).
- **Quyền quyết định cuối cùng**: Quyết định tuyển dụng chính thức (`advance` / `reject` / `hold`) bắt buộc phải do con người mang vai trò chuyên viên tuyển dụng (`recruiter`) hoặc trưởng bộ phận chuyên môn phê duyệt thông qua giao diện Human-In-The-Loop (HITL).
- **Trách nhiệm giải trình**: Mọi quyết định của Recruiter dù đồng thuận hay khác biệt với đề xuất của AI đều phải được ghi nhận vào bản ghi `HumanDecisionRecord` kèm lý do cụ thể và tem thời gian ISO.
