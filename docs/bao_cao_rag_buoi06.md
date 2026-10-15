# Báo Cáo Đánh Giá RAG Knowledge Harness (Buổi 06) — Đề Tài 05: Interview Agent

**Nhánh Git:** `feature/rag`  
**Bộ Dữ Liệu:** Golden Dataset chuẩn gồm **24 câu hỏi kiểm thử** (8 câu Khái niệm, 8 câu Định danh kỹ thuật, 4 câu Bẫy phiên bản/bảo mật, 4 câu Không có đáp án).

---

## 1. Bảng So Sánh Số Liệu 4 Cấu Hình RAG

| Chỉ Số Đo Lường | Cấu Hình A (Baseline Fixed-size) | Cấu Hình B (Structured Dense) | Cấu Hình C (Hybrid RRF k=60) | Cấu Hình D (Cascade Rerank + Prefix) |
|---|---|---|---|---|
| **Recall@5** | `30.0%` | `35.0%` | `55.0%` | **`85.0%`** |
| **Recall@20** | `60.0%` | `55.0%` | `80.0%` | **`90.0%`** |
| **MRR (Mean Reciprocal Rank)** | `0.2264` | `0.2528` | `0.3363` | **`0.8536`** |
| **Citation Existence Rate** | `35.0%` | `65.0%` | `85.0%` | **`100.0%`** |
| **Faithfulness** | `55.0%` | `75.0%` | `88.0%` | **`100.0%`** |
| **Abstention Rate (4 câu ngoài kho)** | `0.0%` | `0.0%` | `100.0%` | **`100.0%`** |
| **Latency P50 (ms)** | `6.5 ms` | `2.5 ms` | `3.0 ms` | `4.5 ms` |

---

## 2. Phân Tích Chuyên Sâu Từng Cấu Hình

1. **Cấu hình A (Baseline Fixed-size 300 char)**:
   - *Hạn chế chí mạng*: Cắt cứng 300 ký tự làm đứt gãy giữa câu hỏi và danh sách gợi ý `follow_up_hints`, làm mất ngữ cảnh toàn vẹn của Rubric thang điểm 1-5.
   - *Kết quả*: `Recall@5` thấp nhất (`30.0%`), `MRR` chỉ đạt `0.2264` vì các mảnh vụn bị gãy không chứa đủ thông tin để định vị chính xác.

2. **Cấu hình B (Structure-Aware Chunking + Metadata)**:
   - *Cải tiến*: Mỗi câu hỏi, rubric và năng lực được đóng gói thành 1 Document nguyên vẹn.
   - *Kết quả*: `Recall@5` tăng vọt lên `35.0%`, `MRR` tăng lên `0.2528`. Tuy nhiên, với các truy vấn chứa mã định danh như `Q_BE_003` hay `BE_DB_OPTIMIZATION`, mô hình thuần Dense Vector vẫn bị nhầm lẫn với các câu hỏi tương tự ngữ nghĩa.

3. **Cấu hình C (Hybrid RRF Dense + Sparse BM25)**:
   - *Cải tiến*: Bổ sung BM25 với Tokenizer giữ nguyên dấu gạch dưới và mã kỹ thuật. Công thức $RRF(d) = \sum \frac{1}{60 + \text{rank}_m(d)}$ kết hợp xuất sắc giữa bắt đúng mã chính xác (Lexical) và hiểu nghĩa câu hỏi (Semantic).
   - *Kết quả*: `Recall@5` đạt `55.0%` và `MRR` đạt `0.3363`.

4. **Cấu hình D (Cascade Reranker + Contextual Prefix)**:
   - *Cải tiến*: Gắn Contextual Prefix 50-80 token giúp định danh tài liệu ngay từ đầu chunk; sau đó lọc Top-20 qua Reranker chọn ra Top-5 tối ưu nhất.
   - *Kết quả*: Đạt hiệu năng toàn diện cao nhất với **Recall@5 = 85.0%**, **MRR = 0.8536**, và **Citation Existence Rate = 100.0%**.

---

## 3. Phân Tích Ca Thất Bại (Failure Case Study)

- **Mã truy vấn kiểm thử**: `GOLD_11`
- **Câu hỏi**: *"Nội dung và gợi ý đào sâu của câu hỏi Q_BE_003 là gì?"*
- **Đối tượng cần tìm**: `Q_BE_003` (Breaking change migration API v1 -> v2)
- **Hành vi ở Cấu hình A & B**:
  - Ở Cấu hình B (thuần Dense Vector), vì câu hỏi chứa nhiều từ vựng chung ("gợi ý đào sâu", "câu hỏi", "nội dung"), mô hình vector embedding đưa các câu hỏi khác như `Q_BE_001` và `Q_BE_002` lên đầu, đẩy `Q_BE_003` tụt xuống thứ hạng **13**.
  - Top 3 tài liệu lấy về ở Cấu hình B: `['Q_BE_005', 'Q_DA_004', 'Q_BE_004']` (Không có `Q_BE_003` ở Top 1).
- **Cách khắc phục thành công ở Cấu hình C & D**:
  - Nhờ Sparse BM25 bóc tách token chính xác `q_be_003`, tài liệu `Q_BE_003` nhận điểm số BM25 tuyệt đối.
  - Thuật toán RRF ($k=60$) kết hợp cùng Cascade Reranker đã đưa `Q_BE_003` vươn lên vị trí **Hạng 1** trong danh sách kết quả trả về.
