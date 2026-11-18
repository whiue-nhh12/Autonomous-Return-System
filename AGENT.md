# 🤖 AGENT.md — QUY TẮC VÀ CHỈ DẪN DÀNH CHO AI AGENT (CASHBACK SYSTEM)

File này quy định các nguyên tắc, tiêu chuẩn kỹ thuật và quy trình xử lý bắt buộc cho mọi AI Agent khi đọc context, lập kế hoạch, và sinh mã nguồn trong dự án **Hệ Thống Agent Hoàn Tiền (Cashback & Refund System)**.

## 1. 🎯 TỔNG QUAN HỆ THỐNG & SỨ MỆNH

Dự án là một hệ thống **Hoàn Tiền (Cashback)** tự động hóa dựa trên Agent. Hệ thống chịu trách nhiệm xử lý các luồng tài chính, tính toán tỷ lệ hoàn tiền, tích hợp với cổng thanh toán/đối tác, và quản lý ví/số dư của người dùng.

> ⚠️ **CẢNH BÁO TÀI CHÍNH:** Mọi sai sót trong logic tính toán, bất đồng bộ số dư hoặc thiếu Idempotency đều có thể dẫn đến **thất thoát tài sản**. Do đó, sự chính xác và tính toàn vẹn dữ liệu được đặt lên hàng đầu.

## 2. 🛡️ BỘ NGUYÊN TẮC CỐT LÕI (NON-NEGOTIABLE RULES)

### 2.1. Nhất Quán Tài Chính & Idempotency (Tính Độc Công)

- **TUYỆT ĐỐI KHÔNG** dùng kiểu số thực (Float / Double) để lưu trữ hoặc tính toán tiền tệ. Bắt buộc dùng kiểu số nguyên định dạng đơn vị nhỏ nhất (ví dụ: `Cents`, `VND`) hoặc thư viện hỗ trợ số thực chuẩn tài chính (`Decimal`, `BigNumber`, `BigDecimal`).
- Mọi API / Hàm xử lý hoàn tiền, cộng/trừ số dư phải yêu cầu **Idempotency Key** (`idempotency_key`).
- Đảm bảo cơ chế **At-Most-Once Execution**: Cùng một yêu cầu hoàn tiền gửi lại nhiều lần chỉ được thực thi duy nhất 01 lần.

### 2.2. Kiểm Soát Tranh Chấp & Đồng Thời (Concurrency Control)

- Mọi thao tác thay đổi số dư ví (Balance Update) phải sử dụng **Row Locking** (Pessimistic Locking `SELECT FOR UPDATE`) hoặc **Distributed Lock** (Redis Redlock).
- Thực hiện mô hình Ghi nợ kép (Double-Entry Bookkeeping) hoặc Trạng thái nâng cấp an toàn (Ledger-based state machine).

### 2.3. Nhật Ký & Ghi Vết Bất Biến (Audit Trail)

- Không bao giờ thực hiện câu lệnh `UPDATE` trực tiếp làm thay đổi số dư mà không tạo một giao dịch lịch sử tương ứng (`ledger_entries` / `audit_logs`).
- Mọi hành động của Agent (từ nhận diện đơn hàng, tính Cashback, phê duyệt đến hoàn tiền) đều phải ghi nhật ký vết với đầy đủ metadata (Session ID, Rule ID, Timestamp, State Before/After).

### 2.4. Khả Năng Khôi Phục & Xử Lý Lỗi (Fault Tolerance & Rollbacks)

- Tất cả giao dịch tài chính liên quan đến cơ sở dữ liệu phải nằm trong một **Database Transaction** nguyên tử (ACID).
- Nếu lời gọi tới bên thứ 3 (Payment Gateway, Partner API) thất bại hoặc Timeout, phải áp dụng mô hình **Saga Pattern** hoặc **Outbox Pattern** để đảm bảo tính nhất quán cuối cùng (Eventual Consistency).

## 3. 📝 QUY TRÌNH PHIÊN THỰC THI BẮT BUỘC (AGENT EXECUTION PROTOCOL)

Khi bắt đầu một phiên làm việc (Session) hoặc nhận một nhiệm vụ sinh code, Agent **BẮT BUỘC** tuân thủ 5 bước sau:

```text
[1. Context Analysis] ➡️ [2. Plan & Blueprint] ➡️ [3. TDD / Implementation] ➡️ [4. Security & Audit Check] ➡️ [5. Review & Output]
```

1. **Bước 1: Phân tích bối cảnh (Context Analysis):**
   - Đọc hiểu yêu cầu nghiệp vụ hoàn tiền (Cashback Rule, Tiêu chí hoàn tiền, Hạn mức, Trạng thái đơn hàng).
   - Xác định các thực thể và DB schema ảnh hưởng.
2. **Bước 2: Lập kế hoạch (Plan & Blueprint):**
   - Trình bày ngắn gọn phương án thiết kế trước khi viết code dài.
   - Liệt kê rõ ràng các trường hợp biên (Edge Cases): Lỗi mạng khi hoàn tiền, Đơn hàng bị hủy sau khi hoàn tiền, Race Condition khi gửi đồng thời 2 request.
3. **Bước 3: Viết mã nguồn & Unit Test (Implementation):**
   - Viết code tuân thủ Clean Architecture / Layered Architecture.(Ở MỨC TIỆM CẬN PRODUCTION)
   - **Bắt buộc** kèm theo Unit Test / Integration Test mô phỏng các kịch bản thành công và thất bại.
4. **Bước 4: Kiểm tra tính an toàn (Safety Check):**
   - Đảm bảo không có nguy cơ SQL Injection, IDOR, Race Condition, hoặc lộ thông tin nhạy cảm (PII/Card Details).
5. **Bước 5: Tổng kết thay đổi (Output Summary):**
   - Nêu rõ những file đã chỉnh sửa, cấu hình mới (nếu có), và hướng dẫn chạy test kiểm thử.

## 4. 📐 QUY CHUẨN CODE VÀ THIẾT KẾ (CODING STANDARDS)

### 4.1. Cấu Trúc Trạng Thái Hoàn Tiền (State Machine)

Luồng trạng thái của một yêu cầu Hoàn Tiền (Cashback Request) phải tuân thủ nghiêm ngặt trạng thái sau, **không được nhảy bước**:

```text
[PENDING] ➡️ [PROCESSING] ➡️ [APPROVED / REJECTED] ➡️ [COMPLETED / FAILED]
                                 │
                                 └──> [REFUNDED / REVERSED] (Nếu đơn hàng hủy)
```

### 4.2. Xử Lý Lỗi & Exception Handling

- Sử dụng **Custom Domain Exceptions** rõ ràng (ví dụ: `InsufficientBalanceException`, `DuplicateTransactionException`, `CashbackLimitExceededException`).
- Không nuốt ngoại lệ (`catch (Exception e) {}`). Luôn log thông tin lỗi kèm theo Context (`user_id`, `order_id`, `transaction_id`).

### 4.3. Logging & Monitoring

- Chỉ log dữ liệu an toàn. **TUYỆT ĐỐI KHÔNG** log token, mật khẩu, thông tin thẻ ngân hàng của người dùng.
- Log định dạng JSON chuẩn structured logging để phục vụ việc đẩy lên ELK / Datadog.

## 5. 🧪 TIÊU CHUẨN KIỂM THỬ (TESTING MANDATE)

Mọi đoạn code do Agent tạo ra liên quan đến nghiệp vụ Cashback phải thỏa mãn:

1. **Unit Tests:**
   - Test tính toán Cashback chuẩn xác theo công thức phần trăm/cố định/tier level.
   - Test kiểm tra vượt hạn mức Cashback ngày/tháng của người dùng.
2. **Integration / Concurrency Tests:**
   - Mô phỏng 2 hoặc nhiều request tính Cashback/rút tiền đồng thời cho 1 ví (Race Condition Check).
   - Mô phỏng Retry do Timeout mạng khi gọi API Cổng thanh toán.

## 6. ✅ CHECKLIST KHI AGENT HOÀN THÀNH CODE (DEFINITION OF DONE)

Trước khi gửi phản hồi cho lập trình viên, Agent phải tự rà soát danh sách sau:

- [ ] Tất cả biến số dư/tiền tệ dùng kiểu dữ liệu chuẩn (không dùng float/double).
- [ ] Có Idempotency check ở đầu các hàm mutation.
- [ ] Mọi câu lệnh cập nhật số dư nằm trong DB Transaction + Lock.
- [ ] Không chứa thông tin bí mật (Hardcoded Secrets / API Keys).
- [ ] Đã tạo/cập nhật đầy đủ Unit Test cho luồng logic mới.
- [ ] Code có comment giải thích rõ ràng các luồng xử lý tài chính phức tạp.
