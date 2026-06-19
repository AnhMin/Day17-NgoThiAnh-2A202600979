# Phân tích kết quả — Day 17: Memory Systems for AI Agent

Tài liệu này tóm tắt kết quả benchmark offline giữa **Baseline Agent** và **Advanced Agent**, cùng trade-off của hệ thống memory.

## Kết quả benchmark

### Standard Benchmark (`data/conversations.json`)

| Agent | Agent tokens | Prompt tokens | Cross-session recall | Response quality | Compactions |
|-------|-------------|---------------|---------------------|------------------|-------------|
| Baseline | 2,698 | 18,770 | 0.11 | 0.38 | 0 |
| Advanced | 2,580 | 24,996 | **0.79** | **0.85** | 0 |

### Long-Context Stress Benchmark (`data/advanced_long_context.json`)

| Agent | Agent tokens | Prompt tokens | Cross-session recall | Response quality | Compactions |
|-------|-------------|---------------|---------------------|------------------|-------------|
| Baseline | 861 | 26,321 | 0.00 | 0.30 | 0 |
| Advanced | 934 | **16,405** | **1.00** | **1.00** | **23** |

---

## 1. Vì sao Advanced có recall tốt hơn Baseline?

**Baseline** chỉ giữ message trong cùng `thread_id`. Khi benchmark hỏi recall ở thread mới, agent không còn ngữ cảnh cũ → recall **0.11** (standard) và **0.00** (stress).

**Advanced** có **persistent memory** qua `User.md`:
- Facts ổn định (tên, nghề, nơi ở, style trả lời…) được trích và lưu qua nhiều phiên
- Thread recall mới vẫn đọc được profile → recall **0.79** và **1.00**

Điều này phù hợp kịch bản benchmark: 10 cuộc hội thoại liên tiếp của cùng user `dungct`, có correction (Đà Nẵng → Huế, backend → MLOps) và câu hỏi recall chéo phiên.

---

## 2. Vì sao Advanced tốn hơn ở hội thoại ngắn?

Ở standard benchmark, Advanced xử lý **24,996 prompt tokens** so với **18,770** của Baseline (~+33%).

Nguyên nhân:
- Mỗi lượt Advanced phải inject thêm **`User.md`** vào prompt context
- Overhead đọc/ghi profile và ước lượng ngữ cảnh compact
- Hội thoại ngắn (10 lượt/conv) **chưa kích hoạt compaction** (Compactions = 0) → chưa có lợi thế nén, nhưng vẫn trả chi phí persistent memory

**Kết luận:** persistent memory đáng giá khi cần recall dài hạn; với hội thoại ngắn, trade-off là tốn token hơn để đổi lấy khả năng nhớ qua session.

---

## 3. Vì sao compact giúp Advanced ở hội thoại dài?

Stress benchmark có **16 lượt** rất dài (tin tức, preference, correction, nhiễu). Baseline giữ **toàn bộ lịch sử** → prompt tokens tăng tuyến tính → **26,321 tokens**.

Advanced kích hoạt **23 compactions**:
- Message cũ được tóm tắt vào `summary`
- Chỉ giữ N message gần nhất (mặc định 6)
- Prompt tokens giảm còn **16,405** (~**-38%** so với Baseline)

Compact chủ yếu tối ưu **`Prompt tokens processed`** — đúng mục tiêu lab: kiểm soát chi phí ngữ cảnh mà vẫn giữ facts quan trọng trong `User.md`.

---

## 4. Memory file tăng trưởng và rủi ro

`User.md` lưu tại `state/profiles/{user_id}.md` và **tăng dần** theo số facts được ghi:
- Mỗi phiên mới có thể thêm hoặc cập nhật field (name, location, profession…)
- Correction ghi đè fact cũ, nhưng file vẫn có thể dài nếu lưu quá nhiều preference

**Rủi ro:**
| Rủi ro | Mô tả |
|--------|-------|
| Lưu sai fact | Regex trích xuất có thể bắt nhầm từ câu hỏi hoặc nhiễu |
| File phình to | Nhiều user / nhiều facts → tăng prompt cost mỗi lượt |
| Fact cũ không xóa | Correction cập nhật field nhưng không có decay → metadata cũ có thể còn trong summary |
| Over-trust | Agent trả lời dựa profile sai mà user không hay review `User.md` |

---

## 5. Bonus: Confidence threshold

Advanced Agent có thêm **`profile_confidence_threshold`** (mặc định `0.6`, cấu hình qua `PROFILE_CONFIDENCE_THRESHOLD`):

- Mỗi fact trích ra được chấm điểm confidence (0–1) trước khi ghi `User.md`
- Fact rõ ràng (ví dụ `"mình tên là DũngCT"`) → ~0.9
- Fact mơ hồ (ví dụ `"có lẽ tuần sau…"`) → ~0.3, **không ghi**

**Lợi ích:** giảm lưu nhầm từ câu đùa / suy đoán (product manager, crypto…).

**Rủi ro thêm:** ngưỡng quá cao có thể bỏ sót fact hợp lệ but implicitly stated.

---

## Tóm tắt trade-off

```
Hội thoại ngắn  → Baseline rẻ hơn, Advanced recall cao hơn (+token profile)
Hội thoại dài   → Advanced + compact thắng rõ về prompt cost + vẫn nhớ profile
Production      → Cần guardrail: confidence threshold, review User.md, conflict handling
```

Hệ thống mạnh hơn Baseline nhưng phức tạp hơn — phù hợp khi **recall dài hạn** và **kiểm soát token** quan trọng hơn độ đơn giản.
