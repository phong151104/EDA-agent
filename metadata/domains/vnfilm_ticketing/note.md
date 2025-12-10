# Ghi chú kiểm tra metadata vnfilm_ticketing

## Các bảng đã xóa
- `bank_pay_source`
- `campaign`
- `branch`
- `labels`
- `label_location`

## Các bảng trong vnfilm_v2 không cần metadata
- `order_change_log`
- `orders_bkl`
- `orders_temp_trash`

---

## Kết quả kiểm tra Relationship

### ✅ Đã xác nhận (DONE)
| Bảng | Ghi chú |
|------|---------|
| `vendor` | Dimension table (root), không có FK |
| `vendor_bank` | OK |
| `sessions` | Sửa `location` → `locations` |
| `pre_order` | Chỉ một số dòng thiếu dữ liệu |
| `pre_order_seat` | Dữ liệu đến tháng 3/2025 |
| `pre_order_map` | OK |
| `pre_order_customer` | Dữ liệu đến tháng 3/2025, có 3 dòng không khớp |
| `pre_order_concession` | Có 1 dòng sai |
| `pre_order_concession_map` | OK |
| `orders` | Đã xóa FK liên quan đến campaign |
| `order_seat` | Có ~4000 dòng không có order_id khớp, cần kiểm tra bank |
| `order_refund` | Có 14 dòng với order_id = -1 hoặc 0 |
| `order_film` | OK |
| `order_concession` | Cần kiểm tra |

### ❌ Loại bỏ (Ít/Sai dữ liệu)
| Bảng | Lý do |
|------|-------|
| `sub_campaign` | Ít dữ liệu |
| `sub_campaign_condition` | Ít dữ liệu |
| `sub_campaign_concession` | Ít dữ liệu |
| `seat_definition` | Ít dữ liệu, dữ liệu test |
| `schedule` | Dữ liệu sai |
