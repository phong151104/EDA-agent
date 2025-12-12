# VNFILM Ticketing Mock Data

## Overview
This folder contains **27 CSV files** with mock data for the `vnfilm_ticketing` domain.
All foreign key relationships are validated and consistent for JOIN operations.

## File Naming Convention
- `lakehouse.lh_vnfilm_v2.<table_name>.csv` - Main tables
- `lakehouse.cdp_mart.<table_name>.csv` - CDP tables

## Data Summary

| Table | Rows | Description |
|-------|------|-------------|
| bank | 8 | Ngân hàng/kênh thanh toán |
| vendor | 6 | Đối tác rạp (CGV, Lotte, Galaxy...) |
| locations | 5 | Địa điểm (HN, HCM, ĐN...) |
| cinema | 8 | Rạp chiếu phim |
| film | 6 | Thông tin phim |
| sessions | 50 | Suất chiếu |
| concession | 6 | Combo bắp nước |
| **orders** | **100** | **Đơn hàng (core fact)** |
| order_film | 100 | Extension thông tin phim/rạp |
| order_seat | ~250 | Chi tiết ghế |
| order_concession | ~90 | Chi tiết combo |
| order_refund | 15 | Hoàn tiền |
| pre_order | 50 | Pre-order |
| notify_email | 50 | Log email |
| notify_ott | 50 | Log OTT |
| notify_sms | 30 | Log SMS |
| dim_campaign | 5 | Chiến dịch marketing |

## Foreign Key Relationships ✅

```
orders.bank_id → bank.id
orders.id → order_film.id (1-1)
orders.id → order_seat.order_id (1-n)
orders.id → order_concession.order_id (1-n)
orders.id → order_refund.order_id (1-n)
orders.id → notify_*.order_id
order_film.cinema_id → cinema.id
order_film.vendor_id → vendor.id
cinema.vendor_id → vendor.id
cinema.location_id → locations.id
sessions.cinema_id → cinema.id
sessions.film_id → film.id
```

## Usage Example

```python
import pandas as pd

# Load tables
orders = pd.read_csv('lakehouse.lh_vnfilm_v2.orders.csv')
order_film = pd.read_csv('lakehouse.lh_vnfilm_v2.order_film.csv')
bank = pd.read_csv('lakehouse.lh_vnfilm_v2.bank.csv')

# Join orders with order_film
df = orders.merge(order_film, on='id', suffixes=('', '_film'))

# Join with bank
df = df.merge(bank, left_on='bank_id', right_on='id', suffixes=('', '_bank'))

print(df[['id', 'vnpay_final_amount', 'film_name_vi', 'cinema_name_vi', 'name_vi']].head())
```

## Regenerate Data
```bash
python generate_mock_data.py
```
