"""
Script to generate mock CSV data for VNFILM Ticketing domain
All tables with correct foreign key relationships
"""
import csv
import os
import random
from datetime import datetime, timedelta
import hashlib

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Helper functions
def hash_value(val):
    return hashlib.sha256(str(val).encode()).hexdigest()[:32]

def random_date(start_days_ago=365, end_days_ago=0):
    days = random.randint(end_days_ago, start_days_ago)
    return (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S.000000')

def write_csv(filename, headers, rows):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"Created: {filename} ({len(rows)} rows)")

# ========== DIMENSION TABLES ==========

def gen_bank():
    headers = ['id','code','name_vi','name_en','gateway_bank_code','ott_bank_code','promotion_bank_code',
               'ott_support','sms_support','promotion_support','status','created_date','created_by',
               'modified_date','modified_by','service_code','provider_code','ott_pilot_support',
               'sms_pilot_support','promotion_pilot_support','channel_ids','email_pilot_support',
               'email_pilot_support_hash','email_support','email_support_hash','logo','live_activity_support']
    banks = [
        (1,'vnpay','Ví VNPAY','VNPAY Wallet','970436','VNPAY','VNPAY'),
        (2,'vcb','Vietcombank','Vietcombank','970436','VCB','VCB'),
        (3,'bidv','BIDV','BIDV','970418','BIDV','BIDV'),
        (4,'vtb','VietinBank','VietinBank','970415','VTB','VTB'),
        (5,'mbbank','MB Bank','MB Bank','970422','MBB','MBB'),
        (6,'tpb','TPBank','TPBank','970423','TPB','TPB'),
        (7,'acb','ACB','ACB','970416','ACB','ACB'),
        (8,'shb','SHB','SHB','970443','SHB','SHB'),
    ]
    rows = []
    for b in banks:
        rows.append([b[0],b[1],b[2],b[3],b[4],b[5],b[6],'supported','supported','supported','ACTIVE',
                     random_date(365,30),'system',random_date(30,0),'system','VNFILM','VNPAY',
                     'supported','supported','supported','1,2,3','supported',hash_value(b[1]),'supported',
                     hash_value(b[1]),f'https://logo.vnpay.vn/{b[1]}.png','supported'])
    write_csv('lakehouse.lh_vnfilm_v2.bank.csv', headers, rows)
    return [b[0] for b in banks]

def gen_vendor():
    headers = ['id','name_vi','name_en','logo','image','description','description_vi','description_en',
               'display_order','local_domain','project_domain','secret','max_seat','max_concession',
               'multiple_seat_support','concession_support','payment_order_lead_second','create_order_lead_second',
               'seat_discount_percent','seat_discount_tax','concession_discount_percent','concession_discount_tax',
               'sub_fee_ignored_film','sub_fee_ignored_cinema','sub_fee_ignored_session','sub_fee_ignored_vendor_seat_type',
               'sub_fee_ignored_room_id','job_status','job_delay_first_time','job_interval_cinema','job_interval_film',
               'job_interval_session','job_interval_concession','status','created_date','created_by','created_by_hash',
               'modified_date','modified_by']
    vendors = [
        (1,'CGV','CGV Cinemas',1),
        (2,'Lotte Cinema','Lotte Cinema',2),
        (3,'Galaxy','Galaxy Cinema',3),
        (4,'BHD','BHD Star',4),
        (5,'Cinestar','Cinestar',5),
        (6,'Beta','Beta Cinemas',6),
    ]
    rows = []
    for v in vendors:
        rows.append([v[0],v[1],v[2],f'https://logo.vnpay.vn/{v[2].lower().replace(" ","")}.png',
                     f'https://img.vnpay.vn/{v[2].lower()}.jpg',f'{v[1]} - Hệ thống rạp',v[1],v[2],v[3],
                     f'http://api.{v[2].lower().replace(" ","")}.local',f'https://api.{v[2].lower().replace(" ","")}.vn',
                     'SECRET_KEY',10,5,'supported','supported',900,60,0.05,0.1,0.03,0.1,'','','','','',
                     'active',60,3600,1800,600,3600,'active',random_date(365,180),'system',hash_value('system'),
                     random_date(30,0),'system'])
    write_csv('lakehouse.lh_vnfilm_v2.vendor.csv', headers, rows)
    return [v[0] for v in vendors]

def gen_locations():
    headers = ['id','display_order','status','created_date','modified_date','modified_by',
               'promotion_location_code','city_name','latitude','longitude','created_by','created_by_hash']
    locs = [
        (1,'HN','Hà Nội',21.0285,105.8542),
        (2,'HCM','Hồ Chí Minh',10.8231,106.6297),
        (3,'DN','Đà Nẵng',16.0544,108.2022),
        (4,'HP','Hải Phòng',20.8449,106.6881),
        (5,'CT','Cần Thơ',10.0452,105.7469),
    ]
    rows = []
    for i,l in enumerate(locs):
        rows.append([l[0],i+1,'active',random_date(365,180),random_date(30,0),'system',l[1],l[2],l[3],l[4],'system',hash_value('system')])
    write_csv('lakehouse.lh_vnfilm_v2.locations.csv', headers, rows)
    return [l[0] for l in locs]

def gen_cinema(vendor_ids, location_ids):
    headers = ['id','name_vi','name_en','keyword','logo','address','description_vi','description_en',
               'latitude','longitude','vendor_id','display_order','branch_id','location_id',
               'vendor_cinema_id','status','created_date','created_by','modified_date','modified_by','phone','phone_hash']
    cinemas = [
        (1,'CGV Vincom Bà Triệu','CGV Vincom Ba Trieu',1,1,'191 Bà Triệu, Hai Bà Trưng, Hà Nội'),
        (2,'CGV Aeon Long Biên','CGV Aeon Long Bien',1,1,'27 Cổ Linh, Long Biên, Hà Nội'),
        (3,'Lotte Cinema Landmark','Lotte Landmark 81',2,2,'772 Điện Biên Phủ, Bình Thạnh, HCM'),
        (4,'Galaxy Nguyễn Du','Galaxy Nguyen Du',3,2,'116 Nguyễn Du, Q1, HCM'),
        (5,'BHD Star Vincom Đà Nẵng','BHD Vincom Da Nang',4,3,'910A Ngô Quyền, Sơn Trà, Đà Nẵng'),
        (6,'Beta Giải Phóng','Beta Giai Phong',6,1,'384 Giải Phóng, Thanh Xuân, Hà Nội'),
        (7,'CGV Crescent Mall','CGV Crescent Mall',1,2,'101 Tôn Dật Tiên, Q7, HCM'),
        (8,'Cinestar Quốc Thanh','Cinestar Quoc Thanh',5,2,'271 Nguyễn Trãi, Q1, HCM'),
    ]
    rows = []
    for c in cinemas:
        rows.append([c[0],c[1],c[2],c[1].lower(),f'https://logo.vnpay.vn/cinema{c[0]}.png',c[5],
                     f'Rạp {c[1]}',f'Cinema {c[2]}',21.0+random.random(),105.8+random.random(),
                     c[3],c[0],c[0],c[4],f'VND-C{c[0]:03d}','active',random_date(365,180),'system',
                     random_date(30,0),'system','0901234567',hash_value('0901234567')])
    write_csv('lakehouse.lh_vnfilm_v2.cinema.csv', headers, rows)
    return [c[0] for c in cinemas]

def gen_film(vendor_ids):
    headers = ['id','name_vi','name_en','trailer_url','poster_url','banner_url','country','category',
               'display_order','duration','language','description_vi','description_en','publish_date',
               'pg_rating','age','keyword','director','actors','status_id','vendor_id','mapping_id',
               'vendor_film_id','vendor_film_name_vi','vendor_film_name_en','mapping_status',
               'created_date','created_by','modified_date','modified_by','alias_id','category_en','language_en','country_en','modified_by_hash']
    films = [
        (1,'Quật Mộ Trùng Ma','The Exorcism of God','Kinh dị',120,'Mỹ','C18',18,'James Wan','Russell Crowe'),
        (2,'Nhím Sonic 3','Sonic the Hedgehog 3','Hoạt hình',110,'Mỹ','P',0,'Jeff Fowler','Jim Carrey'),
        (3,'Mufasa: Vua Sư Tử','Mufasa: The Lion King','Hoạt hình',118,'Mỹ','P',0,'Barry Jenkins','Aaron Pierre'),
        (4,'Kraven: Thợ Săn Thủ Lĩnh','Kraven the Hunter','Hành động',127,'Mỹ','C18',18,'J.C. Chandor','Aaron Taylor'),
        (5,'Linh Miêu','Linh Mieu','Kinh dị',100,'Việt Nam','C16',16,'Lưu Thành Luân','Hồng Đào'),
        (6,'Cô Dâu Hào Môn','The Grand Wedding','Hài',105,'Việt Nam','P',0,'Vũ Ngọc Đãng','Thu Trang'),
    ]
    rows = []
    for f in films:
        # f = (id, name_vi, name_en, category, duration, country, pg_rating, age, director, actors)
        # indices: 0     1        2        3         4        5       6         7      8        9
        vid = random.choice(vendor_ids)
        rows.append([f[0],f[1],f[2],f'https://youtube.com/watch?v={f[0]}',f'https://img.vnpay.vn/film{f[0]}.jpg',
                     f'https://img.vnpay.vn/banner{f[0]}.jpg',f[5],f[3],f[0],f[4],f[5],f'Phim {f[1]}',
                     f'Movie {f[2]}',random_date(60,0),f[6],f[7],f[1].lower(),f[8],f[9],1,vid,f[0],
                     f'VND-F{f[0]:03d}',f[1],f[2],'mapped',random_date(90,30),'system',random_date(30,0),
                     'system',f[0],1,1,1,hash_value('system')])
    write_csv('lakehouse.lh_vnfilm_v2.film.csv', headers, rows)
    return [f[0] for f in films]

def gen_sessions(cinema_ids, film_ids, location_ids, vendor_ids):
    headers = ['id','session_time','session_version','session_dimension','version_type','session_type',
               'vendor_id','vendor_session_id','vendor_film_id','vendor_cinema_id','vendor_room_id',
               'vendor_room_name_vi','vendor_room_name_en','film_id','cinema_id','location_id','status',
               'created_date','created_by','created_by_hash','modified_date','modified_by']
    rows = []
    for i in range(1, 51):
        cid = random.choice(cinema_ids)
        fid = random.choice(film_ids)
        lid = random.choice(location_ids)
        vid = random.choice(vendor_ids)
        stime = (datetime.now() + timedelta(days=random.randint(0,14), hours=random.randint(9,22))).strftime('%Y-%m-%d %H:%M:%S.000000')
        dim = random.choice(['Film2D','Film3D','IMAX'])
        ver = random.choice(['subtitling','dubbing'])
        rows.append([i,stime,ver,dim,f'VER_{dim}_{ver.upper()[:3]}','nowShowing',vid,f'VND-S{i:05d}',
                     f'VND-F{fid:03d}',f'VND-C{cid:03d}',f'R{random.randint(1,10):02d}',
                     f'Phòng {random.randint(1,10)}',f'Room {random.randint(1,10)}',fid,cid,lid,'active',
                     random_date(30,0),'system',hash_value('system'),random_date(7,0),'system'])
    write_csv('lakehouse.lh_vnfilm_v2.sessions.csv', headers, rows)
    return list(range(1, 51))

def gen_concession(vendor_ids, cinema_ids):
    headers = ['id','vendor_concession_id','name_vi','name_en','image','description_vi','description_en',
               'price','vendor_id','cinema_id','vendor_cinema_id','status','created_date','created_by','modified_date','modified_by']
    items = [
        (1,'Combo Couple','Couple Combo',89000,'1 Bắp Lớn + 2 Nước Lớn'),
        (2,'Combo Solo','Solo Combo',59000,'1 Bắp Vừa + 1 Nước Vừa'),
        (3,'Bắp Rang Bơ Lớn','Large Popcorn',45000,'Bắp rang bơ size lớn'),
        (4,'Coca Cola Lớn','Large Coca Cola',35000,'Coca Cola size lớn'),
        (5,'Combo Family','Family Combo',139000,'2 Bắp Lớn + 4 Nước'),
        (6,'Nachos Phô Mai','Cheese Nachos',55000,'Nachos với phô mai'),
    ]
    rows = []
    for item in items:
        vid = random.choice(vendor_ids)
        cid = random.choice(cinema_ids)
        rows.append([item[0],f'CON{item[0]:03d}',item[1],item[2],f'https://img.vnpay.vn/con{item[0]}.jpg',
                     item[4],item[4],item[3],vid,cid,f'VND-C{cid:03d}','active',random_date(180,30),
                     'system',random_date(30,0),'system'])
    write_csv('lakehouse.lh_vnfilm_v2.concession.csv', headers, rows)
    return [item[0] for item in items]

def gen_bank_seat_definition(bank_ids, vendor_ids):
    headers = ['id','bank_id','seat_definition_id','mapping_type','status','created_date','created_by',
               'modified_date','modified_by','vendor_id']
    rows = []
    i = 1
    for bid in bank_ids[:4]:
        for vid in vendor_ids[:3]:
            for mtype in ['NORMAL','VIP','COUPLE']:
                rows.append([i,bid,i,mtype,'ACTIVE',random_date(180,30),'system',random_date(30,0),'system',vid])
                i += 1
    write_csv('lakehouse.lh_vnfilm_v2.bank_seat_definition.csv', headers, rows)

def gen_dim_campaign(bank_ids):
    headers = ['campaign_id','campaign_name','campaign_code','master_campaign_id','service_code',
               'campaign_start_at','campaign_end_at','num_cus_target','bank_code','period','campvoucherids',
               'vouchercodes','campaign_start_at_gmt_7','campaign_end_at_gmt_7','created_date','end_date',
               'master_campaign_code','master_campaign_name']
    campaigns = [
        ('CAMP001','Khuyến mãi Tết 2025','TET2025',50000,'vnpay'),
        ('CAMP002','Flash Sale Thứ 6','FLASH_FRI',30000,'vcb'),
        ('CAMP003','Ưu đãi Thành viên','MEMBER_VIP',20000,'bidv'),
        ('CAMP004','Combo Siêu Rẻ','COMBO_SALE',40000,'vnpay'),
        ('CAMP005','Sinh nhật VNPAY','BDAY_VNPAY',100000,'vnpay'),
    ]
    rows = []
    for c in campaigns:
        start = random_date(60,30)
        end = random_date(30,0)
        rows.append([c[0],c[1],c[2],'MASTER001','VNFILM',start,end,c[3],c[4],'2025-Q1',
                     f'["{c[0]}_V1","{c[0]}_V2"]',f'["{c[2]}50","{c[2]}100"]',start,end,
                     random_date(90,60),end[:10],'MASTER001','Master Campaign 2025'])
    write_csv('lakehouse.cdp_mart.dim_campaign.csv', headers, rows)
    return [c[0] for c in campaigns]

def gen_vendor_bank(vendor_ids, bank_ids):
    headers = ['id','bank_id','bank_name_vi','bank_name_en','vendor_id','vendor_name_vi','vendor_name_en',
               'status','created_date','created_by','created_by_hash','modified_date','modified_by']
    rows = []
    i = 1
    bank_names = {1:('Ví VNPAY','VNPAY'),2:('Vietcombank','VCB'),3:('BIDV','BIDV'),4:('VietinBank','VTB')}
    vendor_names = {1:('CGV','CGV'),2:('Lotte Cinema','Lotte'),3:('Galaxy','Galaxy'),4:('BHD','BHD')}
    for vid in vendor_ids[:4]:
        for bid in bank_ids[:4]:
            vn = vendor_names.get(vid,('Vendor','Vendor'))
            bn = bank_names.get(bid,('Bank','Bank'))
            rows.append([i,bid,bn[0],bn[1],vid,vn[0],vn[1],'active',random_date(180,30),'system',
                         hash_value('system'),random_date(30,0),'system'])
            i += 1
    write_csv('lakehouse.lh_vnfilm_v2.vendor_bank.csv', headers, rows)

def gen_cinema_favourite(bank_ids, cinema_ids, vendor_ids):
    headers = ['id','bank_id','cinema_id','vendor_id','created_date','created_by','bank_identity','bank_identity_hash']
    rows = []
    for i in range(1, 21):
        rows.append([i,random.choice(bank_ids),random.choice(cinema_ids),random.choice(vendor_ids),
                     random_date(90,0),'system',f'user{i:03d}@bank.com',hash_value(f'user{i:03d}')])
    write_csv('lakehouse.lh_vnfilm_v2.cinema_favourite.csv', headers, rows)

# ========== FACT TABLES ==========

def gen_orders(bank_ids):
    headers = ['id','pay_code','status','issue_status','source_booking','expire_date','bank_id',
               'bank_identity','bank_identity_hash','bank_channel_id','vendor_seat_original_amount',
               'vendor_seat_charge_amount','vendor_seat_discount_amount','vendor_seat_sub_amount',
               'vendor_seat_reconciliation_amount','vendor_con_original_amount','vendor_con_reconciliation_amount',
               'vendor_total_reconciliation_amount','vnpay_seat_amount','vnpay_concession_amount',
               'vnpay_total_amount','vnpay_final_amount','vnpay_seat_profit','vnpay_concession_profit',
               'vendor_seat_fee','vnpay_seat_fee','promotion_id','promotion_code','promotion_amount',
               'promotion_vnpay_amount','promotion_vendor_amount','promotion_bank_amount',
               'promotion_reconciliation_code','billing_id','billing_amount','billing_pay_source',
               'billing_status','billing_data','vendor_booking_no','vendor_trace_id','number_of_seats',
               'number_of_concessions','created_date','issued_date','created_by','created_by_hash']
    rows = []
    for i in range(1, 101):
        bid = random.choice(bank_ids)
        status = random.choice(['payment','payment','payment','expired','initial'])
        num_seats = random.randint(1,4)
        num_con = random.randint(0,3)
        seat_price = num_seats * random.choice([75000,85000,95000,120000,150000])
        con_price = num_con * random.choice([59000,89000,139000]) if num_con > 0 else 0
        total = seat_price + con_price
        promo = random.choice([0,20000,50000,100000]) if status == 'payment' else 0
        final = total - promo
        profit_seat = int(seat_price * 0.05)
        profit_con = int(con_price * 0.03) if con_price > 0 else 0
        
        cd = random_date(90,0)
        rows.append([i,f'PAY{i:08d}',status,'issued' if status=='payment' else 'initial','mobile',
                     cd,bid,f'user{i:03d}@bank.com',hash_value(f'user{i:03d}'),f'CH{bid:02d}',
                     seat_price,seat_price,0,0,seat_price,con_price,con_price,seat_price+con_price,
                     seat_price,con_price,total,final,profit_seat,profit_con,int(seat_price*0.02),
                     int(seat_price*0.01),f'PROMO{i:05d}' if promo>0 else '',
                     random.choice(['TET2025','FLASH_FRI','']) if promo>0 else '',promo,int(promo*0.7),
                     int(promo*0.3),0,f'REC{i:05d}' if promo>0 else '',f'BILL{i:08d}',final,'970436',
                     '00','{}',f'BOOK{i:06d}',f'TRACE{i:08d}',num_seats,num_con,cd,cd,'system',hash_value('system')])
    write_csv('lakehouse.lh_vnfilm_v2.orders.csv', headers, rows)
    return list(range(1, 101))

def gen_order_film(order_ids, cinema_ids, film_ids, location_ids, vendor_ids, session_ids):
    headers = ['id','cinema_id','cinema_name_vi','cinema_name_en','cinema_address','film_id','film_name_vi',
               'film_name_en','film_trailer_url','film_poster_url','film_banner_url','film_country',
               'film_category','director','actors','film_duration','film_language','film_description_vi',
               'film_description_en','film_publish_date','film_pg_rating','film_age','location_id',
               'session_id','session_type','session_dimension','session_time','session_version',
               'session_version_type','vendor_id','vendor_name','vendor_name_en','vendor_cinema_id',
               'vendor_film_id','vendor_film_name_vi','vendor_film_name_en','vendor_session_id',
               'vendor_room_id','vendor_room_name_vi','vendor_room_name_en','created_date','modified_date',
               'modified_by','language','created_by','created_by_hash']
    
    cinema_names = {1:('CGV Vincom Bà Triệu','CGV Vincom Ba Trieu','191 Bà Triệu'),2:('CGV Aeon Long Biên','CGV Aeon','27 Cổ Linh'),
                    3:('Lotte Landmark','Lotte Landmark','772 ĐBP'),4:('Galaxy Nguyễn Du','Galaxy ND','116 Nguyễn Du'),
                    5:('BHD Vincom ĐN','BHD DN','910A NQ'),6:('Beta GP','Beta GP','384 GP'),
                    7:('CGV Crescent','CGV Crescent','101 TDT'),8:('Cinestar QT','Cinestar QT','271 NT')}
    film_names = {1:('Quật Mộ Trùng Ma','The Exorcism'),2:('Nhím Sonic 3','Sonic 3'),
                  3:('Mufasa','Mufasa'),4:('Kraven','Kraven'),5:('Linh Miêu','Linh Mieu'),6:('Cô Dâu Hào Môn','Grand Wedding')}
    vendor_names = {1:'CGV',2:'Lotte',3:'Galaxy',4:'BHD',5:'Cinestar',6:'Beta'}
    
    rows = []
    for oid in order_ids:
        cid = random.choice(cinema_ids)
        fid = random.choice(film_ids)
        lid = random.choice(location_ids)
        vid = random.choice(vendor_ids)
        sid = random.choice(session_ids)
        cn = cinema_names.get(cid,('Cinema','Cinema','Address'))
        fn = film_names.get(fid,('Film','Film'))
        vn = vendor_names.get(vid,'Vendor')
        stime = (datetime.now() + timedelta(days=random.randint(-7,7))).strftime('%Y-%m-%d %H:%M:%S.000000')
        
        rows.append([oid,cid,cn[0],cn[1],cn[2],fid,fn[0],fn[1],f'https://yt.com/{fid}',f'https://img/{fid}.jpg',
                     f'https://img/b{fid}.jpg','Mỹ','Hành động','Director','Actors',120,'VI',fn[0],fn[1],
                     random_date(60,0),'C18',18,lid,sid,'nowShowing','Film2D',stime,'subtitling','VER_2D_SUB',
                     vid,vn,vn,f'VND-C{cid:03d}',f'VND-F{fid:03d}',fn[0],fn[1],f'VND-S{sid:05d}','R01',
                     'Phòng 1','Room 1',random_date(30,0),random_date(7,0),'system','vi','system',hash_value('system')])
    write_csv('lakehouse.lh_vnfilm_v2.order_film.csv', headers, rows)

def gen_order_seat(order_ids):
    headers = ['id','name','order_id','vendor_seat_id','vendor_seat_type','vendor_seat_name',
               'vendor_additional_data','vendor_seat_price','vendor_discount_price','vendor_sub_fee',
               'vendor_charge_amount','vendor_reconciliation_price','vendor_fee','vnpay_seat_type',
               'vnpay_seat_price','vnpay_final_price','vnpay_profit','vnpay_fee','campaign_id',
               'sub_campaign_id','campaign_code','campaign_type','campaign_vendor_discount',
               'campaign_vnpay_discount','promotion_vendor_amount','promotion_vnpay_amount',
               'created_date','modified_date','modified_by','created_by','created_by_hash']
    rows = []
    seat_id = 1
    for oid in order_ids:
        num_seats = random.randint(1,4)
        for s in range(num_seats):
            row_letter = chr(ord('A') + random.randint(0,9))
            seat_num = random.randint(1,15)
            seat_type = random.choice(['normal','vip','couple'])
            price = {'normal':85000,'vip':120000,'couple':200000}[seat_type]
            rows.append([seat_id,f'{row_letter}{seat_num}',oid,f'SEAT{seat_id:06d}',seat_type.upper(),
                         f'{row_letter}{seat_num}','{}',price,0,0,price,price,int(price*0.02),seat_type,
                         price,price,int(price*0.05),int(price*0.01),None,None,'','normal',0,0,0,0,
                         random_date(30,0),random_date(7,0),'system','system',hash_value('system')])
            seat_id += 1
    write_csv('lakehouse.lh_vnfilm_v2.order_seat.csv', headers, rows)

def gen_order_concession(order_ids):
    headers = ['id','name','order_id','quantity','vendor_concession_id','vendor_concession_type',
               'vendor_concession_name','vendor_concession_description','vendor_concession_image',
               'vendor_additional_data','vendor_concession_price','vendor_discount_price','vendor_sub_fee',
               'vendor_charge_amount','vendor_reconciliation_price','vnpay_concession_price',
               'vnpay_discount_price','vnpay_final_price','vnpay_profit','campaign_id','sub_campaign_id',
               'campaign_code','campaign_type','campaign_vendor_discount','campaign_vnpay_discount',
               'promotion_vendor_amount','promotion_vnpay_amount','created_date','modified_date',
               'modified_by','created_by','created_by_hash']
    items = [('Combo Couple',89000),('Combo Solo',59000),('Bắp Lớn',45000),('Coca Lớn',35000)]
    rows = []
    con_id = 1
    for oid in order_ids[:60]:
        num_con = random.randint(1,2)
        for _ in range(num_con):
            item = random.choice(items)
            qty = random.randint(1,2)
            rows.append([con_id,item[0],oid,qty,f'CON{con_id:05d}','COMBO',item[0],item[0],
                         f'https://img/con{con_id}.jpg','{}',item[1],0,0,item[1],item[1],item[1],0,
                         item[1],int(item[1]*0.03),None,None,'','normal',0,0,0,0,random_date(30,0),
                         random_date(7,0),'system','system',hash_value('system')])
            con_id += 1
    write_csv('lakehouse.lh_vnfilm_v2.order_concession.csv', headers, rows)

def gen_order_refund(order_ids):
    headers = ['id','refund_code','order_id','pay_code','original_amount','payment_amount','billing_amount',
               'refund_request_amount','refund_vnpay_sub_fee','refund_vendor_sub_fee','refund_bank_sub_fee',
               'refund_amount','status','created_date','created_by','created_by_hash','modified_date',
               'modified_by','rejected_date','rejected_by','approved_date','approved_by',
               'cancel_request_date','is_reportable']
    rows = []
    refund_orders = random.sample(order_ids, 15)
    for i,oid in enumerate(refund_orders, 1):
        amount = random.choice([85000,170000,255000])
        fee = int(amount * 0.1)
        status = random.choice(['approved','approved','rejected','initial'])
        rows.append([i,f'REF{i:06d}',oid,f'PAY{oid:08d}',amount,amount,amount,amount,
                     int(fee*0.5),int(fee*0.3),int(fee*0.2),amount-fee,status,random_date(30,0),
                     'system',hash_value('system'),random_date(7,0),'system',
                     random_date(7,0) if status=='rejected' else None,'admin' if status=='rejected' else None,
                     random_date(7,0) if status=='approved' else None,'admin' if status=='approved' else None,
                     random_date(30,0),True])
    write_csv('lakehouse.lh_vnfilm_v2.order_refund.csv', headers, rows)
    return [r[2] for r in rows]

def gen_notify_email(order_ids, refund_order_ids):
    headers = ['id','order_id','order_refund_id','original_id','type','subject','content','status',
               'server_response_data','created_date','modified_date','modified_by','created_by',
               'created_by_hash','receiver','receiver_hash','sender','sender_hash']
    rows = []
    for i,oid in enumerate(order_ids[:50], 1):
        rows.append([i,oid,None,None,'CONFIRMATION',f'Xác nhận đơn hàng #{oid}','Nội dung email',
                     'SUCCESS','{"status":"sent"}',random_date(30,0),random_date(7,0),'system','system',
                     hash_value('system'),f'user{oid}@email.com',hash_value(f'user{oid}'),
                     'noreply@vnpay.vn',hash_value('noreply')])
    write_csv('lakehouse.lh_vnfilm_v2.notify_email.csv', headers, rows)

def gen_notify_ott(order_ids):
    headers = ['id','order_id','order_refund_id','original_id','type','content','status',
               'server_response_data','created_date','modified_date','modified_by','bank_identity',
               'bank_identity_hash','created_by','created_by_hash']
    rows = []
    for i,oid in enumerate(order_ids[:50], 1):
        rows.append([i,oid,None,None,'CONFIRMATION','Đặt vé thành công','SUCCESS','{"status":"sent"}',
                     random_date(30,0),random_date(7,0),'system',f'user{oid}',hash_value(f'user{oid}'),
                     'system',hash_value('system')])
    write_csv('lakehouse.lh_vnfilm_v2.notify_ott.csv', headers, rows)

def gen_notify_sms(order_ids):
    headers = ['id','order_id','order_refund_id','original_id','type','content','status',
               'server_response_data','created_date','modified_date','modified_by','created_by',
               'created_by_hash','receiver','receiver_hash']
    rows = []
    for i,oid in enumerate(order_ids[:30], 1):
        rows.append([i,oid,None,None,'CONFIRMATION','Đặt vé thành công','SUCCESS','{"status":"sent"}',
                     random_date(30,0),random_date(7,0),'system','system',hash_value('system'),
                     f'0901234{oid:03d}',hash_value(f'0901234{oid:03d}')])
    write_csv('lakehouse.lh_vnfilm_v2.notify_sms.csv', headers, rows)

def gen_customer_tracking(order_ids, bank_ids):
    headers = ['id','order_id','bank_id','device_type','device_name','emei','latitude','longitude',
               'os_version','ip_address','created_date','sdk_version','bank_identity','bank_identity_hash']
    rows = []
    for i,oid in enumerate(order_ids[:40], 1):
        dtype = random.choice(['ANDROID','IOS','WEB'])
        rows.append([i,oid,random.choice(bank_ids),dtype,f'{dtype} Device','EMEI123456',
                     '21.0285','105.8542',f'{random.randint(10,17)}.0',f'192.168.1.{random.randint(1,255)}',
                     random_date(30,0),f'v{random.randint(1,5)}.{random.randint(0,9)}',
                     f'user{oid}',hash_value(f'user{oid}')])
    write_csv('lakehouse.lh_vnfilm_v2.customer_tracking.csv', headers, rows)

# ========== PRE-ORDER TABLES ==========

def gen_pre_order(bank_ids, cinema_ids, film_ids, session_ids, vendor_ids, location_ids, campaign_ids):
    headers = ['id','api_version','bank_id','bank_name','bank_identity','bank_identity_hash','bank_channel_id',
               'promotion_id','promotion_code','promotion_amount','promotion_vnpay_amount','promotion_vendor_amount',
               'promotion_bank_amount','promotion_hold_date','promotion_hold_response','campaign_id',
               'sub_campaign_id','campaign_code','campaign_type','cinema_id','cinema_name_vi','cinema_name_en',
               'cinema_address','location_id','film_id','film_name_vi','film_name_en','session_id','session_type',
               'session_dimension','session_time','session_version','session_version_type','vendor_id','vendor_name',
               'vendor_cinema_id','vendor_film_id','vendor_film_name_vi','vendor_film_name_en','vendor_session_id',
               'vendor_room_id','vendor_room_name_vi','vendor_room_name_en','created_date','created_by','created_by_hash']
    rows = []
    for i in range(1, 51):
        bid = random.choice(bank_ids)
        cid = random.choice(cinema_ids)
        fid = random.choice(film_ids)
        sid = random.choice(session_ids)
        vid = random.choice(vendor_ids)
        lid = random.choice(location_ids)
        rows.append([i,'v4',bid,'VNPAY',f'user{i}',hash_value(f'user{i}'),f'CH{bid:02d}',f'PROMO{i:05d}',
                     'TET2025',50000,35000,15000,0,random_date(30,0),'OK',1,1,'TET2025','normal',
                     cid,'Cinema','Cinema EN','Address',lid,fid,'Film VI','Film EN',sid,'nowShowing',
                     'Film2D',random_date(7,0),'subtitling','VER_2D_SUB',vid,'Vendor',f'VND-C{cid:03d}',
                     f'VND-F{fid:03d}','Film VI','Film EN',f'VND-S{sid:05d}','R01','Phòng 1','Room 1',
                     random_date(30,0),'system',hash_value('system')])
    write_csv('lakehouse.lh_vnfilm_v2.pre_order.csv', headers, rows)
    return list(range(1, 51))

def gen_pre_order_seat(pre_order_ids):
    headers = ['id','pre_order_id','vendor_seat_id','vendor_seat_type','vendor_seat_name','vendor_additional_data',
               'vendor_seat_price','vendor_discount_price','vnpay_seat_type','vnpay_seat_price','campaign_id',
               'sub_campaign_id','campaign_code','campaign_type','campaign_vendor_discount','campaign_vnpay_discount',
               'promotion_vendor_discount','promotion_vnpay_discount','promotion_vendor_amount','promotion_vnpay_amount',
               'created_date','created_by','created_by_hash']
    rows = []
    seat_id = 1
    for poid in pre_order_ids:
        for s in range(random.randint(1,3)):
            row = chr(ord('A') + random.randint(0,9))
            num = random.randint(1,15)
            stype = random.choice(['NORMAL','VIP'])
            price = 85000 if stype=='NORMAL' else 120000
            rows.append([seat_id,poid,f'SEAT{seat_id:06d}',stype,f'{row}{num}','{}',price,0,stype.lower(),
                         price,None,None,'','normal',0,0,0,0,0,0,random_date(30,0),'system',hash_value('system')])
            seat_id += 1
    write_csv('lakehouse.lh_vnfilm_v2.pre_order_seat.csv', headers, rows)

def gen_pre_order_concession(pre_order_ids):
    headers = ['id','pre_order_id','quantity','vendor_concession_id','vendor_concession_type',
               'vendor_concession_name','vendor_concession_description','vendor_concession_image',
               'vendor_additional_data','vendor_concession_price','vendor_discount_price',
               'vnpay_concession_price','campaign_id','sub_campaign_id','campaign_code','campaign_type',
               'campaign_vendor_discount','campaign_vnpay_discount','promotion_vendor_discount',
               'promotion_vnpay_discount','promotion_vendor_amount','promotion_vnpay_amount',
               'created_date','created_by','created_by_hash']
    rows = []
    con_id = 1
    for poid in pre_order_ids[:30]:
        rows.append([con_id,poid,random.randint(1,2),f'CON{con_id:05d}','COMBO','Combo Couple',
                     'Bắp + Nước',f'https://img/con{con_id}.jpg','{}',89000,0,89000,None,None,'',
                     'normal',0,0,0,0,0,0,random_date(30,0),'system',hash_value('system')])
        con_id += 1
    write_csv('lakehouse.lh_vnfilm_v2.pre_order_concession.csv', headers, rows)

def gen_pre_order_customer(pre_order_ids):
    headers = ['id','identity_name','identity_email','identity_code','identity_code_hash','tax_code',
               'tax_name','tax_address','tax_email','created_date','created_by','created_by_hash']
    rows = []
    for poid in pre_order_ids:
        has_tax = random.choice([True, False])
        rows.append([poid,f'Nguyen Van {poid}',f'user{poid}@email.com',f'USER{poid:05d}',
                     hash_value(f'USER{poid:05d}'),f'01234567{poid:02d}' if has_tax else '',
                     f'Cty {poid}' if has_tax else '','HN' if has_tax else '',
                     f'tax{poid}@email.com' if has_tax else '',random_date(30,0),'system',hash_value('system')])
    write_csv('lakehouse.lh_vnfilm_v2.pre_order_customer.csv', headers, rows)

def gen_pre_order_map(bank_ids):
    headers = ['id','seats','concessions','bank_id','bank_identity','campaign_id','sub_campaign_id',
               'campaign_code','created_date','created_by','modified_date','modified_by']
    rows = []
    for i in range(1, 21):
        rows.append([i,f'[{i*100+1},{i*100+2}]',f'{{"1":1,"2":2}}',random.choice(bank_ids),
                     f'user{i}',None,None,'',random_date(7,0),hash_value(f'session{i}'),
                     random_date(1,0),hash_value(f'session{i}')])
    write_csv('lakehouse.lh_vnfilm_v2.pre_order_map.csv', headers, rows)

def gen_pre_order_concession_map(bank_ids):
    headers = ['id','id_value','bank_id','vendor_id','vendor_name','vendor_description','vendor_image',
               'vendor_data','vendor_price','vnpay_price','campaign_id','sub_campaign_id','campaign_code',
               'campaign_type','campaign_vendor_discount','bank_identity','bank_identity_hash',
               'created_date','created_by','created_by_hash']
    rows = []
    for i in range(1, 21):
        bid = random.choice(bank_ids)
        rows.append([f'MAP{i:05d}',i,bid,f'CON{i:03d}','Combo','Mô tả',f'https://img/{i}.jpg','{}',
                     89000,89000,None,None,'','normal',0,f'user{i}',hash_value(f'user{i}'),
                     random_date(7,0),'system',hash_value('system')])
    write_csv('lakehouse.lh_vnfilm_v2.pre_order_concession_map.csv', headers, rows)

def gen_cdp_camp_conversion_stage(campaign_ids):
    headers = ['campaign_id','created_at','num_cus_stage_1','num_cus_stage_2','num_cus_stage_3',
               'num_cus_stage_4','num_cus_stage_5','conversion_rate_stage_1','conversion_rate_stage_2',
               'conversion_rate_stage_3','conversion_rate_stage_4','conversion_rate_stage_5']
    rows = []
    for cid in campaign_ids:
        for _ in range(3):
            s1 = random.randint(5000,10000)
            s2 = int(s1 * random.uniform(0.6,0.8))
            s3 = int(s2 * random.uniform(0.5,0.7))
            s4 = int(s3 * random.uniform(0.4,0.6))
            s5 = int(s4 * random.uniform(0.3,0.5))
            rows.append([cid,random_date(30,0),s1,s2,s3,s4,s5,1.0,s2/s1,s3/s2,s4/s3,s5/s4])
    write_csv('lakehouse.cdp_mart.cdp_camp_conversion_stage.csv', headers, rows)

# ========== MAIN ==========

def main():
    print("Generating mock data for VNFILM Ticketing domain...")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Phase 1: Dimension tables
    bank_ids = gen_bank()
    vendor_ids = gen_vendor()
    location_ids = gen_locations()
    cinema_ids = gen_cinema(vendor_ids, location_ids)
    film_ids = gen_film(vendor_ids)
    session_ids = gen_sessions(cinema_ids, film_ids, location_ids, vendor_ids)
    concession_ids = gen_concession(vendor_ids, cinema_ids)
    gen_bank_seat_definition(bank_ids, vendor_ids)
    campaign_ids = gen_dim_campaign(bank_ids)
    
    # Phase 2: Bridge tables
    gen_vendor_bank(vendor_ids, bank_ids)
    gen_cinema_favourite(bank_ids, cinema_ids, vendor_ids)
    
    # Phase 3: Orders
    order_ids = gen_orders(bank_ids)
    gen_order_film(order_ids, cinema_ids, film_ids, location_ids, vendor_ids, session_ids)
    gen_order_seat(order_ids)
    gen_order_concession(order_ids)
    refund_order_ids = gen_order_refund(order_ids)
    
    # Phase 4: Pre-orders
    pre_order_ids = gen_pre_order(bank_ids, cinema_ids, film_ids, session_ids, vendor_ids, location_ids, campaign_ids)
    gen_pre_order_seat(pre_order_ids)
    gen_pre_order_concession(pre_order_ids)
    gen_pre_order_customer(pre_order_ids)
    gen_pre_order_map(bank_ids)
    gen_pre_order_concession_map(bank_ids)
    
    # Phase 5: Notifications
    gen_notify_email(order_ids, refund_order_ids)
    gen_notify_ott(order_ids)
    gen_notify_sms(order_ids)
    gen_customer_tracking(order_ids, bank_ids)
    
    # Phase 6: CDP
    gen_cdp_camp_conversion_stage(campaign_ids)
    
    print("\n✅ All mock data files generated successfully!")

if __name__ == "__main__":
    main()
