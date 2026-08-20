# สิ่งที่ต้องถาม Professor — ถามอะไร เพื่ออะไร ไปเติมตรงไหน

2026-08-20. เรียงตามความสำคัญ ทุกข้อ map กลับไปยังจุด `\todo` สีแดงหรือช่องว่างใน paper

## A. ข้อมูล simulation (เติม Section 1 + 3 — จุดแดง 2 จุดใหญ่สุด)

| # | ถามว่า | เพื่ออะไร | เติมตรงไหน |
|---|---|---|---|
| 1 | sample นี้ generate ด้วยอะไร (generator + framework + version เช่น Gauss/Geant4 vX) | referee ฟิสิกส์ bounce ทันทีถ้าไม่มี provenance | Sec 3 "Dataset", จุดแดง "Simulation provenance" |
| 2 | เงื่อนไข pileup คือเท่าไหร่ — ν (จำนวน interaction เฉลี่ย/crossing) หรือ instantaneous luminosity | ตอนนี้เขียนว่า "tens of simultaneous interactions" ซึ่งลอย | Sec 1 ย่อหน้าแรก, จุดแดงแรกของเล่ม |
| 3 | timestamp ใน sample ถูก digitize/smear ที่ resolution เท่าไหร่ (spec 15 ps จริงไหม, มี out-of-time spillover ไหม) | ผล timing 20%/38% ของเราตีความไม่ได้เต็มที่ถ้าไม่รู้ resolution ที่จำลอง | Sec 3 + Sec 6 Timing, จุดแดง "timestamp resolution" |
| 4 | photon selection ของ sample คืออะไร (conversion veto? isolation? matching criteria?) | นิยาม population ที่ตัวเลขทุกตัวอ้างถึง + อธิบาย catastrophic tails (wrong-photon match) | Sec 3 + Sec Anatomy (tails) |

## B. เส้นทางตีพิมพ์ (ตัดสินรูปแบบเล่มทั้งเล่ม)

| # | ถามว่า | เพื่ออะไร | เติมตรงไหน |
|---|---|---|---|
| 5 | งานที่ใช้ LHCb simulation แบบนี้ ต้องผ่าน Editorial Board ไหม หรือออกเป็น LHCb-PUB note ก่อน แล้วค่อย arXiv/journal ได้หรือไม่ | ตัดสินว่าเล่มนี้จะเป็น note / JINST / proceedings — และเราจะได้ format ที่ถูกต้อง | Acknowledgements จุดแดง "publication-approval status" + เปลี่ยน template |
| 6 | อาจารย์ทั้งสองอยากเป็น co-author หรือให้ acknowledge (ชื่อเต็ม+affiliation ที่ต้องการ) | ใส่ชื่อโดยไม่ถามก่อนผิดมารยาท | Author list + Acknowledgements จุดแดง |
| 7 | repo GitHub push public ได้เลยไหม (โค้ด+CSV ไม่มี raw data ของ collaboration) | ลิงก์ใน Code availability ต้อง "จริง" — และเป็น M6 ของ proposal | Code availability (ลิงก์แปะแล้ว รอไฟเขียว push) |

## C. Trigger/deployment (ปิด M5 ของ proposal)

| # | ถามว่า | เพื่ออะไร | เติมตรงไหน |
|---|---|---|---|
| 8 | per-event time budget ของ Allen (HLT1) สำหรับ ECAL reconstruction คือเท่าไหร่ | เรามีเลขแล้ว: GPU 24 µs/cluster (batch 1024, H100) — ขาดแค่เส้น budget มาเทียบ จะวาด "latency vs HLT budget" จบ M5 ได้ | Sec Computational cost + รูป throughput (เพิ่มเส้น budget) |
| 9 | ตอน Upgrade II จริง คาดว่า photon cluster ต่อ event ~กี่ก้อน | แปลง µs/cluster → µs/event ให้เทียบ budget ได้ตรงหน่วย | ที่เดียวกับข้อ 8 |

## D. ก้าวต่อไปของวิทยาศาสตร์ (คำขอหลักของเล่ม)

| # | ถามว่า | เพื่ออะไร | เติมตรงไหน |
|---|---|---|---|
| 10 | ขอ minimum-bias simulation เพิ่ม ~3 เท่า ทำได้ไหม ใช้เวลา/คิวเท่าไหร่ | scaling curve วัดแล้ว σ∝N^−0.18 → 3× data ≈ 0.033 — นี่คือ action item เดียวที่พาต่ำกว่า 0.035 | Sec Scaling — เปลี่ยน extrapolation เป็นการวัดจริงเมื่อได้ข้อมูล |
| 11 | ขอ per-event truth เพิ่มใน tuple: จำนวน cluster ซ้อน / photon-match quality flag | พิสูจน์สมมติฐานเรื่อง catastrophic tails (5–10% ที่พลาด 40–60%) ว่าเป็น wrong-match/overlap จริง | Sec Anatomy (2) — เปลี่ยน "likely origin" เป็น measured |
| 12 | containment เลข 41% ของเราขัดกับ 27% ใน internal note เก่า — นิยามที่ collaboration ใช้วัด containment คืออะไร | กัน reviewer ภายในทักเรื่องเลขไม่ตรงระหว่างเอกสาร | Fig containment caption (ระบุนิยามให้ตรงกัน) |

## เคล็ดการถาม

- ข้อ 5–6 ถามตอนต้นมีตติ้ง (เป็น blocker การ push/submit ทุกอย่าง)
- ข้อ 10 ถามตอนจบ หลังโชว์ scaling curve — ให้กราฟขายคำขอเอง
- ข้อ 1–3 ฝากเป็น email follow-up ได้ถ้าเวลาไม่พอ (เป็น fact lookup ไม่ใช่ discussion)
