# Guided notebooks

- `01_input_data_from_mentor_th.ipynb` — บทเรียนปัจจุบัน ใช้ notebook ของ mentor
  เป็น guideline แล้วเขียนใหม่เป็นภาษาไทย: ROOT hierarchy, scalar/jagged branches,
  บทบาทของ input/truth/reconstruction และความหมายของ repeated event IDs

Notebook ที่ AI สร้างไว้ก่อนหน้าถูกนำออกจาก working tree แล้ว และยังเก็บได้จาก
Git history/archive branch หากต้อง audit ภายหลัง

## กติกา

- เรียนครั้งละหนึ่ง notebook และตอบ checkpoint ก่อนสร้างบทถัดไป
- Code cell ต้องรันตามลำดับจาก kernel ใหม่
- ไม่ใช้ machine-specific paths
- ข้อสรุปทุกข้อยืนยันจาก data, mentor หรือ primary source
- Reusable code ย้ายเข้า `src/picocal/` หลังเข้าใจและมี test แล้วเท่านั้น

## จุดเริ่มต้นของครั้งถัดไป (22 มิถุนายน 2026)

เริ่มบทเรียน **Data Exploration** จาก
`01_input_data_from_mentor_th.ipynb` โดยใช้ภาษาไทยและเดินทีละขั้น:

1. ทบทวนว่า 1 file, 1 TTree, 1 entry, 1 event, 1 cluster และ 1 cell ต่างกันอย่างไร
2. เปิด ROOT file และตรวจชื่อ branch, shape, dtype และหน่วยจากข้อมูลจริง
3. แยก input measurement, reconstruction fields และ truth target ให้ชัดเจน
4. ตรวจพลังงานระดับ cell และผลรวมระดับ cluster พร้อม visualization
5. อธิบายทุกบรรทัดของ Python/`uproot` ก่อนรัน และมี checkpoint สั้น ๆ ทุกช่วง

ยังไม่กำหนด training target, data split หรือเริ่ม train model จนกว่าจะได้รับคำตอบจาก
mentor เรื่อง shared reconstructed clusters และหน่วยของ `sig_flux_eTot`
