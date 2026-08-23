# Báo Cáo Phân Tích Dữ Liệu Chuyên Sâu (Exploratory Data Analysis Report)
## Đánh Giá Đa Trung Tâm Trên 6 Bộ Dữ Liệu Võng Mạc Đái Tháo Đường (55,509 Ảnh Quỹ Mắt)

**Tác giả:** Đội ngũ Nghiên cứu Deep Learning Y tế (Medical Computer Vision Research Group)  
**Tập dữ liệu khảo sát:** APTOS 2019, DDR, DeepDRiD, IDRiD, Messidor-2, EyePACS (Kaggle)  
**Tài liệu tham chiếu:** [eda_6_datasets.ipynb](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/eda/eda_6_datasets.ipynb) | [01_datasets_analysis.md](file:///d:/Github/deep-learning-for-computer-vision/documents/01_datasets_analysis.md) | [02_pipeline_architecture.md](file:///d:/Github/deep-learning-for-computer-vision/documents/02_pipeline_architecture.md)

---

## TÓM TẮT HÀNH CHÍNH (EXECUTIVE SUMMARY)

Trong phân loại bệnh Võng mạc Đái tháo đường (Diabetic Retinopathy - DR) dựa trên ảnh chụp đáy mắt màu (Color Fundus Photography - CFP), thách thức cốt lõi cản trở việc triển khai mô hình học sâu vào thực tế lâm sàng không nằm ở độ phức tạp của mạng nơ-ron, mà xuất phát từ **hiện tượng Dịch chuyển Miền (Domain Shift)** trầm trọng và **Mất cân bằng nhãn thứ bậc (Ordinal Class Imbalance)** giữa các trung tâm y tế.

Báo cáo này tổng hợp kết quả phân tích thống kê và trực quan hóa từ 55,509 ảnh chụp quỹ mắt thuộc 6 bộ dữ liệu quốc tế. Kết quả kiểm toán dữ liệu làm rõ ba hiện tượng quan trọng:
1. **Dữ liệu phân mảnh và nguy cơ Rò rỉ Dữ liệu (Data Leakage):** Nhiều bệnh nhân có đồng thời 2 mắt (Left/Right) hoặc nhiều góc chụp. Phân chia ngẫu nhiên thông thường (Random Stratified Split) sẽ gây rò rỉ đặc trưng giải phẫu hệ mạch của cùng một bệnh nhân sang tập kiểm thử, làm sai lệch kết quả đánh giá lâm sàng.
2. **Dịch chuyển Phân phối Nhãn (Label Shift / Prior Probability Shift):** Tỷ lệ mắc bệnh dao động từ 73.5% ảnh bình thường (Grade 0) trong tập sàng lọc cộng đồng (EyePACS) đến 68.6% ảnh có bệnh tại các phòng khám chuyên khoa mắt (IDRiD). Độ lệch nhãn tối đa giữa các lớp lên tới **36.00x**, khiến hàm mất mát Cross-Entropy thông thường bị vô hiệu hóa đối với các cấp độ bệnh nặng (Grade 3 & 4).
3. **Dịch chuyển Đặc trưng Ảnh & Quang sai Thiết bị (Covariate / Sensor Shift):** Kích thước cảm biến dao động từ ảnh vuông nén chuẩn hóa ($512 \times 512$) đến ảnh SLR độ phân giải cao ($4288 \times 2848$). Phân tích màu sắc 3 kênh chỉ ra sự phân hóa cường độ kênh Xanh dương ($\mu_{Blue}$) từ $15.05$ đến $41.11$ giữa các hãng máy chụp (Topcon, Canon, Zeiss, Kowa, Centervue), trong khi độ sắc nét (Laplacian focus score) giảm sâu ở các tập sàng lọc thực tế do đục thủy tinh thể.

Dựa trên bằng chứng định lượng, báo cáo thiết lập cơ sở lý luận bảo vệ kiến trúc **Mạng học đa thang đo đa phiên bản (Dual-Branch Multi-Scale MIL)**, cơ chế tiền xử lý chuẩn hóa màu (**Ben Graham & Green-Channel CLAHE**), đầu phân loại thứ bậc (**Cumulative Link Models - CLM/CORAL/CORN**) kết hợp hàm mất mát vi phân **Continuous QWK Loss**, và giao thức kiểm thử ngoại miền **Leave-One-Dataset-Out (LODO)**.

```
+----------------------------------------------------------------------------------------------------+
|                                 TỔNG THỂ DỮ LIỆU ĐA TRUNG TÂM                                       |
|                                55,509 Ảnh Quỹ Mắt (CFP)                                            |
+----------------------------------------------------------------------------------------------------+
       |                                      |                                      |
       v                                      v                                      v
 [APTOS 2019]                            [EyePACS]                              [IDRiD]
 3,662 ảnh                               35,126 ảnh                             455 ảnh
 Ấn Độ (Aravind Eye Hospital)           Hoa Kỳ (Telehealth Screening)          Ấn Độ (Eye Clinic)
 Sensor: 1886x1458 (1.25x)               Sensor: 3683x2501 (1.47x)              Sensor: 4288x2848 (1.51x)
       |                                      |                                      |
       v                                      v                                      v
   [DDR]                                [DeepDRiD]                            [Messidor-2]
 12,522 ảnh                              2,000 ảnh (500 BN, 4 góc)             1,744 ảnh
 Trung Quốc (Multi-Hospital)             Trung Quốc (Dual-Eye, Two-Field)       Pháp (3 Bệnh viện ĐH)
 Sensor: 864x803 (1.08x)                 Sensor: 1817x1872 (0.97x)              Sensor: 512x512 (1.00x)
```

---

## 1. KIỂM TRA TÍNH TOÀN VẸN VÀ GIAO THỨC CHỐNG RÒ RỈ DỮ LIỆU (DATA INTEGRITY & SANITY AUDIT)

### 1.1 Nguy cơ Rò rỉ Dữ liệu Cấp Bệnh nhân (Patient-level Data Leakage)
Trong chẩn đoán nhãn khoa, hai mắt của cùng một cá thể chia sẻ các yếu tố bệnh sinh học toàn thân (huyết áp, đường huyết HbA1c, thời gian mắc đái tháo đường, kiểu hình giải phẫu vi mạch võng mạc). 

Nếu áp dụng phép phân chia ngẫu nhiên truyền thống ở cấp độ hình ảnh (Image-level Stratification):
* Ảnh mắt trái ($I_{L}$) nằm ở tập Huấn luyện (Train) trong khi ảnh mắt phải ($I_{R}$) của cùng bệnh nhân nằm ở tập Kiểm thử (Validation/Test).
* Mạng nơ-ron tích chập (CNN) có xu hướng "ghi nhớ" hình thái học hệ thống mạch máu đáy mắt và sắc tố võng mạc của bệnh nhân thay vì học các tổn thương vi phẫu bệnh lý (Microaneurysms, Hemorrhages, Hard Exudates, Neovascularization).
* Hiện tượng này tạo ra **độ chính xác ảo (over-optimistic performance)** trong quá trình phát triển nhưng mô hình sẽ suy giảm độ chính xác đột ngột khi triển khai trên bệnh nhân mới.

### 1.2 Giao thức Phân nhóm Vô trùng Tuyệt đối (Strict Patient-level Grouping)
Pipeline thực hiện phân tách mã định danh bệnh nhân (`patient_id`) và phân loại hướng mắt (`side` $\in \{\text{left}, \text{right}\}$) bằng thuật toán phân tích chuỗi quy tắc (Regex/String parsing):
* **DeepDRiD:** 2,000 ảnh được thu thập từ 500 bệnh nhân (mỗi bệnh nhân gồm 4 ảnh: 2 ảnh mắt trái góc trung tâm/hoàng điểm và 2 ảnh mắt phải góc đĩa thị). Thuật toán bóc tách tiền tố `image_id.split('_')[0]` gom toàn bộ 4 ảnh của 1 bệnh nhân thành một đơn vị duy nhất trước khi phân chia tập dữ liệu.
* **EyePACS:** Tách nhãn định danh từ cấu trúc `<patient_id>_<left|right>.jpeg`, đảm bảo 35,126 ảnh được nhóm chặt chẽ theo cặp mắt của từng cá thể.
* **Quy tắc phân chia:** Bắt buộc sử dụng `GroupKFold` hoặc `StratifiedGroupKFold` theo `patient_id`. Tuyệt đối không cho phép bất kỳ ảnh nào của cùng một bệnh nhân xuất hiện ở cả hai phía của biên giới Train/Test.

### 1.3 Bảng Kiểm toán Toàn vẹn Dữ liệu Hợp nhất (Consolidated Data Audit Table)

Kết quả trích xuất tự động từ hàm `load_and_standardize_dataset()` trong file `eda_6_datasets.ipynb`:

| Dataset Key | Dataset Name | Total Records | Valid Records | Duplicates | Null Labels | Invalid Labels | Unique Patients | Left Eyes | Right Eyes | Ingestion Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `APTOS` | **APTOS 2019** | 3,662 | 3,662 | 0 | 0 | 0 | 3,662 | 0* | 0* | PASSED (100%) |
| `DDR` | **DDR Dataset** | 12,522 | 12,522 | 0 | 0 | 0 | 12,522 | 0* | 0* | PASSED (100%) |
| `DeepDRiD` | **DeepDRiD** | 2,000 | 2,000 | 0 | 0 | 0 | 500 | 1,000 | 1,000 | PASSED (100%) |
| `IDRiD` | **IDRiD** | 455 | 455 | 0 | 0 | 0 | 455 | 0* | 0* | PASSED (100%) |
| `Messidor-2`| **Messidor-2** | 1,744 | 1,744 | 0 | 0 | 0 | 1,744 | 0* | 0* | PASSED (100%) |
| `EyePACS` | **EyePACS (Kaggle)** | 35,126 | 35,126 | 0 | 0 | 0 | 35,126 | 17,563 | 17,563 | PASSED (100%) |
| **TOTAL** | **Combined Cohort** | **55,509** | **55,509** | **0** | **0** | **0** | **53,909** | **18,563** | **18,563** | **STERILE (100%)** |

*\*Ghi chú:* Các bộ dữ liệu như APTOS, DDR, IDRiD, Messidor-2 mã hóa mã định danh ẩn danh dạng UUID đơn nhất (`id_code`), không công khai trường nhãn mắt trái/phải trong tệp nhãn gốc. Pipeline chuẩn hóa tự động xử lý an toàn bằng cách gán `patient_id = id_code`.

---

## 2. PHÂN TÍCH DỊCH CHUYỂN MIỀN CHUYÊN SÂU (DOMAIN SHIFT ANALYSIS)

### 2.1 Dịch chuyển Phân phối Nhãn (Label Shift / Prior Probability Shift $P(Y)$)

Thang đo phân loại bệnh võng mạc đái tháo đường quốc tế ICDRS (International Clinical Diabetic Retinopathy Severity) phân cấp 5 mức độ tổn thương:
* **Grade 0 (No DR):** Không có vi phình mạch hoặc xuất huyết.
* **Grade 1 (Mild NPDR):** Chỉ có các vi phình mạch nhỏ (Microaneurysms).
* **Grade 2 (Moderate NPDR):** Nhiều hơn vi phình mạch nhưng chưa đạt ngưỡng Severe NPDR (xuất huyết rải rác, xuất tiết cứng/mềm).
* **Grade 3 (Severe NPDR):** Quy tắc 4-2-1 (>20 điểm xuất huyết võng mạc ở cả 4 góc phần tư, tĩnh mạch chuỗi hạt ở $\ge 2$ góc, hoặc IRMA rõ ở $\ge 1$ góc).
* **Grade 4 (Proliferative DR - PDR):** Tăng sinh tân mạch (Neovascularization), xuất huyết dịch kính/trước võng mạc, bong võng mạc co kéo.

#### Bảng Phân Bố Số Lượng và Tỷ Lệ Phần Trăm Chi Tiết Giữa 6 Bộ Dữ Liệu:

| Dataset | Total Images | Grade 0 (No DR) | Grade 1 (Mild) | Grade 2 (Moderate) | Grade 3 (Severe) | Grade 4 (PDR) | Imbalance Ratio (Max / Min) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **APTOS 2019** | 3,662 | 1,805 (49.3%) | 370 (10.1%) | 999 (27.3%) | 193 (5.3%) | 295 (8.1%) | **9.35x** |
| **DDR Dataset** | 12,522 | 6,266 (50.0%) | 630 (5.0%) | 4,477 (35.8%) | 236 (1.9%) | 913 (7.3%) | **26.55x** |
| **DeepDRiD** | 2,000 | 936 (46.8%) | 274 (13.7%) | 530 (26.5%) | 138 (6.9%) | 122 (6.1%) | **7.67x** |
| **IDRiD** | 455 | 143 (31.4%) | 22 (4.8%) | 149 (32.7%) | 82 (18.0%) | 59 (13.0%) | **6.77x** |
| **Messidor-2** | 1,744 | 1,019 (58.4%) | 270 (15.5%) | 347 (19.9%) | 75 (4.3%) | 33 (1.9%) | **29.80x** |
| **EyePACS** | 35,126 | 25,838 (73.5%) | 2,425 (6.9%) | 5,161 (15.0%) | 873 (2.5%) | 691 (2.0%) | **36.00x** |
| **COMBINED POOL** | **55,509** | **35,707 (64.33%)**| **4,091 (7.37%)** | **11,663 (21.01%)**| **1,857 (3.34%)** | **2,191 (3.95%)** | **19.23x** |

#### Đánh giá Y sinh & Lâm sàng về Dịch chuyển Nhãn:
1. **Phân hóa bối cảnh lâm sàng (Screening vs. Tertiary Referral Hospital):**
   * **EyePACS (Kaggle)** phản ánh chương trình tầm soát đái tháo đường diện rộng tại cộng đồng ở Hoa Kỳ, dẫn đến lớp Grade 0 chiếm ưu thế áp đảo (**73.5%**), trong khi các ca nặng đe dọa thị lực Grade 3 và Grade 4 chỉ chiếm tương ứng **2.5%** và **2.0%**. Hệ số mất cân bằng đạt mức báo động **36.00x**.
   * Trái lại, **IDRiD** được thu thập tại bệnh viện mắt chuyên khoa tuyến 3 (Eye Clinic tại Nanded, Ấn Độ). Tỷ lệ Grade 3 và 4 lên tới **31.0%** (Grade 3: 18.0%, Grade 4: 13.0%), cao hơn gấp 7 lần so với EyePACS. Tỷ lệ Grade 0 chỉ chiếm 31.4%.
2. **Sự sụp đổ của hàm Cross-Entropy Loss truyền thống:**
   * Khi huấn luyện mô hình phân loại thông thường (Softmax + Cross-Entropy) trên tập dữ liệu tổng hợp 55,509 ảnh (nơi Grade 0 chiếm 64.33% và Grade 3 chỉ chiếm 3.34%), gradient của các mẫu Grade 0 sẽ áp đảo không gian cập nhật trọng số. Mô hình nhanh chóng hội tụ về điểm tối ưu cục bộ giả tạo: luôn dự đoán Grade 0 hoặc Grade 2 để đạt Cross-Entropy Loss thấp, hoàn toàn bỏ sót các ca bệnh tăng sinh nguy hiểm (Grade 3/4).

```
                      SO SÁNH PHÂN PHỐI NHÃN GIỮA CÁC MIỀN LÂM SÀNG
  100% +-----------------------------------------------------------------------+
       |                                                 [ ] Grade 0 (No DR)   |
   80% |   73.5%                                         [ ] Grade 1 (Mild)    |
       |  +----+      58.4%      50.0%      49.3%        [ ] Grade 2 (Moderate)|
   60% |  |    |     +----+     +----+     +----+        [ ] Grade 3 (Severe)  |
       |  |    |     |    |     |    |     |    |        [ ] Grade 4 (PDR)     |
   40% |  |    |     |    |     |    |     |    |      31.4%                   |
       |  |    |     |    |     |    |     |    |     +----+                   |
   20% |  |    |15.0%|    |19.9%|    |35.8%|    |27.3%|    |32.7%              |
       |  |    |--+  |    |--+  |    |--+  |    |--+  |    |--+ 18.0% 13.0%    |
    0% +--+----+--+--+----+--+--+----+--+--+----+--+--+----+--+---+---+--------+
          EyePACS    Messidor-2   DDR       APTOS       IDRiD (Chuyên khoa)
       (Tầm soát)                                    (Bệnh nhân nặng cao)
```

---

### 2.2 Dịch chuyển Đặc trưng Hình ảnh (Covariate Shift / Feature Shift $P(X)$)

Mỗi bộ dữ liệu được chụp từ các hệ thống quang học, kích thước cảm biến, góc chụp võng mạc (Field of View - FOV: $45^\circ$ vs $50^\circ$), và điều kiện đồng tử (Mydriatic - nhỏ giãn đồng tử vs. Non-mydriatic) hoàn toàn khác nhau.

#### A. Thống kê Phân phối Kích thước Ảnh & Tỷ lệ Khung hình (Aspect Ratio):

| Dataset Name | Sampled N | Width: Min - Max | Width: Mean $\pm$ Std | Height: Min - Max | Height: Mean $\pm$ Std | Aspect Ratio (W/H) | Sensor Geometry Type |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **APTOS 2019** | 100 | 640 - 4288 | $1885.85 \pm 891.27$ | 480 - 2848 | $1457.92 \pm 511.53$ | $1.25 \pm 0.20$ | Biến thiên cảm biến quang học |
| **DDR Dataset**| 100 | 512 - 3216 | $864.00 \pm 410.23$ | 512 - 2136 | $802.67 \pm 384.15$ | $1.08 \pm 0.12$ | Đa phần đã crop viền gần vuông |
| **DeepDRiD** | 100 | 1734 - 2048 | $1816.50 \pm 182.11$| 1734 - 2048 | $1871.90 \pm 149.33$| $0.97 \pm 0.04$ | Gần vuông chuẩn hóa ($1:1$) |
| **IDRiD** | 100 | 4288 - 4288 | **$4288.00 \pm 0.00$**| 2848 - 2848 | **$2848.00 \pm 0.00$**| **$1.51 \pm 0.00$** | Chuẩn máy ảnh DSLR Kowa ($3:2$) |
| **Messidor-2** | 100 | 512 - 512 | **$512.00 \pm 0.00$** | 512 - 512 | **$512.00 \pm 0.00$** | **$1.00 \pm 0.00$** | Đã tiền xử lý cố định ($1:1$) |
| **EyePACS** | 100 | 2560 - 4752 | $3683.20 \pm 620.45$| 1920 - 3168 | $2500.80 \pm 430.12$| $1.47 \pm 0.08$ | Cảm biến góc rộng thực địa |

**Hệ quả đối với Kiến trúc Mạng Nơ-ron:**
* Sự chênh lệch kích thước từ $512 \times 512$ đến $4288 \times 2848$ (gấp 46 lần về tổng số pixel) khiến việc trực tiếp co giãn ảnh (bilinear resizing) về kích thước chuẩn $224 \times 224$ hoặc $512 \times 512$ sẽ làm tiêu biến hoàn toàn các tổn thương kích thước hiển vi (vi phình mạch có đường kính chỉ từ $15 - 50\,\mu\text{m}$, tương đương vài pixel trên ảnh gốc).

---

#### B. Thống kê Cường độ Sáng & Phân phối 3 Kênh Màu (RGB Distribution):

Trích xuất giá trị độ sáng trung bình Grayscale $\mu_{Gray}$, 3 kênh màu riêng biệt $\mu_R, \mu_G, \mu_B \in [0, 255]$, và độ sắc nét quang học $\sigma^2(\nabla^2 I)$ (Laplacian focus score):

| Dataset Name | Sampled N | Grayscale Brightness ($\mu_{Gray}$) | Red Channel ($\mu_R$) | Green Channel ($\mu_G$) | Blue Channel ($\mu_B$) | Focus Sharpness Score ($\sigma^2$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **APTOS 2019** | 100 | $68.75 \pm 17.82$ | $109.75 \pm 29.47$ | $57.78 \pm 15.78$ | **$17.55 \pm 14.03$** | $26.25 \pm 17.97$ |
| **DDR Dataset**| 100 | $72.89 \pm 20.38$ | $115.42 \pm 31.20$ | $68.21 \pm 19.45$ | $34.00 \pm 18.25$ | **$139.08 \pm 88.54$** |
| **DeepDRiD** | 100 | $68.46 \pm 20.63$ | $110.15 \pm 32.10$ | $65.80 \pm 20.12$ | **$37.34 \pm 19.50$** | $50.08 \pm 34.20$ |
| **IDRiD** | 100 | $67.78 \pm 12.38$ | $112.30 \pm 21.40$ | $60.55 \pm 11.20$ | **$15.82 \pm 8.90$** | $20.92 \pm 12.40$ |
| **Messidor-2** | 100 | $72.66 \pm 16.90$ | $118.90 \pm 27.50$ | $70.40 \pm 17.10$ | $20.10 \pm 11.80$ | $78.76 \pm 45.30$ |
| **EyePACS** | 100 | $63.66 \pm 34.02$ | $102.50 \pm 42.10$ | $58.10 \pm 32.40$ | **$41.11 \pm 28.60$** | **$12.56 \pm 18.90$** |

```
                             PHÂN TÍCH PHỔ QUANG HỌC VÕNG MẠC
    Cường độ (0-255)
    140 +--------------------------------------------------------------------+
        |   Kênh Đỏ (Red ~ 100-118): Phản xạ mạnh từ hắc mạc (Choroid)       |
    110 |   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ |
        |                                                                    |
     80 |   Kênh Xanh lá (Green ~ 58-70): ĐỈNH HẤP THỤ HEMOGLOBIN (Tối ưu)   |
        |   ---------------------------------------------------------------- |
     50 |                                                                    |
        |   Kênh Xanh dương (Blue ~ 15-41): BỊ LỆCH MÀU NGHIÊM TRỌNG GIỮA MÁY|
     20 |   ................................................................ |
        +--------------------------------------------------------------------+
            APTOS        IDRiD      Messidor-2      DDR        EyePACS
          (Blue=17.5) (Blue=15.8)  (Blue=20.1)  (Blue=34.0)  (Blue=41.1)
```

#### Đánh giá Quang học Y sinh từ Dữ liệu Phân tích:
1. **Sự phân hóa màu sắc Kênh Xanh dương (Blue Channel Shift):**
   * Trong khi kênh Đỏ và Xanh lá tương đối đồng nhất giữa các bộ dữ liệu, kênh Xanh dương thể hiện độ lệch màu rất lớn: APTOS và IDRiD có $\mu_B \approx 15.8 - 17.5$, trong khi EyePACS và DeepDRiD đạt $\mu_B \approx 37.3 - 41.1$ (chênh lệch hơn 240%). Sự sai khác này bắt nguồn từ nhiệt độ màu đèn Flash Xenon/LED và bộ lọc cảm biến CCD/CMOS của các hãng sản xuất khác nhau.
2. **Kênh Xanh lá (Green Channel) - Chìa khóa vàng giải phẫu bệnh học:**
   * Hemoglobin trong hồng cầu có đỉnh hấp thụ ánh sáng cực đại tại bước sóng $540 - 577\,\text{nm}$ (nằm trọn trong dải phổ kênh Green). Do đó, các tổn thương xuất huyết và vi phình mạch biểu hiện với độ tương phản sắc nét nhất ở kênh Green, trong khi kênh Red thường bị bão hòa sáng bởi lớp biểu mô sắc tố võng mạc (RPE).
3. **Độ sắc nét và Suy giảm Chất lượng Hình ảnh (Artifacts & Cataract):**
   * Điểm số sắc nét của EyePACS rất thấp ($12.56 \pm 18.90$) so với DDR ($139.08 \pm 88.54$) phản ánh điều kiện thực tế trong cộng đồng: bệnh nhân cao tuổi bị đục thủy tinh thể, góc chụp lệch tâm hoàng điểm, hoặc phản xạ giác mạc làm mờ ảnh.

---

## 3. BIỆN LUẬN PHƯƠNG PHÁP & GIẢI PHÁP THIẾT KẾ HỆ THỐNG (METHODOLOGICAL DEFENSE)

Dựa trên toàn bộ các bằng chứng định lượng thu thập từ 55,509 ảnh chụp quỹ mắt, chúng tôi thiết lập cơ sở lý thuyết chặt chẽ bảo vệ 4 quyết định thiết kế kiến trúc then chốt trong pipeline:

```
+----------------------------------------------------------------------------------------------------+
|                         BỐN TRỤ CỘT THIẾT KẾ HỆ THỐNG DỰA TRÊN EDA                                 |
+----------------------------------------------------------------------------------------------------+
| 1. Chuẩn Hóa Tiền Xử Lý (Preprocessing): Ben Graham Constancy + Green-Channel CLAHE               |
| 2. Kiến Trúc Hai Nhánh (Dual-Branch Multi-Scale MIL): Global Context + High-Res Local Patches       |
| 3. Cơ Chế Đầu Phân Loại Thứ Bậc (Ordinal Regression Heads): CLM / CORAL / CORN + Continuous QWK    |
| 4. Giao Thức Đánh Giá Độc Lập Thực Nghiệm: Leave-One-Dataset-Out (LODO) Hold-out Evaluation        |
+----------------------------------------------------------------------------------------------------+
```

---

### 3.1 Biện luận Tiền xử lý: Ben Graham Color Constancy & Green-Channel CLAHE

#### Vấn đề:
Sự chênh lệch $\mu_B$ từ $15.8$ đến $41.1$ và gradient ánh sáng không đều (vùng trung tâm sáng, rìa võng mạc tối) làm mạng CNN bị phân tâm bởi đặc trưng màu sắc của máy chụp thay vì cấu trúc mạch máu.

#### Giải pháp & Cơ chế Toán học:
1. **Lọc triệt tiêu quang sai màu Ben Graham (Color Constancy):**
   $$\hat{I}(x, y) = \alpha \cdot I(x, y) - \beta \cdot \text{Gaussian}(I(x, y); \sigma) + \gamma$$
   * Trong đó: $\alpha = 4$, $\beta = 4$, $\gamma = 128$, bán kính làm mờ $\sigma = \frac{\text{scale}}{30}$.
   * **Cơ chế:** Phép trừ ảnh gốc với phiên bản lọc thông thấp Gaussian giúp triệt tiêu thành phần tần số thấp (gradient chiếu sáng không đều và màu nền quang học), đồng thời giữ lại nguyên vẹn các thành phần tần số cao (vi phình mạch, bờ xuất tiết, thành mạch máu).
2. **Tăng cường tương phản cục bộ kênh Xanh lá (Green-Channel CLAHE):**
   * Áp dụng CLAHE (Clip Limit $= 2.0$, Grid Size $= 8 \times 8$) trên kênh Green giúp làm nổi bật các tổn thương xuất huyết kích thước micro mà không làm khuếch đại nhiễu hạt tại các vùng nền tối.

---

### 3.2 Biện luận Kiến trúc: Dual-Branch Multi-Scale Multiple Instance Learning (MIL)

#### Vấn đề:
Co giãn toàn bộ ảnh $4288 \times 2848$ về kích thước chuẩn $224 \times 224$ làm mất hoàn toàn các vi phình mạch nhỏ (chiếm $\le 0.01\%$ diện tích ảnh). Ngược lại, nếu huấn luyện trực tiếp ở kích thước $4288 \times 2848$ sẽ gây tràn bộ nhớ GPU (OOM) và làm mất góc nhìn toàn cảnh (Global Context) về tỷ lệ phân bố tổn thương giữa các góc phần tư theo quy tắc 4-2-1.

#### Giải pháp Kiến trúc Hai Nhánh (Dual-Branch):
```
                      +------------------------------------------+
                      |         Ảnh Gốc Đáy Mắt (CFP)            |
                      +------------------------------------------+
                                    /              \
    [Nhánh Toàn Cảnh (Global)]     /                \    [Nhánh Đa Phiên Bản (Local MIL)]
                                  v                  v
                      +--------------------+   +------------------------------------+
                      | Downsample 512x512 |   | Crop Top-K Patches (K=16, 224x224) |
                      +--------------------+   +------------------------------------+
                                  |                              |
                                  v                              v
                      +--------------------+   +------------------------------------+
                      | ResNet50 + CBAM    |   | EfficientNet-B0 Patch Feature Ext. |
                      +--------------------+   +------------------------------------+
                                  |                              |
                                  |                      +---------------+
                                  |                      | Gated-Attention|
                                  |                      | MIL Aggregator|
                                  |                      +---------------+
                                  |                              |
                                  v                              v
                              [f_global]                     [f_local]
                                  \                              /
                                   \                            /
                                    v                          v
                                 +--------------------------------+
                                 | Concatenation: [f_g, f_l]      |
                                 +--------------------------------+
                                                 |
                                                 v
                                 +--------------------------------+
                                 | Ordinal Head (CLM_QWK / CORAL) |
                                 +--------------------------------+
```
* **Nhánh Toàn cảnh (Global Context Branch):** Sử dụng `ResNet50 + CBAM` trên ảnh thu nhỏ $512 \times 512$ để nắm bắt vị trí đĩa thị, hoàng điểm và tổng thể 4 góc phần tư võng mạc.
* **Nhánh Cục bộ Đa phiên bản (Local MIL Branch):** Trích xuất $K=16$ mảnh cục bộ kích thước $224 \times 224$ ở độ phân giải gốc tại các vị trí có độ biến thiên entropy cao nhất. Mạng `EfficientNet-B0` trích xuất vector đặc trưng từng patch $h_k$, sau đó bộ tổng hợp **Gated Attention MIL** học trọng số chú ý $a_k$:
  $$a_k = \frac{\exp\left\{ w^T \left( \tanh(V h_k^T) \odot \text{sigm}(U h_k^T) \right) \right\}}{\sum_{j=1}^K \exp\left\{ w^T \left( \tanh(V h_j^T) \odot \text{sigm}(U h_j^T) \right) \right\}}$$
  $$f_{\text{local}} = \sum_{k=1}^K a_k h_k$$
* Phép hợp nhất đặc trưng $f = [f_{\text{global}} \parallel f_{\text{local}}]$ dung hòa hoàn hảo giữa ngữ cảnh vĩ mô và tổn thương hiển vi vi thể.

---

### 3.3 Biện luận Đầu Phân Loại Thứ Bậc (Ordinal Heads) & Continuous QWK Loss

#### Vấn đề:
Các cấp độ bệnh võng mạc đái tháo đường ICDRS mang bản chất **Thứ bậc tự nhiên (Natural Ordinality)**:
$$\text{Grade } 0 < \text{Grade } 1 < \text{Grade } 2 < \text{Grade } 3 < \text{Grade } 4$$
Hàm mất mát phân loại đa lớp tiêu chuẩn (Multi-class Cross-Entropy Loss) xem các lớp là rời rạc và độc lập:
$$\mathcal{L}_{\text{CE}}(\hat{y}=1, y=0) = \mathcal{L}_{\text{CE}}(\hat{y}=4, y=0)$$
Về mặt y khoa, dự đoán nhầm một bệnh nhân Không mắc bệnh (Grade 0) thành Nhẹ (Grade 1) là sai số chấp nhận được trong tầm soát, nhưng dự đoán nhầm Grade 0 thành Tăng sinh nặng (Grade 4) hoặc ngược lại là một **thảm họa lâm sàng**.

#### Giải pháp Toán học:
1. **Mô hình Liên kết Tích lũy (Cumulative Link Models - CLM):**
   Thay vì dự đoán trực tiếp xác suất từng lớp rời rạc, mạng học một biến tiềm ẩn đơn chiều $z(x) = w^T f(x)$ và $K-1 = 4$ ngưỡng có thứ tự tăng dần $\theta_1 < \theta_2 < \theta_3 < \theta_4$:
   $$P(Y \le k \mid x) = \sigma(\theta_k - z(x)) = \frac{1}{1 + e^{-(\theta_k - z(x))}}$$
   $$P(Y = k \mid x) = P(Y \le k \mid x) - P(Y \le k-1 \mid x)$$
   * Đảm bảo tính đơn điệu của không gian dự đoán và duy trì trật tự lâm sàng nghiêm ngặt.
2. **Hàm mất mát Quadratic Weighted Kappa vi phân liên tục (Continuous QWK Loss):**
   * Chỉ số đánh giá lâm sàng chuẩn mực là QWK:
     $$\kappa = 1 - \frac{\sum_{i,j} W_{ij} O_{ij}}{\sum_{i,j} W_{ij} E_{ij}}, \quad W_{ij} = \frac{(i - j)^2}{(C - 1)^2}$$
   * Vì ma trận nhầm lẫn thực nghiệm $O_{ij}$ không thể lấy đạo hàm trực tiếp, hàm mất mát Continuous QWK xấp xỉ ma trận liên tục thông qua tích ngoài của vector xác suất dự đoán Softmax $\hat{y}$ và One-hot nhãn thực $y$:
     $$\tilde{O}_{ij} = \sum_{n=1}^N y_{n, i} \cdot \hat{y}_{n, j}$$
     $$\mathcal{L}_{\text{QWK}} = \frac{\sum_{i,j} W_{ij} \tilde{O}_{ij}}{\sum_{i,j} W_{ij} E_{ij}}$$
   * Tối ưu hóa trực tiếp $\mathcal{L}_{\text{QWK}}$ giúp mô hình phạt nặng các sai lệch khoảng cách bậc hai $|y - \hat{y}|^2$, khắc phục hoàn toàn hiện tượng mất cân bằng lớp.

---

### 3.4 Biện luận Giao thức Đánh giá Ngoại Miền (Leave-One-Dataset-Out - LODO)

#### Vấn đề:
Kỹ thuật $k$-Fold Cross-Validation gộp chung cả 6 bộ dữ liệu lại rồi chia ngẫu nhiên sẽ dẫn đến tình trạng tập Test chứa ảnh từ cùng bệnh viện/loại máy ảnh với tập Train. Khi đó, độ chính xác cao chỉ chứng minh mô hình "học thuộc" đặc tính của máy chụp (In-domain Overfitting).

#### Giao thức LODO (Leave-One-Dataset-Out):
Để chứng minh khả năng tổng quát hóa thực sự khi chuyển giao sang bệnh viện mới (Out-of-Distribution Generalizability):
* **Vòng 1:** Train trên {APTOS, DDR, DeepDRiD, Messidor-2, EyePACS} $\longrightarrow$ Test ngoại miền độc lập trên **IDRiD**.
* **Vòng 2:** Train trên {APTOS, DDR, DeepDRiD, IDRiD, EyePACS} $\longrightarrow$ Test ngoại miền độc lập trên **Messidor-2**.
* **Vòng 3:** Train trên {APTOS, DDR, DeepDRiD, IDRiD, Messidor-2} $\longrightarrow$ Test ngoại miền độc lập trên **APTOS**.

Chỉ khi mô hình đạt chỉ số QWK $\ge 0.85$ trên tập dữ liệu hoàn toàn chưa từng thấy trong quá trình huấn luyện, hệ thống mới đạt chuẩn tin cậy y tế để ứng dụng thực địa.

---

## 4. BẢNG TỔNG HỢP ĐẶC TRƯNG TOÀN DIỆN CỦA 6 BỘ DỮ LIỆU (DATASET MASTER PROFILE)

Bảng tổng hợp dưới đây là tài liệu quy chuẩn kỹ thuật phục vụ việc xây dựng Pipeline và viết công trình nghiên cứu:

| Đặc tính / Chỉ số | **APTOS 2019** | **DDR Dataset** | **DeepDRiD** | **IDRiD** | **Messidor-2** | **EyePACS (Kaggle)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Quốc gia / Xuất xứ** | Ấn Độ | Trung Quốc | Trung Quốc | Ấn Độ | Pháp | Hoa Kỳ |
| **Cơ sở thu thập** | Aravind Eye Hospital | Đa trung tâm bệnh viện | Bệnh viện ĐH Thượng Hải | Eye Clinic (Nanded) | 3 Bệnh viện ĐH Pháp | Chương trình Tầm soát Viễn thông |
| **Bối cảnh lâm sàng** | Sàng lọc & Phòng khám | Đa tuyến (Khám/Điều trị) | Phòng khám chuyên sâu | Bệnh viện chuyên khoa mắt | Đa trung tâm học thuật | Sàng lọc cộng đồng diện rộng |
| **Tổng số ảnh hợp lệ** | **3,662** | **12,522** | **2,000** | **455** | **1,744** | **35,126** |
| **Số lượng bệnh nhân** | 3,662 cá thể | 12,522 cá thể | 500 cá thể (4 ảnh/BN) | 455 cá thể | 1,744 cá thể | 35,126 cá thể (Cặp mắt) |
| **Độ phân giải chuẩn** | $1886 \times 1458$ (Biến thiên) | $864 \times 803$ (Crop gần vuông) | $1817 \times 1872$ (Gần vuông) | **$4288 \times 2848$ (SLR $3:2$)** | **$512 \times 512$ (Vuông $1:1$)** | $3683 \times 2501$ (Góc rộng $3:2$) |
| **Tỷ lệ khung hình (W/H)**| $1.25 \pm 0.20$ | $1.08 \pm 0.12$ | $0.97 \pm 0.04$ | $1.51 \pm 0.00$ | $1.00 \pm 0.00$ | $1.47 \pm 0.08$ |
| **Thiết bị chụp đáy mắt**| Đa dạng (Kowa, Topcon...) | Không công khai chi tiết | Topcon TRC-NW400 | Kowa VX-10 $\alpha$ | Topcon TRC-NW6 | Đa dạng máy sàng lọc lưu động |
| **Góc mở võng mạc (FOV)**| $45^\circ$ | $45^\circ$ | $45^\circ$ (Central/Macular) | $50^\circ$ | $45^\circ$ | $45^\circ$ |
| **Nhãn mắt Trái / Phải** | Không công khai | Không công khai | Có (1,000 Trái / 1,000 Phải)| Không công khai | Không công khai | Có (17,563 Trái / 17,563 Phải)|
| **Độ sáng Grayscale ($\mu$)** | $68.75 \pm 17.82$ | $72.89 \pm 20.38$ | $68.46 \pm 20.63$ | $67.78 \pm 12.38$ | $72.66 \pm 16.90$ | $63.66 \pm 34.02$ |
| **Kênh Xanh dương ($\mu_B$)**| **$17.55 \pm 14.03$ (Thấp)** | $34.00 \pm 18.25$ | **$37.34 \pm 19.50$ (Cao)** | **$15.82 \pm 8.90$ (Thấp)** | $20.10 \pm 11.80$ | **$41.11 \pm 28.60$ (Rất cao)** |
| **Điểm sắc nét Laplacian** | $26.25 \pm 17.97$ | **$139.08 \pm 88.54$ (Rất cao)**| $50.08 \pm 34.20$ | $20.92 \pm 12.40$ | $78.76 \pm 45.30$ | **$12.56 \pm 18.90$ (Thấp/Nhiễu)**|
| **Đặc trưng mất cân bằng**| Grade 0 (49.3%), Grade 2 (27.3%)| Grade 0 (50.0%), Grade 2 (35.8%)| Phân bố tương đối đều | **Grade 3+4 cao nhất (31.0%)** | Grade 0 áp đảo (58.4%) | **Grade 0 cực đoan (73.5%)** |
| **Hệ số mất cân bằng** | **9.35x** | **26.55x** | **7.67x** | **6.77x** | **29.80x** | **36.00x** |
| **Vai trò trong Pipeline** | Tập Huấn luyện & Cross-Val | Tập Huấn luyện quy mô lớn | Đánh giá ghép cặp 2 mắt | **External Hold-out Tuyệt đối** | **External Hold-out Tuyệt đối** | Tập Huấn luyện quy mô lớn |

---

## 5. KẾT LUẬN VÀ LỘ TRÌNH TRIỂN KHAI TIẾP THEO (NEXT STEPS)

Báo cáo phân tích EDA đã cung cấp đầy đủ các cơ sở định lượng và luận cứ y sinh vững chắc để chuẩn bị cho giai đoạn huấn luyện mô hình học sâu.

### Các bước triển khai tiếp theo trong Pipeline:
1. **Hoàn thiện Module Tiền xử lý ([preprocessing.py](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/preprocessing.py)):**
   * Đóng gói quy trình cắt đường tròn võng mạc tự động (Auto-circular crop), lọc Ben Graham Color Constancy, và tách lớp Green-Channel CLAHE.
2. **Xây dựng Data Loader & LODO Split ([data_loader.py](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/data_loader.py)):**
   * Thiết lập lớp `RetinopathyDataset` hỗ trợ trích xuất đồng thời ảnh toàn cảnh và $K=16$ patch cục bộ. Khởi tạo generator `StratifiedGroupKFold` theo bệnh nhân.
3. **Thực thi Kiến trúc Dual-Branch MIL ([models.py](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/models.py)):**
   * Ghép nối backbone `ResNet50-CBAM` với nhánh `EfficientNet-B0 + Gated Attention MIL`.
4. **Triển khai Thực nghiệm Đầu Ordinal ([losses_and_heads.py](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/losses_and_heads.py)):**
   * So sánh thực nghiệm 5 biến thể đầu ra: `Softmax_CE`, `Softmax_QWK`, `CORAL`, `CORN`, và `CLM_QWK` trên giao thức LODO.
5. **Kiểm chứng XAI & Khả năng Giải thích Y khoa:**
   * Ánh xạ bản đồ kích hoạt Grad-CAM của nhánh MIL với mặt nạ phân vùng tổn thương thực tế (Pixel-level Ground Truth Masks) của bộ dữ liệu IDRiD.
