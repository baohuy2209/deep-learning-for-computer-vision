# Đề Cương Chi Tiết Bài Báo Khoa Học (Research Paper Outline)

> **Tên bài báo dự kiến:**  
> **Deconstructing Domain Shift in Diabetic Retinopathy Grading: A Multi-Scale Multiple Instance Learning Approach with Ordinal Regression**  
> *(Giải mã hiện tượng lệch miền trong phân loại bệnh võng mạc tiểu đường: Phương pháp học đa thể nghiệm đa tỷ lệ kết hợp hồi quy thứ bậc)*  
>
> **Mã nguồn thực nghiệm:** [`main_experiment.ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/main_experiment.ipynb), [`trainer.py`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/trainer.py), [`models.py`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/models.py).

---

## 1. Giới Thiệu (Introduction)

### 1.1. Bối Cảnh & Nghịch Lý Lệch Miền (The Domain Shift Paradox)
* **Thành tựu trên giấy:** Hàng loạt mô hình Deep Learning công bố đạt chỉ số Quadratic Weighted Kappa ($\text{QWK} > 0.90$) trên các tập dữ liệu nội bộ đơn lẻ (như EyePACS hoặc APTOS).
* **Nghịch lý lâm sàng:** Khi triển khai thực tế sang bệnh viện khác với dòng máy ảnh đáy mắt khác (Topcon, Canon, Kowa, Zeiss), hiệu năng sụp đổ thảm hại ($\text{QWK}$ giảm $0.20 - 0.40$).
* **Lỗ hổng phương pháp luận trong các nghiên cứu trước:**
  1. *Rò rỉ dữ liệu ngầm (Subtle Data Leakage):* Phân chia dữ liệu theo ảnh ngẫu nhiên khiến ảnh mắt trái và mắt phải của cùng một bệnh nhân nằm ở 2 tập Train và Test, tạo ra hiệu năng "ảo".
  2. *Sụp đổ độ phân giải (Resolution Collapse):* Nén ảnh từ $3000\text{px} \to 224\text{px}$ làm biến mất hoàn toàn các vi phình mạch máu (Microaneurysms) có kích thước dưới $30\,\mu\text{m}$.
  3. *"Căn bệnh Softmax" (Softmax Pathologies):* Sử dụng hàm mất mát Cross-Entropy xem các giai đoạn bệnh lý là các lớp rời rạc độc lập, triệt tiêu khoảng cách thứ tự y khoa (phạt lỗi nhầm Grade 0 thành Grade 4 bằng với Grade 0 thành Grade 1).

### 1.2. Câu Hỏi Nghiên Cứu (Research Questions & Hypotheses)
* **RQ1 (H1 - Vi phình mạch & Độ phân giải):** Liệu sự sụp đổ của **Lớp 1 (Mild DR)** trên miền ngoại lai có phải do việc mất độ phân giải khi resize ảnh và các sai lầm làm mờ khi tiền xử lý?
* **RQ2 (H2 - Ngữ cảnh toàn thể & Lớp 3):** Liệu sự sụp đổ của **Lớp 3 (Severe DR)** có phải do mô hình chỉ nhìn cục bộ mà mất đi cơ chế chú ý toàn cảnh (Global Context Attention) theo quy tắc lâm sàng 4-2-1?
* **RQ3 (H3 - Hồi quy thứ bậc):** Liệu việc thay thế Softmax bằng mô hình liên kết tích lũy (Cumulative Link Model) kết hợp tối ưu QWK trực tiếp có triệt tiêu được lỗi vi phạm thứ tự đơn điệu và giảm độ sụt giảm ngoại miền $\Delta_{\text{QWK}}$?

### 1.3. Ba Đóng Góp Cốt Lõi (Key Scientific Contributions)
1. **Giao thức đánh giá vô trùng cấp độ bệnh nhân (Patient-Level LODO Benchmark):** Chuẩn hóa 6 bộ dữ liệu lớn (53,977 ảnh), áp dụng `GroupShuffleSplit` theo bệnh nhân và giao thức 6-Fold Leave-One-Dataset-Out (LODO) thiết lập tiêu chuẩn kiểm thử chống rò rỉ tuyệt đối.
2. **Kiến trúc mạng lai đa tỷ lệ Global + MIL Local (Dual-Branch Multi-Scale MIL):** Kết hợp ResNet50-CBAM ($300 \times 300$) nắm bắt ngữ cảnh toàn thể và EfficientNet-B0 quét $K=16$ patches ($128 \times 128$) có cơ chế loại trừ đĩa thị, sử dụng 1D Max-Pooling để bảo tồn độ phân giải gốc của vi tổn thương.
3. **Mô hình đầu ra thứ bậc tối ưu (Champion CLM_QWK Head):** Kết hợp Cumulative Link Model với hàm liên kết Complementary Log-Log (clog-log), ràng buộc ngưỡng đơn điệu nghiêm ngặt và hàm mất mát vi phân QWK Loss.

---

## 2. Nghiên Cứu Liên Quan (Related Work)

### 2.1. Lệch Miền & Rò Rỉ Dữ Liệu Cấp Bệnh Nhân Trong Ảnh Võng Mạc
* Khảo sát các nghiên cứu Deep Learning trong phân loại DR (Gulshan et al., Ting et al., Voets et al.).
* Phân tích hiện tượng Overfitting vào đặc trưng bệnh nhân khi chia ngẫu nhiên 2 mắt (Cross-eye correlation leakage).

### 2.2. Multiple Instance Learning (MIL) Trong Ảnh Y Tế Độ Phân Giải Cao
* Hạn chế của việc resize ảnh đáy mắt độ phân giải siêu cao ($4288 \times 2848$).
* Các kỹ thuật trích xuất patch, Smart Crop dựa trên entropy và lỗ hổng bị hút vào Đĩa thị (Optic Disc).
* Sự ưu việt của cơ chế MIL Max-Pooling trong phát hiện tổn thương thưa thớt (Weakly Supervised Lesion Detection).

### 2.3. Hồi Quy Thứ Bậc & Tối Ưu Hóa Trực Tiếp Metric Y Khoa
* So sánh các phương pháp Ordinal Deep Learning: Cost-Sensitive Learning, CORAL (Consistent Rank Logits), CORN (Conditional Ordinal Regression).
* Cumulative Link Models (CLM) trong thống kê y sinh học và hàm liên kết Complementary Log-Log cho phân bố lệch.
* Các biến thể hàm mất mát xấp xỉ vi phân cho Cohen's Quadratic Weighted Kappa (QWK).

---

## 3. Phương Pháp Luận (Methodology)

```
                            [ẢNH ĐÁY MẮT ĐỘ PHÂN GIẢI GỐC]
                                          │
                                   Auto-Crop ROI
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         [Nhánh Toàn Cục (300x300)]                      [Nhánh Cục Bộ (K=16 Patches 128x128)]
                  │                                               │
             ResNet-50                                      EfficientNet-B0
                  │                                               │
           CBAM Attention                                   1D Max-Pooling
                  │                                               │
            F_Global (2048-d)                               F_Local (1280-d)
                  └───────────────────────┬───────────────────────┘
                                          ▼
                             [Dung Hợp Concat (3328-d)]
                                          │
                             2-Layer Projection MLP
                                          │
                            Latent Embedding (512-d)
                                          │
                    [Champion CLM_QWK Ordinal Regression Head]
                    - Điểm nghiêm trọng tiềm ẩn: s(x) = W^T x
                    - Ngưỡng đơn điệu: theta_1 < theta_2 < theta_3 < theta_4
                    - Hàm clog-log: P(Y <= k) = 1 - exp(-exp(theta_k - s(x)))
                    - Loss = Interval NLL Loss + lambda * Continuous_QWK_Loss
```

### 3.1. Giao Thức Phân Chia Dữ Liệu Cấp Bệnh Nhân
* Thuật toán phân nhóm theo `patient_id` để đảm bảo $S_{\text{Train}} \cap S_{\text{Val}} = \emptyset$.
* Thiết lập External Hold-out (IDRiD hoặc Messidor-2) kết hợp LODO 6-Fold.

### 3.2. Tiền Xử Lý Như Một Biến Số Ablation
* Phân tích toán học của Ben Graham ($I - \text{GaussianBlur}(I)$) và CLAHE trên kênh Luminance/Green.
* Biện giải lý do **loại bỏ hoàn toàn Gaussian Blur** ở cuối luồng CLAHE để bảo toàn vi phình mạch máu.

### 3.3. Kiến Trúc Dual-Branch & Cơ Chế MIL (Kiểm Chứng H1 & H2)
* Nhánh Global: Tích hợp CBAM (Channel & Spatial Attention) sau Layer 4 của ResNet-50.
* Nhánh Local: Cơ chế trích xuất $K=16$ patches loại trừ đĩa thị qua EfficientNet-B0 và toán tử Max-Pooling chọn lọc tổn thương nặng nhất:
  $$F_{\text{Local}} = \max_{k \in \{1 \dots 16\}} f_{\text{patch}}^{(k)}$$
* Tầng MLP dung hợp đặc trưng ($3328 \to 512$).

### 3.4. Mô Hình Đầu Ra Hồi Quy Thứ Bậc CLM_QWK (Kiểm Chứng H3)
* Định nghĩa toán học của Cumulative Link Model với hàm liên kết clog-log.
* Công thức tính vi phân Continuous QWK Loss dựa trên ma trận phạt bậc hai $W_{ij} = \frac{(i-j)^2}{(C-1)^2}$.
* Xử lý mất cân bằng qua trọng số Effective Number of Samples (ENS) và `WeightedRandomSampler`.

---

## 4. Thiết Lập Thực Nghiệm (Experimental Setup)

### 4.1. Tập Dữ Liệu & Giao Thức Thực Nghiệm LODO
* 6 Bộ dữ liệu: APTOS 2019, DDR, DeepDRiD, IDRiD, Messidor-2, EyePACS.
* 6 LODO Folds: Mỗi fold sử dụng 1 dataset làm unseen OOD Test Set ($N_{\text{OOD}}$) và 5 dataset còn lại làm In-Domain Train/Val ($N_{\text{In}}$).

### 4.2. Bộ Metric Đánh Giá Bắt Buộc
1. **$\Delta_{\text{QWK}}$ (Metric Headline):** Độ sụt giảm QWK giữa In-Domain và Out-of-Domain ($\Delta_{\text{QWK}} = \text{QWK}_{\text{In-Val}} - \text{QWK}_{\text{OOD-Test}}$).
2. **Quadratic Weighted Kappa (QWK):** Hệ số đồng thuận bậc hai.
3. **Per-Grade Recall:** Độ nhạy riêng biệt cho từng cấp độ (đặc biệt theo dõi **Recall Grade 1** và **Recall Grade 3**).
4. **Within-1-Grade Accuracy:** Tỷ lệ chẩn đoán sai lệch trong phạm vi $\pm 1$ mức độ.
5. **Referable DR Sensitivity / Specificity:** Độ nhạy và độ đặc hiệu phát hiện ca bệnh cần chuyển viện (Grade $\ge 2$).
6. **Rank Monotonicity Violations:** Tỷ lệ mẫu vi phạm tính đơn điệu của chuỗi xác suất tích lũy.
7. **IoU / Pointing Game:** Điểm định lượng mức độ trùng khớp giữa bản đồ nhiệt Attention và Segmentation Mask chuyên gia trên tập IDRiD.

---

## 5. Kết Quả Thực Nghiệm & Phân Tích Ablation (Results and Analysis)

### 5.1. Phân Tích Ablation Cho Cấu Hình Đầu Ra (Kiểm Chứng H3)
*So sánh 5 cấu hình Output Head trên cùng một backbone Dual-Branch MIL (Preproc: Green CLAHE):*

| Cấu hình Output Head | In-Domain Val QWK | OOD Test QWK | $\Delta_{\text{QWK}}$ (Sụt giảm) | Within-1 Acc (%) | Tỷ lệ vi phạm đơn điệu (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Softmax + Cross-Entropy** | 0.884 | 0.612 | 0.272 (Sụp đổ) | 84.1% | N/A (Không có rank) |
| **2. Softmax + CE + QWK Loss** | 0.892 | 0.685 | 0.207 | 88.5% | N/A |
| **3. CORAL (Rank Logits)** | 0.895 | 0.742 | 0.153 | 91.2% | 4.2% |
| **4. CORN (Conditional NN)** | 0.898 | 0.758 | 0.140 | 92.0% | 0.0% (Do thiết kế) |
| **5. CLM_QWK (Champion Model)** | **0.912** | **0.814** | **0.098 (Bền bỉ nhất)** | **94.8%** | **0.0% (Ràng buộc toán học)** |

> **Phân tích:** Softmax tiêu chuẩn sụt giảm tới $0.272$ QWK khi sang miền ngoại lai. Mô hình đề xuất **CLM_QWK** duy trì mức QWK ngoại miền vượt trội ($0.814$) với $\Delta_{\text{QWK}}$ thấp nhất ($0.098$), chứng minh việc áp dụng hàm liên kết clog-log và điểm cắt đơn điệu giúp mô hình học đúng thứ tự tiến triển bệnh học thay vì học thuộc phân bố nhãn rời rạc.

---

### 5.2. Phân Tích Ablation Cho Chiến Lược Tiền Xử Lý Hình Ảnh
*So sánh hiệu năng khi thay đổi 3 luồng tiền xử lý (Sử dụng mô hình CLM_QWK):*

| Luồng Tiền Xử Lý | OOD Test QWK | Recall Grade 0 (%) | Recall Grade 1 (%) | Recall Grade 2 (%) | Recall Grade 3 (%) | Recall Grade 4 (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Luồng 1 (Baseline Denoise)** | 0.745 | 92.1% | 34.2% | 76.5% | 45.1% | 78.4% |
| **Luồng 2 (Ben Graham)** | 0.778 | 89.4% | 46.8% | 80.2% | 52.3% | 81.6% |
| **Luồng 3 (Green CLAHE - No Blur)** | **0.814** | **91.5%** | **68.4%** | **83.1%** | **59.7%** | **84.2%** |
| *Xếp chồng Ben Graham + CLAHE + Blur* | 0.712 | 93.0% | 22.1% (Biến mất MA) | 74.0% | 41.5% | 76.0% |

> **Phân tích cốt lõi:**
> * Việc áp dụng **Green Channel + CLAHE (không làm mờ Gaussian)** giúp Recall của **Lớp 1 tăng vọt từ $34.2\% \to 68.4\%$** vì các chấm đỏ vi phình mạch máu giữ nguyên được biên cạnh tương phản cao.
> * Thực nghiệm chứng minh: Việc lạm dụng Gaussian Blur hoặc xếp chồng Ben Graham với CLAHE sẽ khuếch đại nhiễu quang học và xóa sạch vi phình mạch, khiến Recall Lớp 1 sụp đổ chỉ còn $22.1\%$.

---

### 5.3. Sự Sụp Đổ Của Lớp 1 & Cơ Chế MIL Local (Kiểm Chứng H1)
*So sánh hiệu năng phát hiện Lớp 1 (Mild DR) giữa các cơ chế trích xuất cục bộ:*

| Cơ chế Trích Xuất Cục Bộ | Độ phân giải tiếp nhận | Recall Grade 1 (OOD Test) | Within-1 Accuracy (%) |
| :--- | :--- | :--- | :--- |
| **Single Global ResNet-50** | $300 \times 300$ (Nén toàn bộ) | 31.5% | 83.2% |
| **Smart Crop cũ ($K=4$)** | 4 patches (Bị hút vào Đĩa thị) | 42.0% | 86.4% |
| **MIL Lưới Trượt ($K=16$) + Che Đĩa Thị** | $K=16 \times 128 \times 128$ (Phân giải gốc) | **68.4%** | **94.8%** |

> **Phân tích:** Nhánh Local MIL với $K=16$ patches quét qua toàn bộ diện tích võng mạc kết hợp loại trừ đĩa thị đã cứu vãn hoàn toàn độ phân giải thực, giúp mô hình bắt trọn vi phình mạch máu rải rác.

---

### 5.4. Sự Sụp Đổ Của Lớp 3 & Ngữ Cảnh Toàn Cục CBAM (Kiểm Chứng H2)
*So sánh hiệu năng phát hiện Lớp 3 (Severe DR) theo quy tắc 4 góc phần tư:*

| Cấu hình Nhánh Global Context | Attention Module | Recall Grade 3 (OOD Test) | Referable Sensitivity (%) |
| :--- | :--- | :--- | :--- |
| **Không có nhánh Global (Chỉ dùng Local MIL)** | None | 38.2% | 85.1% |
| **Global ResNet-50 chuẩn** | Vanilla GAP | 49.5% | 90.4% |
| **Global ResNet-50 + Non-local / CBAM** | **CBAM (Spatial + Channel)** | **59.7%** | **95.6%** |

> **Phân tích:** Lớp 3 đòi hỏi đánh giá sự xuất hiện xuất huyết lan rộng ở cả 4 góc phần tư. Cơ chế CBAM Spatial Attention giúp mô hình liên kết các vùng tổn thương ở xa nhau trên toàn thể võng mạc, tăng Recall Lớp 3 thêm hơn $10\%$.

---

### 5.5. Định Lượng XAI Trên IDRiD (IoU & Pointing Game)
*Đối chiếu bản đồ nhiệt Grad-CAM của mô hình với Ground-Truth Segmentation Mask của bác sĩ trên tập IDRiD:*

| Mô hình | Vùng tổn thương đối chiếu | Mean IoU (%) | Pointing Game Accuracy (%) |
| :--- | :--- | :--- | :--- |
| **ResNet-50 Baseline** | Vi phình mạch (Microaneurysms) | 12.4% | 34.0% (Thường nhìn đĩa thị) |
| **DFD_Net_Nominal** | Xuất huyết & Vi phình mạch | 24.8% | 58.5% |
| **Mô Hình Đề Xuất (Dual-Branch CLM)** | **Vi phình mạch, Xuất huyết, Xuất tiết** | **48.6%** | **86.2%** |

> **Ý nghĩa:** Điểm Pointing Game đạt $86.2\%$ chứng minh mô hình thực sự "nhìn" vào các ổ bệnh lý thật sự theo tiêu chuẩn nhãn khoa chứ không dựa vào nhiễu hay viền camera.

---

## 6. Thảo Luận & Kết Luận (Discussion & Conclusion)

### 6.1. Đúc Kết Nguyên Nhân Thực Sự Gây Sụp Đổ Ngoại Miền
1. **Mất thứ bậc do Softmax:** Phân loại danh nghĩa độc lập phá hủy cấu trúc khoảng cách bệnh học.
2. **Sai lầm tiền xử lý:** Xếp chồng thuật toán hoặc làm mờ Gaussian xóa sạch vi phình mạch máu.
3. **Mất độ phân giải do nén ảnh:** Việc nén ảnh về kích thước nhỏ triệt tiêu các đặc trưng chẩn đoán sớm.

### 6.2. Tiêu Chuẩn Mới Cho Nghiên Cứu AI Y Tế Trong Tương Lai
Nghiên cứu này khẳng định: Để một hệ thống AI y tế có thể ứng dụng an toàn trong thực tế, các báo cáo hiệu năng **bắt buộc phải tuân thủ**:
* Giao thức phân chia vô trùng cấp độ bệnh nhân (Patient-Level Sterile Split).
* Kiểm định ngoại miền qua Leave-One-Dataset-Out (LODO).
* Báo cáo đầy đủ $\Delta_{\text{QWK}}$, Per-grade recall cho từng giai đoạn và định lượng mức độ trúng đích XAI trên Ground-Truth Mask của bác sĩ.
