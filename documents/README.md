# Tài Liệu Kỹ Thuật & Nghiên Cứu: Pipeline Phân Loại Bệnh Võng Mạc Tiểu Đường (Diabetic Retinopathy)

> **Dự án:** Deconstructing Domain Shift in Diabetic Retinopathy Grading: A Multi-Scale Multiple Instance Learning Approach with Ordinal Regression  
> **Thư mục mã nguồn:** [`notebooks/diabetic_retinopathy/pipeline`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline)  
> **Ngôn ngữ tài liệu:** Tiếng Việt  

---

## 📌 Giới Thiệu Tổng Quan

Hệ thống tài liệu này được biên soạn từ việc phân tích chi tiết toàn bộ mã nguồn, cấu hình thử nghiệm và dữ liệu thực tế tại thư mục [`notebooks/diabetic_retinopathy/pipeline`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline). Hệ thống được thiết kế để giải quyết bài toán sụp đổ hiệu năng ngoại miền (**Domain Shift**) và duy trì tính thứ bậc y khoa (**Ordinal Severity Grading**) trong phân loại 5 giai đoạn Bệnh võng mạc tiểu đường (ICDRS 0–4).

---

## 🗂️ Cấu Trúc Tài Liệu

Bộ tài liệu được chia thành các chuyên đề chuyên sâu trong thư mục [`documents/`](file:///d:/Github/deep-learning-for-computer-vision/documents):

```plaintext
documents/
├── README.md                              # (File này) Tổng quan hệ thống tài liệu
├── dataset_eda_summary.md                 # Báo cáo chuyên sâu phân tích dữ liệu EDA (55,509 ảnh, 6 datasets)
├── 01_datasets_analysis.md                # Phân tích toàn diện 6 bộ dữ liệu DR & Vấn đề rò rỉ dữ liệu
├── 02_pipeline_architecture.md            # Kiến trúc Pipeline xử lý hoàn chỉnh (Multi-Scale MIL + Ordinal)
└── 03_paper_outline_and_methodology.md    # Outline chi tiết bài báo khoa học & Thiết kế thực nghiệm Ablation
```

---

## 📑 Tóm Tắt Các Tài Liệu Thành Phần

### 1. [Báo Cáo Phân Tích Dữ Liệu Chuyên Sâu EDA (dataset_eda_summary.md)](file:///d:/Github/deep-learning-for-computer-vision/documents/dataset_eda_summary.md)
* **Tổng quan phân tích 55,509 ảnh:** Đánh giá định lượng toàn diện trên 6 tập dữ liệu (APTOS 2019, DDR, DeepDRiD, IDRiD, Messidor-2, EyePACS).
* **Kiểm toán vô trùng (Data Integrity & Sanity Audit):** Phân chia cấp bệnh nhân (`patient_id`), chống rò rỉ dữ liệu qua cặp mắt (Left/Right eyes).
* **Phân tích Dịch chuyển Miền (Domain Shift):** 
  * *Label Shift:* Phân hóa tỷ lệ mất cân bằng từ $6.77\text{x}$ (IDRiD) đến $36.00\text{x}$ (EyePACS), Grade 0 chiếm 64.33% trong khi Grade 3 chỉ 3.34%.
  * *Covariate / Photometric Shift:* Sai lệch kích thước ($512\times 512$ vuông đến $4288\times 2848$ SLR 3:2), độ lệch màu kênh Xanh dương ($\mu_B$ từ $15.8$ đến $41.1$), và độ sắc nét quang học.
* **Biện luận phương pháp:** Cơ sở lý thuyết y sinh học bảo vệ Ben Graham Color Constancy, Green-Channel CLAHE, Dual-Branch Multi-Scale MIL, Ordinal Heads (CLM / CORAL / CORN), và Continuous QWK Loss.
* **Bảng tổng hợp đặc trưng Master:** Tra cứu nhanh thông số kỹ thuật, xuất xứ, thiết bị, và vai trò của từng bộ dữ liệu.

### 2. [Phân Tích 6 Bộ Dữ Liệu & Rò Rỉ Dữ Liệu Bệnh Nhân (01_datasets_analysis.md)](file:///d:/Github/deep-learning-for-computer-vision/documents/01_datasets_analysis.md)
* **6 Bộ Dữ Liệu:** APTOS 2019, DDR, DeepDRiD, IDRiD, Messidor-2, EyePACS (Tổng cộng hơn 53,900 ảnh).
* **Phân bố nhãn ICDRS:** Mất cân bằng dữ liệu nghiêm trọng (Lớp 0 chiếm đa số, Lớp 1 và Lớp 3 là các nhóm thiểu số nguy cơ cao).
* **Vấn đề Data Leakage cấp bệnh nhân:** Rò rỉ thông tin qua hai mắt của cùng một bệnh nhân (Left/Right eyes) và giải pháp phân chia vô trùng `GroupKFold` / `GroupShuffleSplit`.
* **Đặc tính kỹ thuật & Lệch miền:** Độ phân giải gốc ($4288 \times 2848$ đến $1024 \times 1024$), thiết bị chụp (Topcon, Zeiss, Canon), sự khác biệt về độ sáng, độ tương phản và độ sắc nét Laplacian.

### 2. [Kiến Trúc Pipeline Xử Lý Hoàn Chỉnh (02_pipeline_architecture.md)](file:///d:/Github/deep-learning-for-computer-vision/documents/02_pipeline_architecture.md)
* **Giai đoạn 1: Hợp nhất & Phân chia Vô trùng (Strict Sterile Split):** Cơ chế `GroupShuffleSplit` theo `patient_id` và giao thức kiểm thử ngoại miền `Leave-One-Dataset-Out` (LODO).
* **Giai đoạn 2: Tiền xử lý như một Biến số Ablation:** 
  * Auto-crop ROI loại bỏ viền đen không chẩn đoán.
  * Tách biệt rõ ràng 3 luồng tiền xử lý (Baseline Denoise, Ben Graham, Green Channel + CLAHE).
  * **Quy tắc cốt lõi:** Tuyệt đối không xếp chồng Ben Graham + CLAHE và không sử dụng Gaussian Blur ở bước cuối của CLAHE để bảo toàn vi phình mạch (Microaneurysms - dấu hiệu độc quyền của Lớp 1).
* **Giai đoạn 3: Trích xuất Đặc trưng Lai Đa tỷ lệ (Global + MIL Local):**
  * *Nhánh Global (Ngữ cảnh toàn cục):* $300 \times 300$ qua ResNet-50 + CBAM Attention $\to F_{Global} \in \mathbb{R}^{2048}$.
  * *Nhánh Local (Chi tiết tổn thương vi mô):* $K=16$ patches $128 \times 128$ qua cơ chế cửa sổ trượt MIL loại trừ đĩa thị $\to$ EfficientNet-B0 $\to$ 1D Max-Pooling $\to F_{Local} \in \mathbb{R}^{1280}$.
  * *Dung hợp đặc trưng:* Concat $[F_{Global}, F_{Local}] \to$ 2-layer MLP $\to D_{Latent} = 512$.
* **Giai đoạn 4: Phân loại Thứ bậc & Tối ưu hóa:**
  * 5 Cấu hình đầu ra: Softmax_CE, Softmax_QWK, CORAL, CORN, và **CLM_QWK (Champion Model - Cumulative Link Model clog-log)**.
  * Hàm mất mát vi phân QWK (Continuous Differentiable Quadratic Weighted Kappa Loss).
  * Bộ cân bằng lấy mẫu: Effective Number of Samples (ENS) & `WeightedRandomSampler`.
* **Giai đoạn 5: Định lượng XAI (Explainable AI):**
  * Grad-CAM trên Local Patches, đánh giá độ trúng đích qua IoU / Pointing Game đối chiếu với Ground-Truth Segmentation Mask của IDRiD.
* **Hệ thống Mô hình Đối chuẩn (Benchmark Baselines):**
  * *Tier 1 (Single Anchor):* ResNet50, EfficientNet-B0, DenseNet121.
  * *Tier 2 (Complex SOTA):* EfficientNetB0_MSAG, Hybrid_EffB0_ViT, EfficientNet-B3, DFD_Net_Nominal.

### 3. [Outline Chi Tiết Bài Báo Nghiên Cứu (03_paper_outline_and_methodology.md)](file:///d:/Github/deep-learning-for-computer-vision/documents/03_paper_outline_and_methodology.md)
* **Tiêu đề bài báo:** *Deconstructing Domain Shift in Diabetic Retinopathy Grading: A Multi-Scale Multiple Instance Learning Approach with Ordinal Regression*
* **Cấu trúc 6 chương chuẩn quốc tế:**
  1. Introduction (Nghịch lý Domain Shift & Đóng góp khoa học).
  2. Related Work (Domain Shift, MIL trong ảnh võng mạc, Hồi quy thứ bậc).
  3. Methodology (Toán học của Patient-level split, Preprocessing, Dual-Branch MIL, và CLM Head).
  4. Experimental Setup (Giao thức LODO, Bộ metric bắt buộc: $\Delta_{QWK}$, Per-grade recall, Within-1-grade accuracy, Monotonicity Violations, IoU).
  5. Results and Analysis (5 mảng phân tích Ablation: Output Head, Tiền xử lý ảnh, Cứu vãn Lớp 1 bằng MIL, Cứu vãn Lớp 3 bằng Global Context, Định lượng XAI trên IDRiD).
  6. Discussion & Conclusion (Nguyên nhân cốt lõi gây sụp đổ ngoại miền và tiêu chuẩn đánh giá mới).

---

## 🔗 Liên Kết Nhanh Tới File Mã Nguồn

| Thành phần | Đường dẫn mã nguồn | Mô tả |
| :--- | :--- | :--- |
| **Data Loader** | [`notebooks/diabetic_retinopathy/pipeline/full/data_loader.py`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/data_loader.py) | Ingestion 6 dataset, GroupKFold theo bệnh nhân, LODO generator, PyTorch Dual-Branch Dataset |
| **Preprocessing** | [`notebooks/diabetic_retinopathy/pipeline/full/preprocessing.py`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/preprocessing.py) | Auto-crop ROI, 3 chế độ tiền xử lý Ablation, torchvision transforms |
| **Dual-Branch Model** | [`notebooks/diabetic_retinopathy/pipeline/full/models.py`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/models.py) | ResNet50 + CBAM (Global), EfficientNet-B0 + MIL Max-Pooling (Local), Fusion MLP |
| **Losses & Heads** | [`notebooks/diabetic_retinopathy/pipeline/full/losses_and_heads.py`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/losses_and_heads.py) | Softmax_CE, Softmax_QWK, CORAL, CORN, CLM_QWK, Continuous QWK Loss |
| **Baselines** | [`notebooks/diabetic_retinopathy/pipeline/full/baselines.py`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/baselines.py) | Tier 1 (ResNet50, EffNetB0, DenseNet121) & Tier 2 (MSAG, Hybrid ViT, EffNetB3, DFD-Net) |
| **Trainer & Metrics** | [`notebooks/diabetic_retinopathy/pipeline/full/trainer.py`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/trainer.py) | Training loop AMP, tính toán QWK, $\Delta_{QWK}$, Within-1 Acc, Referable DR, Monotonicity Violations |
| **Main Experiment** | [`notebooks/diabetic_retinopathy/pipeline/main_experiment.ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/main_experiment.ipynb) | Notebook chạy toàn bộ thử nghiệm LODO 6-fold |
| **EDA Notebook** | [`notebooks/diabetic_retinopathy/pipeline/eda/eda_6_datasets.ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/eda/eda_6_datasets.ipynb) | Khảo sát phân bố, kích thước ảnh, độ sáng, độ sắc nét của 6 datasets |
