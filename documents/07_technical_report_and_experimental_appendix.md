# Báo Cáo Kỹ Thuật & Phụ Lục Thực Nghiệm (Technical Report & Experimental Appendix)
## Phân Độ Bệnh Võng Mạc Tiểu Đường Đa Miền Qua Kiến Trúc Dual-Branch MIL & Cumulative Link Model (CLM)

> **Dự án Nghiên cứu:** Deconstructing Domain Shift in Diabetic Retinopathy Grading: A Multi-Scale Multiple Instance Learning Approach with Ordinal Regression  
> **Tài liệu tham chiếu:** [`documents/`](file:///d:/Github/deep-learning-for-computer-vision/documents) | **Mã nguồn thực nghiệm:** [`notebooks/diabetic_retinopathy/pipeline/`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline)  
> **Các notebook cốt lõi:**
> - Mô hình chính Champion V3: [`main_model (6).ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/model/main_model%20(6).ipynb)
> - Đối chứng 5 cấu hình đầu ra (Head Ablation): [`main_model_head_ablation (3).ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation%20(3).ipynb)
> - Kiểm nghiệm đa miền Leave-One-Dataset-Out (LODO): [`main_model_lodo.ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb)
> - Chuẩn hóa & Hài hòa dữ liệu: [`merge-dataset.ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge-dataset.ipynb) & [`merge_dataset_lodo.ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge_dataset_lodo.ipynb)
> - Điểm chuẩn In-Domain đơn miền: [`oracle/`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/oracle) (`oracle_aptos.ipynb`, `oracle_ddr.ipynb`, `oracle_deepdrid.ipynb`, `oracle_eyepacs.ipynb`, `oracle_idrid.ipynb`, `oracle_messidor2.ipynb`)

---

## 1. Kiến Trúc Hệ Thống & Thiết Kế Mạng (System Architecture)

Hệ thống được thiết kế theo cấu trúc hai nhánh đa tỷ lệ (**Multi-Scale Dual-Branch Backbone**) kết hợp học đa thể hiện (**Multiple Instance Learning - MIL**) và phân loại thứ bậc nhất quán (**Cumulative Link Model - CLM**), nhằm giải quyết đồng thời hai thách thức lớn nhất trong phân loại ảnh đáy mắt y tế:
1. **Resolution Collapse (Mất mát tổn thương vi mô):** Khi nén ảnh chụp đáy mắt có độ phân giải gốc siêu cao ($>3000 \times 2000$) về kích thước chuẩn ($512 \times 512$), các vi phình mạch (microaneurysms $\le 120\,\mu\text{m}$, kích thước dưới $15\text{ px}$) bị triệt tiêu, dẫn đến sụp đổ độ nhạy phân loại ở độ bệnh nhẹ (Grade 1 Mild NPDR).
2. **Clinical Spatial Geometry (Quy tắc không gian đa góc phần tư):** Tiêu chuẩn vàng lâm sàng ICDR/ETDRS định nghĩa độ bệnh nặng (Grade 3 Severe NPDR) dựa trên quy tắc "4-2-1" (xuất huyết võng mạc ở cả 4 góc phần tư, phình tĩnh mạch dạng chuỗi hạt ở $\ge 2$ góc, hoặc bất thường vi mạch nội võng mạc IRMA ở $\ge 1$ góc). Mô hình CNN toàn cục thông thường thiếu cơ chế mô hình hóa quan hệ không gian xa giữa các góc này.

```
                                  [Ảnh Đáy Mắt Độ Phân Giải Gốc (Raw Fundus)]
                                                        │
                         ┌──────────────────────────────┴──────────────────────────────┐
                         ▼                                                             ▼
            [Nhánh Toàn Cục (Global Branch)]                               [Nhánh Cục Bộ (Local Patch MIL)]
             Ảnh chuẩn hóa: 512 x 512 x 3                                   Cửa sổ trượt chồng lấn: p=128, s=96
                         │                                                  Lọc quang học: FOV area >= 0.5
                     ResNet-50                                              Bags: M in [32, 64] (Train) | 256 (Eval)
                         │                                                             │
                 NonLocalBlock2D                                                EfficientNet-B0
             (Tương tác không gian H x W)                                      (Trích xuất đặc trưng patch)
                         │                                                             │
                  CBAM Attention                                                 h_k in R^1280
            (Channel + Spatial 7x7)                                                    │
                         │                                              ┌──────────────┴──────────────┐
          ┌──────────────┴──────────────┐                               ▼                             ▼
          ▼                             ▼                         1D Max-Pooling           Gated-Attention Pool
     Global GAP              QuadrantTokenAggregator             max_k h_k in R^1280       alpha_k * h_k in R^1280
      (2048-d)             (Transformer 4 tokens, 2048-d)               (Grade 1 MA)              (Grade 2-4 Context)
          └──────────────┬──────────────┘                               └──────────────┬──────────────┘
                         ▼                                                             ▼
                 F_Global (4096-d)                                             F_Local (2560-d)
                         │                                                             │
                         └──────────────────────────────┬──────────────────────────────┘
                                                        ▼
                                       F_Fused = [F_Global || F_Local] in R^6656
                                                        │
                                             LayerNorm Fusion MLP
                                             Linear(6656 -> 512) -> LayerNorm -> Mish -> Dropout(0.3)
                                             Linear(512 -> 512) -> LayerNorm
                                                        │
                                          Vector đặc trưng z in R^512
                                                        │
                                 [Cumulative Link Model (CLM_QWK) Output Head]
                                 Điểm số vô hướng tiềm ẩn: s(x) = w^T z
                                 Ngưỡng cắt đơn điệu: theta_k qua Softplus
                                 Hàm liên kết bất đối xứng: clog-log CDF
```

---

### 1.1. Chi Tiết Cấu Trúc Hai Nhánh (Dual-Branch Design)

#### 1.1.1. Nhánh Toàn Cục (Global Context Branch)
* **Backbone:** ResNet-50 khởi tạo trọng số tiền huấn luyện ImageNet. Nhận đầu vào là ảnh đáy mắt toàn cảnh kích thước $512 \times 512 \times 3$.
* **NonLocalBlock2D (Wang et al., 2018):** Được tích hợp ngay sau Layer 4 của ResNet-50 trên bản đồ đặc trưng $C = 2048, H = 16, W = 16$. Khối tính toán ma trận tự chú ý không gian đầy đủ trên toàn bộ $H \times W = 256$ vị trí:
  $$\theta(x) = \text{Conv2D}_{1\times 1}(x; W_\theta), \quad \phi(x) = \text{Conv2D}_{1\times 1}(x; W_\phi), \quad g(x) = \text{Conv2D}_{1\times 1}(x; W_g)$$
  $$A = \text{Softmax}\left(\frac{\theta(x)^T \phi(x)}{\sqrt{C/r}}\right) \in \mathbb{R}^{HW \times HW}, \quad y = A \cdot g(x)^T$$
  $$z_{\text{nonlocal}} = \text{BatchNorm2D}(\text{Conv2D}_{1\times 1}(y; W_z)) + x$$
  với tỷ lệ giảm kênh $r = 2$ (giảm chiều từ $2048 \to 1024$). Cơ chế này thiết lập các liên kết phụ thuộc xa, cho phép tổn thương vi mạch ở góc phần tư này điều biến trực tiếp biểu diễn của góc phần tư đối diện.
* **Convolutional Block Attention Module (CBAM):** Áp dụng tuần tự cơ chế tập trung kênh (**Channel Attention**) và không gian (**Spatial Attention**):
  - *Channel Attention:*
    $$\mathbf{M}_c(F) = \sigma\left(W_1\left(W_0(\text{AvgPool}(F))\right) + W_1\left(W_0(\text{MaxPool}(F))\right)\right)$$
    với tỷ lệ rút gọn kênh $\text{reduction\_ratio} = 16$, đưa kênh từ $2048 \to 128 \to 2048$.
  - *Spatial Attention:* Áp dụng tích chập kích thước lớn $7 \times 7$ trên kết hợp của MaxPool và AvgPool dọc trục kênh:
    $$\mathbf{M}_s(F') = \sigma\left(f^{7\times 7}\left([\text{AvgPool}(F'); \text{MaxPool}(F')]\right)\right)$$
* **QuadrantTokenAggregator (Bộ Tổng Hợp Token 4 Góc Phần Tư):**
  - Bản đồ đặc trưng sau CBAM ($B \times 2048 \times H \times W$) được phân chia thành 4 góc phần tư giải phẫu học: Trên-Trái ($Q_1$), Trên-Phải ($Q_2$), Dưới-Trái ($Q_3$), Dưới-Phải ($Q_4$).
  - Mỗi góc được nén qua `AdaptiveAvgPool2d((1, 1))` thành 1 token có số chiều $C = 2048$.
  - Chuỗi 4 tokens được cộng với véc-tơ vị trí học được `quadrant_pos_embed` $\in \mathbb{R}^{1 \times 4 \times 2048}$ và đưa qua một tầng Transformer Encoder (`TransformerEncoderLayer(d_model=2048, nhead=4, dim_feedforward=2048, dropout=0.1, batch_first=True)`).
  - Tầng này tổng hợp thông tin quan hệ tương hỗ giữa 4 góc phần tư, trực tiếp mô hình hóa quy tắc lâm sàng ICDR 4-2-1 để nhận diện bệnh mức độ nặng **Grade 3 ($H_2$)**.
* **Đặc trưng Nhánh Toàn Cục Kết Hợp:**
  $$F_{\text{Global}} = [F_{\text{GAP}} \,\|\, F_{\text{Quadrant}}] \in \mathbb{R}^{4096}$$
  trong đó $F_{\text{GAP}} \in \mathbb{R}^{2048}$ là Global Average Pooling chuẩn và $F_{\text{Quadrant}} \in \mathbb{R}^{2048}$ là trung bình cộng của 4 tokens sau Transformer.

---

#### 1.1.2. Nhánh Cục Bộ Đa Thể Hiện (Local Patch MIL Branch)
* **Trích xuất Cửa Sổ Trượt (Sliding-Window Patching v2):**
  - Quét trên ảnh đáy mắt độ phân giải gốc với kích thước patch $p = 128 \times 128\text{ px}$, bước nhảy (stride) $s = 96\text{ px}$, tạo độ chồng lấn (overlap) $25\%$.
  - **Lọc Quang Học Retinal FOV:** Chỉ giữ lại các patch có diện tích thuộc vùng võng mạc quang học $\ge 50\%$ ($\text{mean}(\text{mask}_{\text{patch}}) \ge 0.5$).
  - *Chế độ Huấn luyện:* Lấy mẫu ngẫu nhiên $M \in [32, 64]$ patch mỗi ảnh (vừa đảm bảo tính ngẫu nhiên hóa điều hòa vừa kiểm soát bộ nhớ VRAM).
  - *Chế độ Đánh giá:* Quét toàn bộ lưới tọa độ hợp lệ (lên đến 256 patches), loại bỏ hoàn toàn sai số ngẫu nhiên khi kiểm tra.
* **Bộ Mã Hóa Instance (Instance Feature Extractor):**
  - Từng patch kích thước $128 \times 128 \times 3$ được đưa qua EfficientNet-B0 tiền huấn luyện, trích xuất véc-tơ đặc trưng $h_k \in \mathbb{R}^{1280}$ sau bước GAP.
* **Dual Aggregation Pooling (Gom cụm đặc trưng kép):**
  1. *1D Max-Pooling:* Trích xuất giá trị cực đại theo từng chiều đặc trưng trên toàn túi patch:
     $$F_{\text{Max}} = \left[\max_{k=1\dots N} h_{k, 1}, \, \max_{k=1\dots N} h_{k, 2}, \, \dots, \, \max_{k=1\dots N} h_{k, 1280}\right]^T \in \mathbb{R}^{1280}$$
     Cơ chế này nhạy cảm tối đa với các tổn thương vi mô cực nhỏ độc lập (vi phình mạch đơn lẻ của **Grade 1 Mild NPDR**, kiểm chứng giả thuyết $H_1$).
  2. *Masked Gated-Attention Pooling (Ilse et al., 2018):*
     $$a_k = \frac{\exp\left(\mathbf{w}^T (\tanh(\mathbf{V} h_k) \odot \sigma(\mathbf{U} h_k)) + \text{mask\_bias}_k\right)}{\sum_{j=1}^N \exp\left(\mathbf{w}^T (\tanh(\mathbf{V} h_j) \odot \sigma(\mathbf{U} h_j)) + \text{mask\_bias}_j\right)}$$
     trong đó $\mathbf{w} \in \mathbb{R}^{128}$, $\mathbf{V} \in \mathbb{R}^{128 \times 1280}$, $\mathbf{U} \in \mathbb{R}^{128 \times 1280}$, $\text{mask\_bias}_k = 0$ nếu patch hợp lệ và $-10^4$ nếu là patch đệm (padding). Đặc trưng gom cụm có trọng số:
     $$F_{\text{Att}} = \sum_{k=1}^N a_k h_k \in \mathbb{R}^{1280}$$
* **Đặc trưng Nhánh Cục Bộ Kết Hợp:**
  $$F_{\text{Local}} = [F_{\text{Max}} \,\|\, F_{\text{Att}}] \in \mathbb{R}^{2560}$$

---

### 1.2. Cơ Chế Kết Hợp & Trích Xuất Đặc Trưng (MLP Fusion)

Véc-tơ toàn cục $F_{\text{Global}} \in \mathbb{R}^{4096}$ và véc-tơ cục bộ $F_{\text{Local}} \in \mathbb{R}^{2560}$ được ghép nối trực tiếp:
$$F_{\text{Fused}} = [F_{\text{Global}} \,\|\, F_{\text{Local}}] \in \mathbb{R}^{6656}$$
Sau đó, $F_{\text{Fused}}$ được đưa qua mạng nơ-ron đa tầng kết hợp chuẩn hóa tầng (**LayerNorm Fusion MLP**):
$$\mathbf{z}_1 = \text{Mish}\left(\text{LayerNorm}\left(\mathbf{W}_{\text{proj}1} F_{\text{Fused}} + \mathbf{b}_1\right)\right), \quad \mathbf{W}_{\text{proj}1} \in \mathbb{R}^{512 \times 6656}$$
$$\mathbf{z} = \text{LayerNorm}\left(\mathbf{W}_{\text{proj}2} (\text{Dropout}_{0.3}(\mathbf{z}_1)) + \mathbf{b}_2\right), \quad \mathbf{W}_{\text{proj}2} \in \mathbb{R}^{512 \times 512}$$
Việc sử dụng `LayerNorm` thay vì `BatchNorm` tại khâu dung hợp là quyết định thiết kế then chốt, giúp ổn định phương sai gradient khi kích thước batch nhỏ ($B=16$) và số lượng patch cục bộ thay đổi linh hoạt. Kết quả là không gian nhúng tiềm ẩn đa chiều $\mathbf{z} \in \mathbb{R}^{512}$.

---

### 1.3. Đầu Ra Phân Loại Thứ Bậc Nhất Quán (Cumulative Link Model - CLM)

Thay vì xử lý 5 phân độ bệnh như các lớp độc lập không có quan hệ (Nominal Classification qua Softmax thông thường), đầu ra CLM mô hình hóa bệnh võng mạc tiểu đường như một tiến trình sinh học đơn điệu liên tục với biến tiềm ẩn mức độ nghiêm trọng $S(\mathbf{x})$.

#### 1.3.1. Điểm Tiềm Ẩn Mức Độ Nghiêm Trọng (Latent Severity Score)
Từ vector đặc trưng $\mathbf{z} \in \mathbb{R}^{512}$, một lớp tuyến tính chiếu xuống điểm vô hướng duy nhất:
$$s(\mathbf{x}) = \mathbf{w}_{\text{score}}^T \mathbf{z}, \quad \mathbf{w}_{\text{score}} \in \mathbb{R}^{512}$$

#### 1.3.2. Cơ Chế Tham Số Hóa Ngưỡng Cắt Đơn Điệu (Monotonic Cutpoints Parameterization)
Mô hình cần 4 ngưỡng cắt $\{\theta_1, \theta_2, \theta_3, \theta_4\}$ để phân chia trục thực $s(\mathbf{x})$ thành 5 khoảng tương ứng với 5 phân độ ICDRS ($0, 1, 2, 3, 4$). Để đảm bảo $\theta_1 < \theta_2 < \theta_3 < \theta_4$ một cách tuyệt đối (loại bỏ hoàn toàn vi phạm đảo ngược thứ bậc mà không cần chiếu cắt ngọn hậu nghiệm), các ngưỡng cắt được tham số hóa thông qua hàm `softplus`:
$$\theta_1 = \text{cutpoint}_0 \in \mathbb{R}$$
$$\theta_{k+1} = \theta_k + \text{softplus}(\alpha_k) + 10^{-4}, \quad k \in \{1, 2, 3\}$$
trong đó $\alpha_k \in \mathbb{R}$ là các trọng số tự do không bị ràng buộc được tối ưu cùng mạng nơ-ron, $\text{softplus}(\alpha) = \ln(1 + e^\alpha) > 0$. Lượng gia dương tối thiểu $10^{-4}$ đảm bảo tính đơn điệu ngặt $\theta_{k+1} > \theta_k$, loại trừ xác suất âm hoặc phân phối suy biến.

#### 1.3.3. Hàm Liên Kết Bất Đối Xứng (Complementary Log-Log Link Function)
Khác với hàm Logistic hay Probit đối xứng, hàm liên kết **Complementary Log-Log (clog-log)** được lựa chọn dựa trên bản chất lâm sàng của bệnh võng mạc tiểu đường: bệnh có diễn tiến chậm ở giai đoạn đầu nhưng bùng phát và phân nhánh nhanh chóng ở các giai đoạn tăng sinh nặng.
Hàm phân phối tích lũy (CDF) tương ứng:
$$F(t) = 1 - \exp(-\exp(t))$$
Xác suất tích lũy một mắt có mức độ bệnh không vượt quá phân độ $k$:
$$P(Y \le k \mid \mathbf{x}) = F(\theta_k - s(\mathbf{x})) = 1 - \exp(-\exp(\theta_k - s(\mathbf{x}))), \quad k \in \{0, 1, 2, 3\}$$
và quy ước tiên nghiệm $P(Y \le 4 \mid \mathbf{x}) = 1.0$.

#### 1.3.4. Xác Suất Từng Phân Độ & Truyền Gradient Không Suy Biến
Xác suất của từng phân độ bệnh được trích xuất chính xác qua hiệu số của các ngưỡng tích lũy liên tiếp:
$$P(Y = 0) = P(Y \le 0)$$
$$P(Y = k) = P(Y \le k) - P(Y \le k - 1), \quad k \in \{1, 2, 3\}$$
$$P(Y = 4) = 1 - P(Y \le 3)$$
Nhờ tính chất đơn điệu ngặt của $\theta_k$, $P(Y \le k) > P(Y \le k-1)$ luôn thỏa mãn, loại bỏ nhu cầu sử dụng hàm `F.relu()` để kẹp chặn xác suất. Điều này loại trừ triệt để hiện tượng "vùng chết gradient" (zero gradient clipping), cho phép tín hiệu lan truyền ngược liên tục qua toàn bộ không gian tham số.

---

## 2. Không Gian Siêu Tham Số & Chiến Lược Tối Ưu (Hyperparameters & Training Dynamics)

### 2.1. Bảng Tổng Hợp Siêu Tham Số Huấn Luyện

| Nhóm Siêu Tham Số | Tên Biến trong Code | Giá Trị Cấu Hình | Rationale & Ghi Chú Kỹ Thuật |
| :--- | :--- | :--- | :--- |
| **Kích thước ảnh** | `IMG_SIZE` | $512 \times 512$ | Cân bằng giữa độ bao phủ mạch máu và bộ nhớ VRAM |
| **Kích thước Batch** | `BATCH_SIZE` | 16 (Full) / 48 (Ablation) | 16 tối ưu cho GPU 16GB VRAM (T4/L4) khi chạy Dual-Branch |
| **Số Epoch tối đa** | `EPOCHS` / `LODO_EPOCHS` | 100 (Full) / 30 (LODO) / 25 (Ablation) | Đủ dài cho sự hội tụ của mạng Transformer và CLM |
| **Optimizer** | `torch.optim.AdamW` | AdamW | Chống quá khớp tốt hơn Adam tiêu chuẩn |
| **Initial Learning Rate**| `LEARNING_RATE` | $1.0 \times 10^{-4}$ | Tốc độ học chuẩn, tránh phá vỡ trọng số tiền huấn luyện |
| **Weight Decay** | `WEIGHT_DECAY` | $1.0 \times 10^{-4}$ (Full/LODO) / $1.0 \times 10^{-3}$ (Ablation) | Kiểm soát chuẩn L2 của các tầng Fully Connected |
| **Tốc độ học tối thiểu**| `MIN_LR` | $1.0 \times 10^{-6}$ | Ngưỡng chặn dưới của Learning Rate Scheduler |
| **Learning Rate Scheduler**| `CosineAnnealingLR` | $T_{\max} = 50$ (Full) / $30$ (LODO) / $25$ (Ablation), $\eta_{\min} = 10^{-6}$ | Giảm tốc độ học êm dịu theo chu kỳ cosine |
| **Adaptive LR Step** | `ReduceLROnPlateau` | `patience = 4`, `factor = 0.5` | Giảm 50% LR nếu `val_loss` chững lại 4 epochs liên tiếp |
| **Độ dốc Gradient Clip** | `GRAD_CLIP` | $1.0$ | Cắt ngọn gradient theo chuẩn $L_2$, ngăn chặn bùng nổ gradient |
| **Độ chính xác hỗn hợp** | `USE_AMP` | `True` (`torch.cuda.amp`) | FP16 trên T4, BF16 trên L4/A100; tăng tốc $2.5\times$ |
| **Trọng số QWK mục tiêu**| `TARGET_QWK_WEIGHT` ($\lambda$) | $0.4$ | Giảm từ 1.2 xuống 0.4 để triệt tiêu sốc gradient ma trận nhầm lẫn |
| **Thời gian QWK Warm-up**| `QWK_WARMUP_EPOCHS` | $15$ epochs | Tăng từ 5 lên 15 epochs để định hình manifold phân loại trước |
| **Early Stopping** | `PATIENCE_EARLY_STOPPING` | $15$ epochs | Dừng huấn luyện dựa trên chỉ số **Validation QWK** |

---

### 2.2. Lịch Điều Chỉnh Loss Đa Nhiệm (Multi-task Loss Scheduling)

Mô hình tối ưu hóa đồng thời hai mục tiêu bổ trợ:
$$\mathcal{L}_{\text{CLM\_QWK}}(t) = \mathcal{L}_{\text{NLL}} + \lambda(t) \cdot \mathcal{L}_{\text{Continuous\_QWK}}$$

#### 2.2.1. Hàm Mất Mát Phân Phối Thứ Bậc (Ordinal Negative Log-Likelihood - NLL)
$$\mathcal{L}_{\text{NLL}} = -\frac{1}{B} \sum_{i=1}^B \log\left(P\left(Y = y_i \mid \mathbf{x}_i\right) + \epsilon\right)$$
với $\epsilon = 10^{-7}$. NLL chịu trách nhiệm kéo cực đại mật độ xác suất tại đúng nhãn lớp mục tiêu và định hình không gian tiềm ẩn $s(\mathbf{x})$.

#### 2.2.2. Hàm Mất Mát Differentiable Continuous QWK Loss
Quadratic Weighted Kappa đo lường mức độ đồng thuận thứ bậc với ma trận phạt bậc hai:
$$W_{i, j} = \frac{(i - j)^2}{(K - 1)^2}, \quad K = 5$$
Để đạo hàm được qua mạng nơ-ron, ma trận nhầm lẫn khả vi dạng mềm (**Soft Confusion Matrix**) $\mathbf{O}$ được xây dựng từ xác suất dự báo $\mathbf{p}_b \in \mathbb{R}^5$ và vector one-hot nhãn thực tế $\mathbf{t}_b \in \mathbb{R}^5$:
$$O_{i, j} = \sum_{b=1}^B p_{b, i} \cdot t_{b, j}$$
Ma trận kỳ vọng ngẫu nhiên $\mathbf{E}$:
$$E_{i, j} = \frac{\left(\sum_{b=1}^B p_{b, i}\right) \cdot \left(\sum_{b=1}^B t_{b, j}\right)}{B}$$
Hàm mất mát QWK khả vi liên tục:
$$\mathcal{L}_{\text{Continuous\_QWK}} = \frac{\sum_{i=0}^{K-1} \sum_{j=0}^{K-1} W_{i, j} O_{i, j}}{\sum_{i=0}^{K-1} \sum_{j=0}^{K-1} W_{i, j} E_{i, j} + \epsilon}$$

#### 2.2.3. Lịch Trình Tăng Trọng Số Tuyến Tính (Dynamic Warm-up Schedule)
Tính toán ma trận nhầm lẫn mềm trên mini-batch nhỏ ($B=16$ hoặc $48$) có phương sai mẫu lớn. Nếu đưa hàm mất mát QWK vào ngay từ đầu với trọng số lớn ($\lambda = 1.0 - 1.2$), gradient của ma trận bất định sẽ áp đảo NLL, dẫn đến phá vỡ không gian đặc trưng (gây sụp đổ Val QWK từ $0.61$ về $0.47$).
Hệ thống triển khai lịch trình tăng trọng số mềm dẻo theo cấp số bước/epoch:
$$\lambda(t) = \min\left(1.0, \, \frac{t}{T_{\text{warmup}}}\right) \cdot \lambda_{\text{target}}$$
với $T_{\text{warmup}} = 15$ epochs và $\lambda_{\text{target}} = 0.4$. Trong 15 epoch đầu, mạng tập trung tối ưu hóa biên phân loại qua NLL; sau đó, hình phạt khoảng cách thứ bậc bậc hai được đưa vào một cách êm thuận.

---

### 2.3. Cơ Chế Kiểm Soát Hội Tụ & Lưu Checkpoint

Huấn luyện mạng y tế đa miền đối mặt với hiện tượng "nghịch lý tối ưu": `val_loss` thấp nhất không đồng nghĩa với `val_QWK` cao nhất, do `val_loss` nhạy cảm với xác suất tự tin của các mẫu dễ, trong khi QWK phạt nặng các lỗi nhầm lẫn phân độ xa (như nhầm Grade 0 sang Grade 3).

1. **Decoupled Dual Monitoring (Tách biệt hai bộ đếm):**
   - **Lưu Checkpoint Tốt Nhất (`_best.pth`):** Được kích hoạt độc quyền bởi kỷ lục mới của **`Validation QWK`** (`val_qwk > best_val_qwk`). Khi đạt kỷ lục, bộ đếm `epochs_no_improve_es` của Early Stopping được reset về 0.
   - **Điều chỉnh Tốc Độ Học (`ReduceLROnPlateau`):** Được theo dõi độc lập trên tín hiệu êm hơn là **`Validation Loss`** (`val_loss < best_val_loss - 1e-4`) với bộ đếm riêng `epochs_no_improve_lr`. Nếu `val_loss` không giảm sau 4 epochs, tốc độ học giảm đi $50\%$ (`factor = 0.5`) cho đến `MIN_LR = 1e-6`.
2. **Early Stopping:** Kiểm tra dừng sớm nếu `val_qwk` không cải thiện trong 15 epoch liên tiếp (`epochs_no_improve_es >= 15`).
3. **Bảo Toàn Phiên Huấn Luyện:** File `_last.pth` được cập nhật liên tục sau mỗi epoch, lưu toàn bộ `model_state_dict`, `optimizer_state_dict`, `scaler_state_dict`, `scheduler_state_dict`, cùng lịch sử huấn luyện, cho phép khôi phục hoàn hảo khi bị ngắt kết nối môi trường điện toán đám mây.

---

## 3. Pipeline Tiền Xử Lý & Trích Xuất Đặc Trưng (Preprocessing & Data Pipeline)

```
                            [Raw High-Resolution Fundus Image]
                                            │
                             crop_fundus_roi (tolerance=7)
                             (Cắt viền đen ngoại biên theo bọt cường độ)
                                            │
                                            ▼
                           [Ảnh Võng Mạc Cắt Gọn (Cropped ROI)]
                                            │
            ┌───────────────────────────────┴───────────────────────────────┐
            ▼                                                               ▼
   4-Step Optical FOV Mask                                       Color Harmonization Stream
   (Tính toán TRỰC TIẾP trên Raw RGB)                             (Chuẩn hóa màu Ben Graham)
   1. Otsu Adaptive Thresholding                                  I_enhanced = 4*I - 4*Gaussian(I) + 128
   2. Morphological Close + Open (25x25)                                    │
   3. Largest Connected Component                                           ▼
   4. Inward Erosion (erode_px = 15)                             [Ảnh Đáy Mắt Tiền Xử Lý]
            │                                                               │
            └───────────────────────────────┬───────────────────────────────┘
                                            ▼
                          compute_valid_anchors (s=96, p=128)
                          Lọc patch có mean(mask_patch) >= 0.5
                                            │
                                            ▼
                   [2-Tier Offline Storage Engine (.npz Pre-Cache)]
                   Lưu: {"img": uint8_array, "anchors": int32_array}
                   Tiết kiệm 158 CPU-hours tính toán lặp lại qua 100 epochs
```

---

### 3.1. Cơ Chế Cắt Viền Võng Mạc (Tight Retinal-Region Crop)

Các ảnh đáy mắt thô từ các thiết bị quang học khác nhau (Zeiss, Canon, Topcon) thường chứa các dải viền đen không mang thông tin với độ dày biến thiên. Hàm `crop_fundus_roi(img_bgr, tolerance=7)` chuẩn hóa kích thước viền:
1. Chuyển đổi ảnh BGR sang thang độ xám: $I_{\text{gray}} = \text{cv2.cvtColor}(I_{\text{bgr}}, \text{COLOR\_BGR2GRAY})$.
2. Tạo mặt nạ nhị phân tách nền đen: $M_{\text{tissue}} = I_{\text{gray}} > \text{tolerance}$.
3. Tìm hộp bao nhỏ nhất (bounding box) chứa toàn bộ điểm ảnh thỏa mãn $M_{\text{tissue}}$:
   $$x_{\min} = \min(\{x \mid M_{y, x} = 1\}), \quad x_{\max} = \max(\{x \mid M_{y, x} = 1\})$$
   $$y_{\min} = \min(\{y \mid M_{y, x} = 1\}), \quad y_{\max} = \max(\{y \mid M_{y, x} = 1\})$$
4. Cắt trích xuất vùng võng mạc hữu dụng $I_{\text{cropped}} = I_{\text{bgr}}[y_{\min}:y_{\max}, \, x_{\min}:x_{\max}]$, bảo toàn tối đa tỷ lệ khung hình và độ sắc nét tổn thương ngoại vi.

---

### 3.2. Phân Đoạn Trường Nhìn Quang Học (Optical FOV Segmentation)

Hàm `compute_fov_mask(img_np, erode_px=15)` tạo mặt nạ nhị phân chính xác vùng nhãn cầu. **Yêu cầu bắt buộc:** Quá trình này phải thực hiện trực tiếp trên ảnh màu gốc (Raw RGB), trước khi thực hiện bước trừ nền Ben Graham. (Nếu tính trên ảnh Ben Graham, nền đen bị biến thành màu xám 128, khiến thuật toán phân ngưỡng sụp đổ hoàn toàn).
1. **Otsu Adaptive Thresholding:** Tự động xác định ngưỡng phân tách bimodal giữa cấu trúc đáy mắt và viền quang học dựa trên cực tiểu hóa phương sai nội lớp.
2. **Morphological Filtering:**
   - Phép đóng (**Morphological Close**) với phần tử cấu trúc elip $25 \times 25$ (`cv2.MORPH_ELLIPSE`) để lấp đầy các khoảng trống do mạch máu tối màu hoặc bóng quang học.
   - Phép mở (**Morphological Open**) $25 \times 25$ để dọn sạch các đốm nhiễu bên ngoài.
3. **Largest Connected Component:** Dùng `cv2.connectedComponentsWithStats` chỉ giữ lại thành phần liên thông lớn nhất (đĩa võng mạc tròn), loại bỏ toàn bộ ký tự hiệu chuẩn máy, vệt sáng chói phản xạ thấu kính và bụi cảm biến.
4. **Inward Morphological Erosion:** Co mặt nạ vào phía trong 15 pixels (`erode_px = 15`) để loại bỏ hoàn toàn hiện tượng tối góc viền ảnh (vignetting halos) do quang sai thấu kính.

**Điều kiện lọc Patch:** Một patch ứng viên tại tọa độ $[y_0:y_0+p, x_0:x_0+p]$ được chấp nhận đưa vào túi MIL khi và chỉ khi:
$$\frac{1}{p^2} \sum_{u=0}^{p-1} \sum_{v=0}^{p-1} M_{\text{FOV}}(y_0 + u, x_0 + v) \ge 0.5$$

---

### 3.3. Chuẩn Hóa Màu Sắc & Bộ Nhớ Đệm Tối Ưu Hóa I/O (Offline Cache Engine)

#### 3.3.1. Chuẩn Hóa Màu Ben Graham (Ben Graham Preprocessing)
Khắc phục sự khác biệt về cường độ chiếu sáng và sắc tố võng mạc giữa các chủng tộc:
$$I_{\text{enhanced}} = 4 \cdot I - 4 \cdot \text{GaussianBlur}\left(I; \sigma = \frac{W}{30}\right) + 128$$
Trong đó $\sigma = W / 30$ tạo ra ước lượng phông nền mờ cục bộ, phép trừ giúp nổi bật sắc nét hệ mạch máu và các xuất huyết dạng chấm.

#### 3.3.2. Động Lực Học Tối Ưu I/O & Cấu Trúc Cache 2-Tier (.npz)
* **Điểm nghẽn tính toán trực tiếp (On-the-fly bottleneck):**
  - Lọc song phương + Khử nền Gaussian: $354\text{ ms/ảnh}$.
  - Phân đoạn 4 bước FOV + Tìm lưới patch hợp lệ: $265\text{ ms/ảnh}$.
  - Tổng cộng $\approx 810\text{ ms/ảnh}$. Với $7{,}000$ ảnh $\times 100$ epoch, chi phí tính toán lặp lại lãng phí lên tới **$158\text{ CPU-hours}$**, biến CPU thành nút cổ chai nghẽn GPU ($0\%$ GPU utilization).
* **Kiến Trúc Cache 2-Tier Tối Ưu:**
  - Định dạng: Tệp nén NumPy archive (`.npz`) lưu tại `cache/{dataset_key}/{image_id}.npz`.
  - Nội dung lưu trữ:
    1. `img`: Mảng ảnh `uint8` đã qua tiền xử lý Ben Graham / cắt viền.
    2. `anchors`: Mảng $(N, 2)$ kiểu `int32` chứa tọa độ góc trên-trái $[y_0, x_0]$ của các patch thỏa mãn điều kiện FOV.
  - *Tối ưu hóa dung lượng đĩa:* Chỉ lưu tọa độ 2D thay vì lưu toàn bộ các mảng patch cắt sẵn (giảm từ $>50\text{ MB/mẫu}$ xuống chỉ còn $\approx 50 - 100\text{ KB/mẫu}$).
  - *Truy xuất siêu tốc:* Khi tải batch, việc đọc `.npz` chỉ mất $20\text{ ms}$, và thao tác cắt mảng in-memory trong `collate_fn` mất $<1\text{ ms}$, tăng thông lượng DataLoader lên **gấp $>25$ lần**.
* **Tối ưu truyền dữ liệu IPC (Inter-Process Communication):**
  - DataLoader worker truyền dữ liệu dạng tensor `uint8` thay vì `float32`, giảm băng thông bộ nhớ chia sẻ từ $755\text{ MB}$ xuống $189\text{ MB/batch}$ ($4\times$ reduction).
  - Phép chuẩn hóa ImageNet (`div(255).sub(mean).div(std)`) được chuyển dịch toàn bộ lên xử lý song song trên Tensor Cores của GPU thông qua hàm `normalize_fundus_tensor`.
  - Kích hoạt `cv2.setNumThreads(0)` để triệt tiêu hiện tượng tranh chấp tài nguyên luồng giữa OpenCV và PyTorch workers.

---

## 4. Thiết Lập Kiểm Nghiệm & Đối Chứng (Experimental Protocols)

### 4.1. Thực Nghiệm Đối Chứng 5 Cấu Hình Đầu Ra (Head/Loss Ablation Setup)

Để chứng minh tính ưu việt của Cumulative Link Model (CLM) có cơ chế phạt khoảng cách QWK, thực nghiệm đối chứng có kiểm soát được triển khai trên cùng một kiến trúc trích xuất đặc trưng ResNet-50 (ảnh $512 \times 512$, $B=48$, cùng tập dữ liệu chuẩn hóa, 25 epochs):

| STT | Tên Cấu Hình | Kiến Trúc Đầu Ra (Head) | Hàm Mất Mát (Loss Formulation) | Bản Chất Thứ Bậc & Ràng Buộc |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Softmax_CE** | Categorical Logits: $\text{Linear}(512 \to 5)$ | Standard Cross-Entropy: $-\sum t_k \log p_k$ | Phi thứ bậc (Nominal). Phạt nhầm lẫn giữa G0-G1 bằng G0-G4. |
| **2** | **Softmax_QWK** | Categorical Logits: $\text{Linear}(512 \to 5)$ | $\text{CE} + \lambda(t) \cdot \mathcal{L}_{\text{Continuous\_QWK}}$ | Có phạt khoảng cách nhưng logits rời rạc không đảm bảo thứ tự. |
| **3** | **CORAL** | Shared weight $\mathbf{w}$, 4 biases $\{b_k\}$: $\text{Linear}(512 \to 1)$ | Ordinal Binary CE trên 4 ngưỡng độc lập | $P(y > k) = \sigma(\mathbf{w}^T \mathbf{z} + b_k)$. Xảy ra vi phạm thứ bậc nếu $b_k$ bị cắt chéo. |
| **4** | **CORN** | 4 bộ phân loại nhị phân có điều kiện | Conditional Binary CE: $\sum \text{BCE}(y > k \mid y \ge k)$ | $P(y > k) = \prod_{j=0}^k P(y > j \mid y \ge j)$. Khắc phục vi phạm thứ bậc của CORAL. |
| **5** | **CLM_QWK** *(Proposed)* | Latent Score $s(\mathbf{x}) = \mathbf{w}^T \mathbf{z}$, 4 monotonic cutpoints $\theta_k$ | Ordinal Interval NLL $+ \lambda(t) \cdot \mathcal{L}_{\text{Continuous\_QWK}}$ | **Ưu việt tuyệt đối:** $0.0\%$ vi phạm thứ bậc do $\Delta \theta_k > 0$, hàm liên kết clog-log bất đối xứng. |

---

### 4.2. Giao Thức Leave-One-Dataset-Out (LODO) Trên 6 Cohorts

Đánh giá khả năng khái quát hóa đa miền (**Cross-Domain Generalization**) mà không phụ thuộc vào phân phối ảnh nguồn:
* **6 Cohorts Quốc Tế Độc Lập:**
  1. **APTOS 2019** (Ấn Độ, cảm biến cầm tay/phòng khám, $N=3{,}662$)
  2. **DDR** (Trung Quốc, đa trung tâm, $N=13{,}673$)
  3. **DeepDRiD** (Trung Quốc, 2 góc chụp/mắt, $N=2{,}000$)
  4. **IDRiD** (Ấn Độ, độ phân giải cao, $N=516$)
  5. **Messidor-2** (Pháp, camera Topcon $45^\circ$, $N=1{,}748$)
  6. **EyePACS** (Hoa Kỳ, khám sàng lọc viễn thông đa cơ sở, $N=33{,}971$)
  *(Tổng cộng $N = 55{,}570$ ảnh đáy mắt từ $35{,}474$ bệnh nhân).*

```
   ┌─────────────────────────────────────────────────────────────────────────────────┐
   │ 6-Fold Leave-One-Dataset-Out (LODO) Protocol                                    │
   ├────────┬─────────────────────────┬──────────────────────────────────────────────┤
   │ Fold 1 │ Test (OOD): APTOS       │ In-Domain Pool: DDR + DeepDRiD + IDRiD + ... │
   │ Fold 2 │ Test (OOD): DDR         │ In-Domain Pool: APTOS + DeepDRiD + IDRiD +...│
   │ Fold 3 │ Test (OOD): DeepDRiD    │ In-Domain Pool: APTOS + DDR + IDRiD + ...    │
   │ Fold 4 │ Test (OOD): IDRiD       │ In-Domain Pool: APTOS + DDR + DeepDRiD + ... │
   │ Fold 5 │ Test (OOD): Messidor-2  │ In-Domain Pool: APTOS + DDR + DeepDRiD + ... │
   │ Fold 6 │ Test (OOD): EyePACS     │ In-Domain Pool: APTOS + DDR + DeepDRiD + ... │
   └────────┴─────────────────────────┴──────────────────────────────────────────────┘
```

#### Cơ Chế Tránh Rò Rỉ Dữ Liệu Tuyệt Đối (Zero Data Leakage Mechanism)
1. **Namespace Hóa Mã Bệnh Nhân:** Nhận dạng bệnh nhân được gắn tiền tố mã nguồn: `patient_id = f"{dataset_key}::{raw_patient_id}"`. Điều này ngăn chặn việc trùng khớp ngẫu nhiên mã số bệnh nhân giữa các bệnh viện khác nhau.
2. **Bảo Tồn Liên Kết Hai Mắt (Zero Cross-Eye Contamination):** Tất cả các ảnh mắt trái và mắt phải của cùng một cá nhân đều chung một `patient_id`. Phân chia tập dữ liệu được thực thi bằng thuật toán `GroupShuffleSplit(train_size=0.85, groups=df['patient_id'])`. Toàn bộ ảnh của một bệnh nhân chỉ được nằm trong Train HOẶC Validation HOẶC Test.
3. **Bảo Tồn Phân Phối Tự Nhiên Ở Tập Đánh Giá:**
   - Tập Validation in-domain ($15\%$) và tập Test ngoại miền ($100\%$ cohort bị loại) **hoàn toàn giữ nguyên tỷ lệ lưu hành bệnh tự nhiên** (không thực hiện quota balancing), phản ánh trung thực điều kiện triển khai sàng lọc lâm sàng thực tế.
   - Tập huấn luyện Train ($85\%$) được cân bằng định mức nguồn (**Multi-Source Quota Balancing**) để duy trì $1{,}500 - 2{,}500$ ảnh cho mỗi phân độ bệnh (từ Grade 0 đến Grade 4), loại trừ hiện tượng mô hình thiên vị lớp chiếm đa số (Majority Class Bias).

---

### 4.3. Hệ Thống Độ Đo Đánh Giá Lâm Sàng (Clinical Evaluation Metrics)

Hệ thống đánh giá toàn diện gồm 4 trục độ đo độc lập:

1. **Quadratic Weighted Kappa (QWK):**
   $$\kappa = 1 - \frac{\sum_{i=0}^4 \sum_{j=0}^4 W_{i, j} O_{i, j}}{\sum_{i=0}^4 \sum_{j=0}^4 W_{i, j} E_{i, j}}$$
   Độ đo chính phản ánh mức độ nhất quán chẩn đoán giữa mô hình và hội đồng bác sĩ chuyên khoa mắt quốc tế, phạt theo bình phương khoảng cách phân độ lệch.
2. **Per-Grade Recall / Sensitivity ($R_0$ đến $R_4$):**
   $$R_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}, \quad k \in \{0, 1, 2, 3, 4\}$$
   Đặc biệt giám sát độ nhạy ở **Grade 1 ($R_1$, kiểm chứng giả thuyết $H_1$ về MIL cục bộ)** và **Grade 3 ($R_3$, kiểm chứng giả thuyết $H_2$ về tương tác 4 góc phần tư)**.
3. **Within-1 Grade Accuracy ($\text{Acc}_{\pm 1}$):**
   $$\text{Acc}_{\pm 1} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(|\hat{y}_i - y_i| \le 1\right)$$
   Đo lường tỷ lệ các chẩn đoán nằm trong phạm vi sai số lâm sàng cho phép ($\pm 1$ phân độ).
4. **Referable DR AUROC & Hiệu Chuẩn Ngưỡng Sàng Lọc Hậu Nghiệm (Post-Hoc Calibration):**
   - Phân loại bệnh cần chuyển tuyến chuyên khoa (**Referable DR: Grade $\ge 2$** vs Non-Referable: Grade $0, 1$).
   - Xác suất chuyển tuyến liên tục: $P(Y \ge 2) = \sum_{k=2}^4 P(Y = k)$.
   - Tách rời quyết định sàng lọc nhị phân khỏi phép lấy nhãn danh nghĩa `argmax`: ROC curve được xây dựng trên tập Validation để xác định ngưỡng vận hành $\tau_s$ tương ứng với các mức độ đặc hiệu lâm sàng cố định: $95\%$, $90\%$, $85\%$, và $80\%$.
   - Các ngưỡng $\tau_s$ sau đó được đóng băng và đánh giá trên tập Test độc lập nhằm kiểm tra sự tuân thủ tiêu chuẩn sàng lọc quốc gia Vương Quốc Anh (**UK National Screening Committee - UK NSC**: $\text{Sensitivity} \ge 80\%$ tại $\text{Specificity} \ge 95\%$).
5. **Tỷ Lệ Vi Phạm Thứ Bậc (Rank Monotonicity Violations):**
   $$\text{Violations} = \frac{1}{N \cdot (K-2)} \sum_{i=1}^N \sum_{k=0}^{K-3} \mathbb{I}\left(P(Y > k) < P(Y > k + 1)\right)$$
   Đo lường trực tiếp trên logits tích lũy trước khi kích hoạt phi tuyến để kiểm tra tính nhất quán logic của mô hình phân thứ bậc.

---

## 5. Bảng Đối Chiếu Kết Quả Thực Nghiệm Cốt Lõi (Empirical Benchmark Synthesis)

### 5.1. So Sánh 5 Cấu Hình Đầu Ra (Head Ablation on ResNet-50)

| Cấu Hình Đầu Ra | Val QWK | Test QWK | Acc (%) | Within-1 (%) | Ref-AUC | Recall G1 (%) | Recall G3 (%) | Monotonicity Violations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Softmax_CE** | 0.7421 | 0.7185 | 68.4 | 91.2 | 0.8842 | 38.5 | 42.1 | $14.2\%$ |
| **Softmax_QWK** | 0.7712 | 0.7490 | 69.1 | 93.4 | 0.8910 | 41.2 | 45.3 | $8.7\%$ |
| **CORAL** | 0.7680 | 0.7435 | 70.2 | 94.1 | 0.8935 | 44.0 | 48.6 | $3.4\%$ |
| **CORN** | 0.7805 | 0.7582 | 71.0 | 94.8 | 0.8988 | 46.5 | 51.2 | $0.0\%$ |
| **CLM_QWK (Proposed)**| **0.8124**| **0.7896**| **73.5**| **96.8** | **0.9145**| **53.8**| **59.4**| **0.0% (by construction)**|

*Nhận xét:* CLM_QWK vượt trội toàn diện trên mọi độ đo: vừa triệt tiêu $100\%$ vi phạm thứ bậc vừa cải thiện vượt bậc Recall ở Grade 1 ($+15.3\%$) và Grade 3 ($+17.3\%$) so với đường cơ sở Softmax Cross-Entropy.

---

### 5.2. Kết Quả Thác Đóng Góp Kiến Trúc (Architecture Waterfall V0 $\to$ V3 trên LODO)

| Phiên Bản | Kiến Trúc Mô Hình | Mục Tiêu & Giả Thuyết Lâm Sàng | Mean LODO QWK | $\Delta_{\text{LODO}}$ (In $\to$ OOD) | Recall G1 (OOD) | Recall G3 (OOD) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **V0** | ResNet-50 Global Only + Softmax CE | Đường cơ sở phân loại toàn ảnh truyền thống | 0.5842 | -0.1985 | 22.4% | 28.1% |
| **V1** | + Local Sliding-Window MIL | **Giả thuyết $H_1$:** Bảo toàn vi tổn thương Grade 1 | 0.6418 | -0.1520 | 44.6% *(+22.2%)* | 31.5% |
| **V2** | + NonLocal Context & Quadrant Tokens | **Giả thuyết $H_2$:** Khôi phục quy tắc 4-2-1 cho Grade 3 | 0.6935 | -0.1140 | 46.2% | 52.8% *(+21.3%)* |
| **V3** | **+ Champion CLM_QWK Head** | **Giả thuyết $H_3$:** Ổn định hình học thứ bậc đa miền | **0.7485** | **-0.0712** | **54.2%** | **61.5%** |

*Ý nghĩa khoa học:*
- **$H_1$ được xác nhận:** Nhánh Local Patch MIL với cơ chế 1D Max-Pooling làm tăng vọt độ nhạy Grade 1 ngoại miền từ $22.4\%$ lên $44.6\%$ (tăng hơn gấp đôi).
- **$H_2$ được xác nhận:** Transformer 4 góc phần tư kết hợp NonLocal Block giúp Grade 3 tăng độ nhạy từ $31.5\%$ lên $52.8\%$.
- **$H_3$ được xác nhận:** Đầu ra CLM_QWK thu hẹp khoảng cách suy giảm khái quát hóa đa miền ($\Delta_{\text{LODO}}$) từ $-0.1985$ xuống chỉ còn $-0.0712$, chứng minh việc bảo tồn hình học thứ bậc giúp mạng kháng cự hiệu quả hiện tượng dịch chuyển miền ảnh chụp.

---

## 6. Sẵn Sàng Trích Dẫn & Tích Hợp Bài Báo (Manuscript Section Mapping)

Tài liệu này được ánh xạ trực tiếp vào bản thảo bài báo ([`documents/05_manuscript_draft_sections.md`](file:///d:/Github/deep-learning-for-computer-vision/documents/05_manuscript_draft_sections.md)):
- **Section 3 (Methodology):** Trích dẫn toàn văn Mục 1 (Kiến trúc Dual-Branch, NonLocal, CBAM, Quadrant Tokens, Dual Pooling MIL, và công thức tham số hóa Softplus cho CLM).
- **Section 4 (Experimental Setup):** Trích dẫn Bảng siêu tham số tại Mục 2.1, Lịch trình Warm-up $\lambda(t)$ tại Mục 2.2, và Giao thức LODO / Chống rò rỉ dữ liệu tại Mục 4.2.
- **Section 5 (Results & Discussion):** Trích dẫn các Bảng so sánh thực nghiệm tại Mục 5.1 và 5.2.
- **Appendix / Technical Report:** Trích dẫn Mục 3 (Chi tiết 4 bước FOV, thuật toán nén .npz cache, và kỹ thuật tăng tốc IPC `uint8`).
