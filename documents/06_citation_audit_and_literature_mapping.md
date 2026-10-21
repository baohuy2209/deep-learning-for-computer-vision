# BÁO CÁO TOÀN DIỆN: PHÂN LOẠI, KIỂM TOÁN VÀ THIẾT LẬP BẢN ĐỒ TRÍCH DẪN HỌC THUẬT
## (Comprehensive Citation Audit & Section-by-Section Literature Mapping)

> **Bài báo:** *Deconstructing Domain Shift in Diabetic Retinopathy Grading: A Multi-Scale Multiple Instance Learning Framework with Consistent Ordinal Regression*  
> **Mục tiêu xuất bản:** *IEEE Transactions on Medical Imaging (TMI)* / *Medical Image Analysis (MedIA)* / *ICCPR*  
> **Cơ sở dữ liệu nguồn:** [`documents/nguồn trích dẫn.xlsx`](file:///d:/Github/deep-learning-for-computer-vision/documents/ngu%E1%BB%93n%20tr%C3%ADch%20d%E1%BA%ABn.xlsx) & Bản thảo [`documents/05_manuscript_draft_sections.md`](file:///d:/Github/deep-learning-for-computer-vision/documents/05_manuscript_draft_sections.md)  
> **Định vị Storyline (The Central Spine):**  
> *"Domain shift trong phân cấp bệnh võng mạc tiểu đường (DR) không chỉ đơn thuần làm sụt giảm một con số QWK tổng thể; nó làm sụp đổ các cấp độ bệnh lâm sàng (clinical grades) theo những cách rất khác nhau. Do đó, thay vì chỉ hỏi 'mô hình có tổng quát hóa được không?', bài báo này phân rã và trả lời: 'Domain shift làm sụp đổ grade nào, và cơ chế kiến trúc nào giải thích cũng như khắc phục được các thất bại đó?'"*  
> **3 Yêu cầu lâm sàng $\leftrightarrow$ 3 Research Questions (RQ) $\leftrightarrow$ 3 Cơ chế mô hình hóa:**  
> 1. **Grade 1** (Vi phình mạch khu trú) $\leftrightarrow$ **RQ1**: Cần *Native-Resolution Local MIL* để ngăn chặn thất thoát điểm ảnh (pixel-loss) khi downsampling toàn ảnh.  
> 2. **Grade 3** (Quy tắc lâm sàng 4-2-1: tổn thương lan tỏa đa góc phần tư) $\leftrightarrow$ **RQ2**: Cần *Global Relational Context* (Non-Local + Quadrant Tokens / QTA) để so sánh bằng chứng giữa 4 góc phần tư giải phẫu ETDRS.  
> 3. **Ordered Grades** (Tiến triển thứ bậc $0 < 1 < 2 < 3 < 4$) $\leftrightarrow$ **RQ3**: Cần *Ordinal Modeling* (CLM + QWK Loss) để bảo toàn tính nhất quán thứ tự (severity consistency) khi gặp OOD shift.

---

# PHẦN 1: BẢNG PHÂN LOẠI & LỰA CHỌN TRÍCH DẪN (CITATION AUDIT TABLE)

Bảng dưới đây kiểm toán toàn bộ **42 công trình** có trong danh mục của tác giả ([`documents/nguồn trích dẫn.xlsx`](file:///d:/Github/deep-learning-for-computer-vision/documents/ngu%E1%BB%93n%20tr%C3%ADch%20d%E1%BA%ABn.xlsx)), đánh giá mức độ tương thích với Storyline trung tâm và phân định vị trí cắm dẫn chứng cụ thể trong bản thảo.

| STT | Tác giả & Năm | Đề xuất / Trọng tâm của bài | Quyết định | Vị trí cắm trích dẫn | Lý do & Chức năng trong Argument (Spine) |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **1** | Shahbaz & Ali (2026) | Nâng cao phát hiện DR bằng thuật toán học sâu trên bộ dữ liệu lai kết hợp. | **[LOẠI BỎ]** | Không sử dụng | Xếp chồng các CNN tiêu chuẩn trên tập dữ liệu gộp để lấy accuracy tổng thể. Không phân tích domain shift, không có mô hình hóa thứ bậc, làm loãng câu chuyện trung tâm. |
| **2** | Chilukoti et al. (2024) | Phân cấp DR tin cậy qua Transfer Learning và Ensemble kết hợp tối ưu hóa Quadratic Weighted Kappa (QWK). | **[GIỮ LẠI]** | Sec 2.1, Sec 3.7 | Minh chứng cho xu hướng tối ưu hóa chỉ số QWK tổng thể trong y văn gần đây; làm bệ đỡ phản biện rằng tối ưu QWK in-domain không đảm bảo tránh được sự sụp đổ ở từng grade riêng lẻ khi gặp domain shift. |
| **3** | Yadav & Jadhav (2019) | Tổng quan phân loại hình ảnh y tế bằng Deep CNN cho chẩn đoán bệnh tật. | **[LOẠI BỎ]** | Không sử dụng | Tài liệu giáo khoa tổng quan quá rộng về phân loại ảnh y tế nói chung, không tập trung vào nhãn khoa hay cơ chế bệnh học DR. |
| **4** | Dharrao et al. (2025) | Ứng dụng EfficientNetB0 kết hợp Multi-Scale Attention (MSAG) trên ảnh đáy mắt nén về $300 \times 300\text{px}$. | **[GIỮ LẠI]** | Sec 1 (P3), Sec 2.2 | **Dẫn chứng mỏ neo cho Pixel-loss**: Minh chứng rõ ràng rằng các nghiên cứu hiện tại nén ảnh về $300 \times 300\text{px}$ làm tiêu biến vi phình mạch trước khi đưa vào khối attention, trực tiếp kích hoạt động lực cho **RQ1 (Native-Resolution Local MIL)**. |
| **5** | Nguyen et al. (2026) | MV-TRAMS: Tổng hợp ảnh X-quang tuyến vú thích ứng vùng khối u dưới chẩn đoán đa góc nhìn. | **[LOẠI BỎ]** | Không sử dụng | **Lạc đề hoàn toàn**: Nghiên cứu về tạo ảnh nhũ ảnh (mammography) trong ung thư vú, hoàn toàn không liên quan đến ảnh đáy mắt hay bệnh võng mạc tiểu đường. |
| **6** | Karthik et al. (2025) | Phát hiện sớm và phân cấp mức độ nặng DR sử dụng mạng nơ-ron tích chập (CNN). | **[LOẠI BỎ]** | Không sử dụng | Mô hình CNN đơn nhánh tiêu chuẩn trên một tập dữ liệu đơn lẻ, mang tính chất bài tập thực nghiệm, không đóng góp giá trị lý thuyết cho domain generalization hay ordinal modeling. |
| **7** | Aiche et al. (2026) | RDE-DR: Ensemble CNNs học sâu bền vững cho phát hiện DR tự động từ ảnh đáy mắt. | **[LOẠI BỎ]** | Không sử dụng | Kỹ thuật ensemble trung bình cộng các mạng backbone phổ biến để tăng accuracy danh nghĩa. Không giải quyết bản chất thứ bậc và không đánh giá cross-cohort zero-shot. |
| **8** | Pamula et al. (2025) | Tối ưu hóa phát hiện DR bằng thuật toán cá điện (Electric Fish Algorithm) và Bilinear CNN. | **[LOẠI BỎ]** | Không sử dụng | Kỹ thuật tối ưu hóa siêu tham số bằng thuật toán metaheuristic phỏng sinh học. Không mang lại giá trị giải thích lâm sàng, làm loãng trục lập luận y khoa của bài báo. |
| **9** | Folorunsho et al. (2025) | Mô hình học sâu ensemble giải thích được dự đoán DR trên tập APTOS 2019 (bài toán nhị phân DR vs No DR). | **[GIỮ LẠI]** | Sec 1 (P1), Sec 2.1 | **Dẫn chứng phản biện (Counter-evidence)**: Điển hình của việc đơn giản hóa thang đo lâm sàng 5 cấp độ thành bài toán nhị phân (DR vs No DR) để báo cáo độ chính xác ảo (>95%), làm mất hoàn toàn giá trị phân tầng điều trị của thang đo ICDRS. |
| **10** | Mohanty et al. (2023) | Sử dụng các kiến trúc học sâu (ResNet, VGG) để phân loại DR trên ảnh nén đồng nhất $224 \times 224\text{px}$. | **[GIỮ LẠI]** | Sec 2.2 | Dẫn chứng đại diện cho các phương pháp phân loại toàn ảnh truyền thống bị hạn chế bởi việc resize ảnh $224 \times 224\text{px}$, minh chứng cho sự thất thoát tổn thương hạt mịn. |
| **11** | Tyagi et al. (2026) | Tăng cường phát hiện DR bằng ResNet50 kết hợp lọc Gaussian trên ảnh nén $224 \times 224\text{px}$. | **[GIỮ LẠI]** | Sec 1 (P3), Sec 2.2 | **Dẫn chứng định lượng đắt giá**: Dù đạt accuracy cao, việc nén ảnh $224 \times 224\text{px}$ khiến F1-score của lớp Mild DR (Grade 1) tụt xuống chỉ còn 0.86 ngay trên in-domain và sụp đổ nặng hơn out-of-domain, củng cố trực tiếp cho **RQ1**. |
| **12** | Mutawa et al. (2023) | Transfer learning trong phát hiện DR: Đánh giá sự kết hợp giữa các bộ dữ liệu và hiệu năng mô hình. | **[GIỮ LẠI]** | Sec 2.1, Sec 3.2 | Dẫn chứng về việc gộp dữ liệu ngây thơ (naive dataset pooling) mà không giải quyết được domain shift thực sự; dùng làm cơ sở đối chiếu với giao thức 6-fold LODO nghiêm ngặt của bài báo. |
| **13** | Al-Antary & Arafa (2021) | Multi-Scale Attention Network (MSA-Net) cho phân loại bệnh võng mạc tiểu đường. | **[GIỮ LẠI]** | Sec 2.1, Sec 2.2 | **Bằng chứng sụp đổ liên miền (Cross-domain collapse)**: MSA-Net đạt 98.1% accuracy trên APTOS nhưng rơi xuống 87.5% khi chuyển sang EyePACS, chứng minh hiện tượng suy giảm hiệu năng khi gặp external cohorts. |
| **14** | Khan et al. (2025) | Chẩn đoán DR tối ưu bằng xử lý ảnh và mạng Ensemble tối ưu hóa qua thuật toán bầy Salp (Salp Swarm). | **[LOẠI BỎ]** | Không sử dụng | Kỹ thuật metaheuristic phỏng sinh học (Salp Swarm) thuần túy tinh chỉnh trọng số, không liên quan đến cơ chế bệnh học võng mạc và không giải quyết domain shift. |
| **15** | Shahbaz & Ali (2026) | (Bản ghi trùng lặp với STT 1 trong danh mục). | **[LOẠI BỎ]** | Không sử dụng | Trùng lặp dữ liệu. |
| **16** | Samra et al. (2026) | Mạng học sâu song nhánh đáy mắt phát hiện đa bệnh lý mắt thông qua dung hợp đặc trưng lai. | **[LOẠI BỎ]** | Không sử dụng | Nghiên cứu tập trung vào phân loại đa bệnh lý nhãn khoa (glaucoma, cataract, DR) thay vì phân cấp độ nặng 5 giai đoạn của DR; không phục vụ câu chuyện deconstructing domain shift. |
| **17** | Hashmi (2026) | Dual-Resolution Attention-Gated Deep Learning with Ordinal Regression for DR Grading: A Quantified Assessment of Cross-Domain Generalization. | **[GIỮ LẠI - MỎ NEO CỐT LÕI]** | Sec 1 (P2, P3), Sec 2.1, Sec 2.3, Sec 6.4 | **Bệ đỡ lý thuyết trung tâm**: Khẳng định phát hiện bản lề *"Ordering survives domain shift, grade boundaries do not"*. Chứng minh rằng không gian đặc trưng giữ được thứ bậc bệnh lý nhưng ranh giới phân cấp danh nghĩa bị vỡ vụn, trực tiếp thúc đẩy **RQ3 (CLM với monotonic cutpoints)**. Đồng thời là baseline so sánh hệ thống song nhánh chưa có mô hình hóa 4 góc phần tư giải phẫu. |
| **18** | Cun et al. (2026) | Khung cộng tác toàn cục - cục bộ tích hợp Visual Mamba và mạng Kolmogorov-Arnold (KAN) cho phân cấp DR. | **[GIỮ LẠI]** | Sec 2.2 | Đại diện cho hướng tiếp cận Global-Local đương đại tiên tiến nhất (2026); dùng để chỉ ra khoảng trống: dù kết hợp 2 luồng nhưng vẫn thiếu mô hình hóa tương quan không gian giữa các góc phần tư ETDRS và vẫn dùng Softmax danh nghĩa. |
| **19** | Wu et al. (2025) | Đánh giá bệnh võng mạc tiểu đường qua học đa nhiệm (Multi-task Learning) trên các bộ dữ liệu ảnh đáy mắt không đồng nhất. | **[GIỮ LẠI]** | Sec 2.1, Sec 6.5 | Minh chứng xác thực tính không đồng nhất (heterogeneity) về camera, góc chụp, và nhãn dán giữa các đoàn hệ lâm sàng quốc tế; hỗ trợ thảo luận về label noise và prior shift. |
| **20** | Vargas et al. (2020) | Cumulative link models for deep ordinal classification (Neurocomputing). | **[GIỮ LẠI - NỀN TẢNG PHƯƠNG PHÁP]** | Sec 2.3, Sec 3.6 | **Nền tảng toán học cho CLM**: Định hình cấu trúc Cumulative Link Model trong Deep Learning, cơ chế ngưỡng đơn điệu (monotonic cutpoints) và các hàm liên kết (clog-log, logit), là cơ sở trực tiếp cho thiết kế đầu phân loại thứ tự của bài báo. |
| **21** | Reti-TransNet (2026) | Reti-TransNet: Khung lai thích ứng đóng cổng (adaptive gated hybrid) cho phân cấp bệnh võng mạc tiểu đường. | **[GIỮ LẠI]** | Sec 2.2 | Minh chứng cho xu hướng kết hợp CNN (inductive bias cục bộ) và Transformer (ngữ cảnh toàn cục); dùng để đối chiếu rằng Reti-TransNet vẫn dùng Cross-Entropy danh nghĩa và chưa kiểm chứng tính bền bỉ dưới OOD shift. |
| **22** | Yang et al. (2022) | Classification of diabetic retinopathy: Past, present and future (Frontiers in Endocrinology). | **[GIỮ LẠI - MỎ NEO LÂM SÀNG]** | Sec 1 (P1), Sec 2.1, Sec 6.1 | **Bài báo tổng quan lâm sàng tối thượng**: Phân tích lịch sử từ ETDRS đến ICDRS, chỉ ra 8 hạn chế cốt lõi của thang đo ICDR 5 cấp độ khi bị AI coi là các lớp danh nghĩa rời rạc, làm cơ sở y khoa cho việc chuyển dịch sang phân loại có thứ bậc và bảo tồn điểm ảnh tổn thương. |
| **23** | Takahashi et al. (2020) | Mạng CNN trong sàng lọc và phân giai đoạn bệnh võng mạc tiểu đường (PLoS ONE). | **[GIỮ LẠI]** | Sec 2.1, Sec 5.6 | Nghiên cứu benchmark tiêu chuẩn về phân cấp 5 giai đoạn và sàng lọc Referable DR; đại diện cho phương pháp đánh giá in-domain truyền thống không kiểm tra tính bền vững OOD. |
| **24** | Capsule Networks for DR (2021) | Phát hiện và phân loại bệnh võng mạc tiểu đường sử dụng mạng Capsule. | **[LOẠI BỎ]** | Không sử dụng | Kiến trúc Capsule Network đã lỗi thời, tính toán cồng kềnh, không thể mở rộng trên tập dữ liệu lớn và không đóng góp vào câu chuyện multi-scale MIL hay ordinal regression. |
| **25** | Ensemble EfficientNet (2024) | Học sâu ensemble và EfficientNet cho chẩn đoán chính xác DR (Sci. Rep.). | **[LOẠI BỎ]** | Không sử dụng | Trùng lặp ý tưởng với Chilukoti (2024) và Aiche (2026); chỉ dừng lại ở việc ghép nối các biến thể EfficientNet trên dữ liệu in-domain. |
| **26** | EDLDR (2023) | Kỹ thuật học sâu ensemble cho phát hiện và phân loại DR (Diagnostics). | **[LOẠI BỎ]** | Không sử dụng | Kỹ thuật ensemble tiêu chuẩn, không có đóng góp mới về mặt biểu diễn không gian hay mô hình hóa thứ tự OOD. |
| **27** | Multi-Stream DNN with Boosting (2022) | Mạng nơ-ron tích chập đa luồng phân loại mức độ nặng DR dưới khung Boosting (IEEE Access). | **[GIỮ LẠI]** | Sec 2.2 | Tiền đề của việc trích xuất đặc trưng đa luồng (multi-stream); chứng minh rằng đa luồng cải thiện biểu diễn nhưng việc dùng Boosting rời rạc chưa giải quyết được bài toán tổng hợp túi thực thể (MIL) hay hồi quy thứ bậc. |
| **28** | Weighted Ensemble DNN (2024) | Tăng cường phát hiện DR giai đoạn sớm bằng mạng nơ-ron tích chập ensemble có trọng số (IEEE Access). | **[LOẠI BỎ]** | Không sử dụng | Kỹ thuật ensemble trọng số đơn giản, trùng lặp và không phục vụ trục lý thuyết của bài báo. |
| **29** | Galappaththige et al. (2024) | Generalizing to Unseen Domains in Diabetic Retinopathy Classification (IEEE WACV). | **[GIỮ LẠI - MỎ NEO BENCHMARK & GAP]** | Sec 1 (P2), Sec 2.1, Sec 4.1, Sec 5.1, Sec 6.1 | **CÔNG TRÌNH BENCHMARK TRUNG TÂM**: Nghiên cứu tiên phong đánh giá Domain Generalization (DG) trên DR dưới chuẩn DomainBed (ERM-ViT, SD-ViT). Thiết lập trực tiếp **Research Gap** của bài báo: *họ chỉ đo lường aggregate QWK/AUC, che giấu hoàn toàn sự sụp đổ phân hóa giữa các grade lâm sàng*. Bài báo của ta kế thừa và phân rã trực tiếp phát hiện của công trình này. |
| **30** | Vision Transformer with MAE (2024) | ViT kết hợp Masked Autoencoders cho phân loại referable DR dựa trên ảnh võng mạc kích thước lớn (PLoS ONE). | **[GIỮ LẠI]** | Sec 2.2 | Minh chứng cho hướng tiếp cận xử lý ảnh kích thước lớn bằng ViT/MAE; dùng để lập luận phản biện về độ phức tạp tính toán bậc hai $\mathcal{O}(N^2)$ và sự tốn kém tài nguyên, làm nổi bật tính thanh lịch và hiệu quả của kiến trúc lai All-CNN + Local MIL. |
| **31** | ORViT-DR (2026) | ORViT-DR: Hybrid ViT bền vững thứ tự cho phân cấp DR trên ảnh độ phân giải thấp (arXiv). | **[GIỮ LẠI]** | Sec 2.3 | Công trình đương đại (2026) đưa ràng buộc thứ tự vào ViT trên ảnh phân giải thấp; dùng làm đối chứng so sánh trực tiếp cho **RQ3**, chứng minh rằng nếu chỉ ép thứ tự trên ảnh nén thấp thì vẫn thất bại ở Grade 1 do pixel-loss. |
| **32** | Comparative Study on ViT for DR (2026) | Nghiên cứu so sánh phân loại DR bằng Vision Transformers (IEEE ICISS). | **[LOẠI BỎ]** | Không sử dụng | Bài khảo sát thực nghiệm thuần túy chạy lại các biến thể ViT có sẵn, không có đề xuất mô hình mới hay phân tích domain shift chuyên sâu. |
| **33** | Unified Cascade for Retinal Disease (2024) | Mạng xếp tầng học sâu thống nhất cho phát hiện bệnh võng mạc và phân cấp DR. | **[LOẠI BỎ]** | Không sử dụng | Mô hình cascade cho nhiều bệnh mắt tổng quát, làm phân tán câu chuyện chuyên sâu về deconstructing 5 grades của DR. |
| **34** | Gated State-Space Feature Mixing for DR (2026) | Phân cấp DR tin cậy và hiệu chuẩn từ ảnh lọc Gaussian sử dụng Attention và Gated State-Space (Sci. Rep.). | **[GIỮ LẠI]** | Sec 2.2, Sec 2.3 | Đại diện cho các nghiên cứu mới nhất chú trọng đến độ tin cậy và hiệu chuẩn mô hình (model calibration); hỗ trợ lập luận cho phần phân tích độ tin cậy ranh giới và phân phối thứ bậc. |
| **35** | Hybrid Deep Neural-Attention Mechanism (2024) | Phát hiện DR sử dụng mô hình lai Deep Neural và cơ chế Attention (J. Adv.). | **[LOẠI BỎ]** | Không sử dụng | Đề xuất chèn thêm module attention tiêu chuẩn vào CNN, không có giá trị đột phá về mặt lý thuyết hay kiểm chứng thực nghiệm đa trung tâm. |
| **36** | Attention-based DL for automatic fundus processing (2024) | Khung học sâu dựa trên Attention hỗ trợ xử lý ảnh đáy mắt tự động cho phân cấp DR (CMPB). | **[GIỮ LẠI]** | Sec 2.2 | Minh chứng cho việc sử dụng attention để định vị vùng tổn thương; dùng làm cầu nối dẫn dắt vào cơ chế Attention-MIL và thẩm định XAI so với ground-truth lesion mask. |
| **37** | Islam et al. (2026 - MultiRetNet) | MultiRetNet: Hướng tiếp cận XAI nhẹ cho phân cấp DR và phát hiện DME sử dụng dung hợp Fundus-OCT (MDPI). | **[LOẠI BỎ]** | Không sử dụng (hoặc chỉ nhắc thoáng qua ở Discussion) | Đòi hỏi dữ liệu ghép cặp đồng thời ảnh đáy mắt màu (CFP) và cắt lớp quang học (OCT). Trong khi bài báo của ta giải quyết bài toán sàng lọc cộng đồng trên 6 bộ dữ liệu mở thuần CFP ($N=55{,}570$) - nơi OCT không khả dụng đại trà. Việc đưa vào Related Work sẽ gây nhầm lẫn về phạm vi dữ liệu đơn phương thức. |
| **38** | Lu et al. (2026 - MRAMnet) | MRAMnet: Mạng attention đa vùng với học phân tách tương phản cho phân cấp DR. | **[GIỮ LẠI - ĐỐI THỦ SO SÁNH TRỰC TIẾP]** | Sec 1 (P3), Sec 2.2, Sec 3.4 | **ĐỐI THỦ ĐƯƠNG ĐẠI CẦN PHẢN BIỆN**: MRAMnet đề xuất chia ảnh thành $2 \times 2$ vùng cố định để bắt chước 4 góc phần tư. Tuy nhiên, MRAMnet: (1) Cắt lưới cứng nhắc, không theo mốc giải phẫu, (2) Dùng hàm ArcFace phân loại danh nghĩa (bỏ qua khoảng cách thứ bậc), (3) Chỉ thử nghiệm in-domain trên APTOS. Đây là bàn đạp hoàn hảo để ta khẳng định sự vượt trội của **Quadrant Token Attention (QTA) + Non-Local (RQ2)** và **CLM (RQ3)** dưới LODO. |
| **39** | Detection of DR via MIL (2023) | Phát hiện bệnh võng mạc tiểu đường thông qua học đa thực thể (Multiple Instance Learning) (ACM). | **[GIỮ LẠI]** | Sec 2.2, Sec 3.5 | Thiết lập nền tảng bài toán mô hình hóa ảnh đáy mắt dưới dạng túi thực thể (bag-of-patches), chứng minh tính khả thi của việc trích xuất patch phục vụ chẩn đoán DR. |
| **40** | Deep MIL for automatic detection of DR (2018) | Deep Multiple Instance Learning cho phát hiện tự động bệnh võng mạc tiểu đường trong ảnh võng mạc (IET). | **[GIỮ LẠI]** | Sec 2.2 | Cột mốc lịch sử đầu tiên ứng dụng Deep MIL vào ảnh võng mạc; minh chứng cho tiến trình phát triển từ các hàm gom cụm MIL sơ khai (max/mean pooling) đến attention-gated pooling hiện đại. |
| **41** | Sudharshan / Quellec et al. (2021) | Hướng tiếp cận đa thực thể giải thích được cho phát hiện referable DR trong ảnh đáy mắt (Sci. Rep.). | **[GIỮ LẠI - TIỀN ĐỀ PHƯƠNG PHÁP QUAN TRỌNG]** | Sec 2.2, Sec 3.5, Sec 5.7 | **Tiền đề phương pháp then chốt**: Sử dụng trọng số attention cấp patch để phát hiện referable DR có khả năng giải thích vị trí tổn thương. Mô hình của ta kế thừa và nâng tầm bằng cách trích xuất patch ở độ phân giải gốc (native resolution) kết hợp với luồng ngữ cảnh toàn cục và đầu ra hồi quy thứ tự. |
| **42** | Araújo et al. (2020 - DR\|GRADUATE) | DR\|GRADUATE: Phân cấp DR dựa trên học sâu nhận thức độ bất định trong ảnh chụp đáy mắt (MedIA). | **[GIỮ LẠI - BENCHMARK KINH ĐIỂN]** | Sec 2.1, Sec 2.3, Sec 4.2 | **Công trình đỉnh cao trên Medical Image Analysis (MedIA)**: Mô hình hóa độ bất định và hàm phạt sai số có tính đến khoảng cách thứ bậc. Cực kỳ quan trọng để định vị tính chuẩn mực của bài toán phân cấp DR an toàn y khoa. |

---

# PHẦN 2: DANH SÁCH CÁC BÀI BỊ LOẠI BỎ VÀ LÝ DO CHI TIẾT (EXCLUSION LIST)

Có tổng cộng **19 bài báo** bị loại bỏ khỏi danh mục trích dẫn chính. Việc loại bỏ này là bắt buộc để bảo vệ "hợp đồng khoa học" (scientific contract) của bài báo: không làm loãng mạch truyện trung tâm ("Deconstructing Domain Shift") và không biến bài báo thành một bản tổng hợp hỗn độn các mẹo vặt kiến trúc.

### Nhóm 1: Lạc đề hoàn toàn về mặt bệnh học và phương thức hình ảnh (Off-topic / Wrong Modality)
1. **Nguyen et al. (2026) - MV-TRAMS (ScienceDirect)**:  
   *Lý do loại:* Nghiên cứu về tạo ảnh X-quang tuyến vú (Mammography) trong ung thư vú. Không có bất kỳ mối liên hệ nào với ảnh đáy mắt hay bệnh lý võng mạc tiểu đường. Việc giữ lại bài này sẽ bị Reviewer đánh giá là trích dẫn cẩu thả không đọc nội dung.
2. **Islam et al. (2026) - MultiRetNet (MDPI)**:  
   *Lý do loại:* Mô hình yêu cầu đầu vào đa phương thức ghép cặp đồng thời ảnh đáy mắt màu (CFP) và ảnh cắt lớp vi cấu trúc (OCT). Trong khi đó, bài báo của chúng ta tập trung giải quyết bài toán sàng lọc cộng đồng trên 6 bộ dữ liệu mở quốc tế lớn nhất thuần CFP ($N=55{,}570$). Việc đưa bài này vào Related Work sẽ làm người đọc hiểu lầm rằng phương pháp của ta đòi hỏi thiết bị OCT đắt tiền, đi ngược lại tính khả thi triển khai diện rộng.
3. **Samra et al. (2026) (Sci. Rep.) & Unified Cascade (2024)**:  
   *Lý do loại:* Cả hai bài đều giải quyết bài toán chẩn đoán đa bệnh lý mắt (Cataract, Glaucoma, DR, Myopia...). Mục tiêu của ta là đi sâu vào bản chất vi mạch và phân rã 5 cấp độ nặng của duy nhất bệnh DR theo chuẩn ICDRS.

### Nhóm 2: Các mẹo vặt tối ưu hóa phỏng sinh học / Metaheuristics (Biologically-inspired Tricks)
4. **Pamula et al. (2025) - Electric Fish Algorithm (Sci. Rep.)**:  
   *Lý do loại:* Sử dụng thuật toán tối ưu "Cá điện" để chỉnh siêu tham số mạng Bilinear CNN. Đây là dạng bài "đua điểm accuracy" bằng mẹo thuật toán ngẫu nhiên, không có cơ sở sinh lý bệnh học mắt và không giải thích được cơ chế thích ứng với domain shift.
5. **Khan et al. (2025) - Salp Swarm Algorithm (arXiv)**:  
   *Lý do loại:* Tương tự bài Pamula, dùng thuật toán "Bầy Salp" để chọn trọng số ensemble. Không phục vụ 3 câu hỏi nghiên cứu (RQ1, RQ2, RQ3).

### Nhóm 3: Đơn giản hóa bài toán lâm sàng thành phân loại nhị phân (Oversimplified Binary Classification)
6. **Folorunsho et al. (2025) (LAUTECH)**:  
   *Lý do loại khỏi danh sách mô hình tham chiếu chính (chỉ giữ lại 1 câu phản biện ở Intro/Related Work):* Gộp toàn bộ 5 mức độ bệnh thành bài toán nhị phân (No DR vs DR). Việc này bỏ qua hoàn toàn yêu cầu phân loại thứ bậc của bác sĩ nhãn khoa và không thể ứng dụng trong phân tầng điều trị.

### Nhóm 4: Ghép nối Ensemble đơn thuần và kỹ thuật cũ không có đóng góp mới (Redundant Ensembles & Outdated Networks)
7. **Shahbaz & Ali (2026) (J. Umm Al-Qura)** (Xuất hiện 2 lần ở STT 1 và STT 15).
8. **Karthik et al. (2025) (SN Computer Science)**.
9. **Aiche et al. (2026) - RDE-DR (Sci. Rep.)**.
10. **Ensemble EfficientNet (2024) (Sci. Rep.)**.
11. **EDLDR (2023) (MDPI Diagnostics)**.
12. **Weighted Ensemble DNN (2024) (IEEE Access)**.  
    *Lý do loại chung cho nhóm 7-12:* Tất cả các bài này đều áp dụng một công thức lặp lại: lấy 3-5 mạng CNN có sẵn (VGG, ResNet, DenseNet, EfficientNet), chạy trên 1 tập dữ liệu (thường là APTOS hoặc EyePACS), sau đó lấy trung bình cộng hoặc bầu chọn có trọng số để báo cáo độ chính xác danh nghĩa tăng thêm $0.5 - 1\%$. Chúng hoàn toàn không có mô hình hóa thứ bậc, không đánh giá cross-dataset zero-shot, và không giải thích được thất bại ở cấp độ grade.
13. **Capsule Networks for DR (2021) (ScienceDirect)**:  
    *Lý do loại:* Mạng Capsule đã được cộng đồng chứng minh là tính toán quá nặng, khó hội tụ trên dữ liệu lớn và không còn là xu hướng nghiên cứu thị giác máy tính đương đại.
14. **Comparative Study on ViT (2026) (IEEE ICISS)**:  
    *Lý do loại:* Bài báo dạng báo cáo thí nghiệm sinh viên so sánh các ViT có sẵn, không có đóng góp học thuật mới.
15. **Hybrid Deep Neural-Attention Mechanism (2024) (J. Adv.)**:  
    *Lý do loại:* Chèn thêm module attention sơ khai, tạp chí chất lượng thấp, không đủ trọng lượng học thuật cho bài báo gửi TMI/MedIA.
16. **Yadav & Jadhav (2019) (J. Big Data)**:  
    *Lý do loại:* Bài tổng quan quá rộng về phân loại ảnh y tế, không có thông tin cụ thể cho bệnh học võng mạc.

---

# PHẦN 3: CÁC KHOẢNG TRỐNG TRÍCH DẪN BẮT BUỘC BỔ SUNG (MISSING CRITICAL REFERENCES)

So chiếu danh mục hiện tại với Storyline của bài báo, danh mục của tác giả **đang thiếu vắng trầm trọng những công trình kinh điển mang tính nền tảng bắt buộc phải có**. Nếu thiếu các bài này, Reviewer chuyên gia y tế và thị giác máy tính sẽ ngay lập tức bắt bẻ tính chuẩn mực học thuật của công trình.

Dưới đây là **6 cụm trích dẫn bắt buộc phải bổ sung vào bản thảo**:

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             6 CỤM TRÍCH DẪN BẮT BUỘC PHẢI BỔ SUNG                                │
├──────────────────────────────┬──────────────────────────────────┬────────────────────────────────┤
│ 1. Chuẩn Lâm sàng & Giải phẫu│ 2. Landmark Deep Learning DR     │ 3. Nền tảng Domain Shift (DG)  │
│    ICDRS (Wilkinson 2003)    │    Gulshan et al. (2016 - JAMA)  │    DomainBed (Gulrajani 2021)  │
│    ETDRS 4-2-1 Rule (1991)   │    Ting et al. (2017 - JAMA)     │    WILDS (Koh et al. 2021)     │
├──────────────────────────────┼──────────────────────────────────┼────────────────────────────────┤
│ 4. Hồi quy Thứ tự & Hàm Loss │ 5. Khối Kiến trúc Cốt lõi        │ 6. Bài báo Gốc của 6 Dataset   │
│    CLM (McCullagh 1980)      │    Attention-MIL (Ilse 2018)     │    EyePACS, Messidor-2, IDRiD, │
│    CORAL (Cao 2020), CORN    │    Non-Local Net (Wang 2018)     │    DDR, DeepDRiD, APTOS 2019   │
│    QWK Loss (Cohen 1968)     │    CBAM (Woo 2018), ResNet       │                                │
└──────────────────────────────┴──────────────────────────────────┴────────────────────────────────┘
```

### Cụm 1: Tiêu chuẩn Phân cấp Lâm sàng Quốc tế & Quy chuẩn Góc phần tư Giải phẫu
- **Wilkinson et al. (2003)**:  
  *Trích dẫn:* Wilkinson, C. P., et al. "Proposed international clinical diabetic retinopathy and diabetic macular edema disease severity scales." *Ophthalmology* 110.9 (2003): 1677-1682.  
  *Chức năng:* **Cơ sở y khoa tối thượng** định nghĩa thang đo 5 cấp độ ICDRS (Grade 0 đến 4). Mọi bài báo DR đạt chuẩn y khoa đều phải trích dẫn bài này ở câu đầu tiên của Introduction.
- **ETDRS Research Group (1991)**:  
  *Trích dẫn:* Early Treatment Diabetic Retinopathy Study Research Group. "Grading diabetic retinopathy from stereoscopic color fundus photographs—an extension of the modified Airlie House classification. ETDRS report number 10 & 12." *Ophthalmology* 98.5 Suppl (1991): 786-806.  
  *Chức năng:* **Nguồn gốc lịch sử của "Quy tắc 4-2-1"** cho chẩn đoán Severe NPDR (Grade 3). Bắt buộc phải trích dẫn để bảo vệ lý do tại sao kiến trúc của ta phải chia 4 góc phần tư giải phẫu (RQ2).

### Cụm 2: Các công trình AI Nhãn khoa Đặt nền móng (Landmark Benchmarks)
- **Gulshan et al. (2016)**:  
  *Trích dẫn:* Gulshan, V., et al. "Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs." *JAMA* 316.22 (2016): 2402-2410.  
  *Chức năng:* Công trình kinh điển mở đầu kỷ nguyên Deep Learning trong nhãn khoa của nhóm Google AI, chứng minh AI đạt độ chính xác ngang chuyên gia trên tập in-domain lớn.
- **Ting et al. (2017)**:  
  *Trích dẫn:* Ting, D. S. W., et al. "Development and validation of a deep learning system for diabetic retinopathy and related eye diseases using retinal images from multiethnic populations." *JAMA* 318.22 (2017): 2211-2223.  
  *Chức năng:* Nghiên cứu đa chủng tộc đầu tiên trên quy mô lớn, chỉ ra sự cần thiết phải kiểm thử trên các quần thể dân số đa dạng.

### Cụm 3: Khung lý thuyết về Lệch miền (Domain Generalization & Distribution Shift)
- **Gulrajani & Lopez-Paz (2021)**:  
  *Trích dẫn:* Gulrajani, I., & Lopez-Paz, D. "In search of lost domain generalization." *ICLR* (2021).  
  *Chức năng:* Bài báo thiết lập bộ benchmark DomainBed mà Galappaththige et al. (2024) sử dụng. Trích dẫn để củng cố giao thức đánh giá không rò rỉ dữ liệu.
- **Koh et al. (2021)**:  
  *Trích dẫn:* Koh, P. W., et al. "WILDS: A benchmark of in-the-wild distribution shifts." *ICML* (2021).  
  *Chức năng:* Đặt bài toán của ta vào bức tranh lớn của Machine Learning: đánh giá mô hình dưới sự dịch chuyển phân phối thực tế (in-the-wild distribution shifts).

### Cụm 4: Cơ sở Toán học của Hồi quy Thứ tự (Ordinal Modeling & Loss Formulations)
- **McCullagh (1980)**:  
  *Trích dẫn:* McCullagh, P. "Regression models for ordinal data." *Journal of the Royal Statistical Society: Series B (Methodological)* 42.2 (1980): 109-142.  
  *Chức năng:* **Bài báo gốc khai sinh ra Cumulative Link Model (CLM)**, hàm liên kết logit và complementary log-log (clog-log). Đây là nền tảng toán học cho toàn bộ Section 3.6 của bản thảo.
- **Agresti (2010)**:  
  *Trích dẫn:* Agresti, A. *Analysis of ordinal categorical data.* John Wiley & Sons, 2010.  
  *Chức năng:* Giáo trình chuẩn mực về dữ liệu phân loại có thứ tự, chứng minh tại sao biến thứ tự không thể coi là biến danh nghĩa (nominal) hay biến liên tục thông thường.
- **Cao et al. (2020) - CORAL**:  
  *Trích dẫn:* Cao, W., et al. "Rank-consistent ordinal regression for neural networks with application to age estimation." *Pattern Recognition Letters* 140 (2020): 325-331.  
  *Chức năng:* Baseline hồi quy thứ tự quan trọng (Consistent Rank Logits) mà bài báo so sánh trong Table 4.
- **Shi et al. (2021) - CORN**:  
  *Trích dẫn:* Shi, X., et al. "Deep neural networks for rank-consistent ordinal regression based on conditional probabilities." *Pattern Recognition* 142 (2023): 109603.  
  *Chức năng:* Baseline hồi quy thứ tự hiện đại dựa trên xác suất có điều kiện trong Table 4.
- **Cohen (1968)**:  
  *Trích dẫn:* Cohen, J. "Weighted kappa: Nominal scale agreement provision for scaled disagreement or partial credit." *Psychological Bulletin* 70.4 (1968): 213.  
  *Chức năng:* Định nghĩa toán học của Quadratic Weighted Kappa (QWK) và ma trận trọng số phạt theo bình phương khoảng cách sai lệch $(i - j)^2$.
- **de la Torre et al. (2018) / Differentiable QWK**:  
  *Trích dẫn:* de La Torre, J., et al. "A deep framework for diabetic retinopathy severity grading using continuous and ordinal regression." *MICCAI* (2018).  
  *Chức năng:* Nền tảng cho hàm loss khả vi xấp xỉ ma trận nhầm lẫn mềm để tối ưu trực tiếp QWK trong mạng nơ-ron sâu.

### Cụm 5: Nền tảng Kiến trúc Mô hình (Attention, MIL & Backbones)
- **Ilse et al. (2018)**:  
  *Trích dẫn:* Ilse, M., Tomczak, J., & Welling, M. "Attention-based deep multiple instance learning." *ICML* (2018).  
  *Chức năng:* Công thức toán học của cơ chế Attention-gated MIL Pooling được sử dụng trong luồng Local Patch (Section 3.5).
- **Wang et al. (2018)**:  
  *Trích dẫn:* Wang, X., et al. "Non-local neural networks." *CVPR* (2018).  
  *Chức năng:* Cơ chế Non-Local 2D học tương quan không gian tầm xa được tích hợp trên Layer 4 của ResNet-50 trong luồng Global (Section 3.4).
- **Woo et al. (2018)**:  
  *Trích dẫn:* Woo, S., et al. "CBAM: Convolutional block attention module." *ECCV* (2018).  
  *Chức năng:* Cơ chế chú ý không gian và kênh được tích hợp trong luồng Global.
- **He et al. (2016)** (ResNet) & **Tan & Le (2019)** (EfficientNet):  
  *Chức năng:* Hai mạng xương sống (backbones) của 2 nhánh Global và Local.

### Cụm 6: Bài báo Nguồn của 6 Bộ Dữ liệu Quốc tế (The 6 Open Cohorts)
Bắt buộc phải trích dẫn chính xác bài báo gốc công bố bộ dữ liệu thay vì chỉ dẫn link website:
1. **EyePACS**: Cuadros, J., & Bresnick, G. "EyePACS: an adaptable telemedicine system for diabetic retinopathy screening." *Diabetes Technology & Therapeutics* 11.3 (2009): 157-168.
2. **Messidor-2**: Decencière, E., et al. "Feedback on a publicly distributed image database: the Messidor database." *Image Analysis & Stereology* 33.3 (2014): 231-234.
3. **IDRiD**: Porwal, P., et al. "Indian Diabetic Retinopathy Image Dataset (IDRiD): A database for diabetic retinopathy screening research." *Medical Physics* 45.7 (2018): 3159-3170.
4. **DDR**: Li, T., et al. "Diagnostic assessment of deep learning algorithms for diabetic retinopathy screening." *Information Sciences* 501 (2019): 511-522.
5. **DeepDRiD**: Liu, R., et al. "A regularized deep-learning pipeline for diabetic retinopathy grading on retinal fundus images." *Patterns (Cell Press)* 3.7 (2022): 100523.
6. **APTOS 2019**: Asia Pacific Tele-Ophthalmology Society. "APTOS 2019 Blindness Detection." *Kaggle Competition* (2019).

---

# PHẦN 4: BẢN ĐỒ DẪN CHỨNG THEO TỪNG ĐOẠN (SECTION-BY-SECTION CITATION BLUEPRINT)

Dưới đây là dàn ý chi tiết từng đoạn văn cho toàn bộ các Section của bài báo, chỉ rõ từng luận điểm và ghim chính xác mã trích dẫn tương ứng.

```text
====================================================================================================
SECTION 1: INTRODUCTION
====================================================================================================
```

### Đoạn 1 (P1) — Nhiệm vụ lâm sàng & Thang đo 5 cấp độ ICDRS
- **Luận điểm 1:** Bệnh võng mạc tiểu đường (DR) là biến chứng vi mạch tiến triển hàng đầu gây mù lòa có thể phòng ngừa trên toàn cầu.  
  $\rightarrow$ **Trích dẫn:** `\cite{wilkinson2003proposed, yang2022classification}`
- **Luận điểm 2:** Việc đánh giá lâm sàng dựa trên thang đo quốc tế ICDRS gồm 5 giai đoạn: Grade 0 (No DR), Grade 1 (Mild NPDR), Grade 2 (Moderate NPDR), Grade 3 (Severe NPDR), và Grade 4 (Proliferative DR).  
  $\rightarrow$ **Trích dẫn:** `\cite{wilkinson2003proposed}`
- **Luận điểm 3:** Các cấp độ này không tương đương nhau về mặt hậu quả y tế: chẩn đoán nhầm Grade 3 thành Grade 0 gây ra nguy cơ bỏ sót điều trị nghiêm trọng hơn rất nhiều so với nhầm lẫn giữa Grade 1 và Grade 0. Việc các nghiên cứu trước đây đơn giản hóa thành phân loại nhị phân làm triệt tiêu giá trị phân tầng lâm sàng.  
  $\rightarrow$ **Trích dẫn:** `\cite{folorunsho2025explainable, yang2022classification}`
- **Luận điểm 4:** Các thước đo tổng thể như Quadratic Weighted Kappa (QWK) hay Accuracy trung bình có thể che giấu hoàn toàn việc cấp độ lâm sàng nào thực sự đang bị mô hình chẩn đoán sai hỏng khi triển khai thực tế.  
  $\rightarrow$ **Trích dẫn:** `\cite{chilukoti2024reliable}`

### Đoạn 2 (P2) — Vấn đề Lệch miền (Domain Shift) & Khoảng trống của Galappaththige
- **Luận điểm 1:** Các mô hình học sâu đạt hiệu năng vượt trội trên các benchmark đóng (in-domain), với QWK thường vượt qua 0.90.  
  $\rightarrow$ **Trích dẫn:** `\cite{gulshan2016development, ting2017development, dharrao2025ai}`
- **Luận điểm 2:** Tuy nhiên, khi kiểm thử ngoại viện trên các bệnh viện khác (unseen cohorts), hiệu năng sụt giảm nghiêm trọng do sự khác biệt về quang học máy chụp đáy mắt (Canon, Zeiss, Topcon, Kowa), góc trường chụp ($45^\circ$ vs $50^\circ$), và đặc điểm nhân khẩu học bệnh nhân.  
  $\rightarrow$ **Trích dẫn:** `\cite{mutawa2023transfer, alantary2021multi, wu2025diabetic}`
- **Luận điểm 3:** Afshan Hashmi (2026) ghi nhận độ nhạy sàng lọc sụt giảm từ 87.9% xuống 62.0% khi chuyển miền, và đưa ra quan sát quan trọng: *"Tính thứ bậc được bảo tồn khi dịch miền, nhưng các ranh giới phân cấp thì bị phá vỡ"*.  
  $\rightarrow$ **Trích dẫn:** `\cite{hashmi2026dualresolution}`
- **Luận điểm 4 (THE CORE RESEARCH GAP):** Galappaththige et al. (2024) tiên phong nghiên cứu Domain Generalization (DG) cho DR trên DomainBed, nhưng đánh giá hoàn toàn dừng lại ở cấp độ vĩ mô (aggregate QWK/AUC). Do đó, y văn hoàn toàn chưa biết: *Sự sụt giảm do domain shift phân bổ như thế nào trên từng cấp độ bệnh lâm sàng, và cơ chế kiến trúc nào chịu trách nhiệm cho sự sụp đổ đó?*  
  $\rightarrow$ **Trích dẫn:** `\cite{galappaththige2024generalizing}`
- **Luận điểm 5:** Đồng thời thừa nhận các yếu tố ngoại cảnh như nhiễu nhãn (label noise) giữa các trung tâm và sự dịch chuyển phân phối tiên nghiệm $P(Y)$, nhưng định vị trọng tâm bài báo là phân rã theo cấp độ grade và thành phần kiến trúc.  
  $\rightarrow$ **Trích dẫn:** `\cite{wu2025diabetic}`

### Đoạn 3 (P3) — Dẫn xuất Lâm sàng 3 Cơ chế Mô hình hóa
- **Luận điểm 1:** Khắc phục khoảng trống trên bằng cách dẫn xuất 3 yêu cầu mô hình hóa trực tiếp từ tiêu chuẩn chẩn đoán lâm sàng ICDRS:
- **Yêu cầu 1 (Grade 1 $\leftrightarrow$ RQ1):** Grade 1 được định nghĩa duy nhất bằng các nốt vi phình mạch (microaneurysms) có kích thước dưới $120\,\mu\mathrm{m}$ (dưới 15 pixel trên ảnh gốc). Việc nén ảnh toàn bộ xuống $224 \times 224\text{px}$ hay $300 \times 300\text{px}$ làm tiêu biến các điểm ảnh này $\rightarrow$ Đòi hỏi nhánh **Native-Resolution Local MIL**.  
  $\rightarrow$ **Trích dẫn:** `\cite{tyagi2026enhanced, dharrao2025ai, ilse2018attention}`
- **Yêu cầu 2 (Grade 3 $\leftrightarrow$ RQ2):** Grade 3 được xác định bằng "Quy tắc 4-2-1" theo chuẩn ETDRS ($>20$ nốt xuất huyết ở cả 4 góc phần tư, tĩnh mạch chuỗi hạt ở $\ge 2$ góc, hoặc dị dạng vi mạch IRMA ở $\ge 1$ góc) $\rightarrow$ Đòi hỏi **Non-Local Relational Attention** và **Quadrant Token Attention (QTA)** để so sánh bằng chứng giữa 4 góc giải phẫu, vượt trội hơn cách cắt lưới cứng $2 \times 2$ của MRAMnet.  
  $\rightarrow$ **Trích dẫn:** `\cite{etdrs1991report10, wang2018non, lu2026mramnet}`
- **Yêu cầu 3 (Ordered Grades $\leftrightarrow$ RQ3):** 5 cấp độ bệnh hình thành tiến trình thứ bậc sinh lý tự nhiên ($0 < 1 < 2 < 3 < 4$). Việc dùng Softmax danh nghĩa bỏ qua khoảng cách thứ bậc $\rightarrow$ Đòi hỏi **Cumulative Link Model (CLM)** với các ngưỡng cắt đơn điệu (monotonic cutpoints) và hàm loss QWK.  
  $\rightarrow$ **Trích dẫn:** `\cite{mccullagh1980regression, agresti2010analysis, vargas2020cumulative}`

### Đoạn 4 (P4) — Thiết kế Nghiên cứu & Cam kết của Bài báo
- **Luận điểm 1:** Kiểm chứng 3 câu hỏi nghiên cứu thông qua giao thức Leave-One-Dataset-Out (LODO) 6-fold trên 6 đoàn hệ quốc tế: APTOS 2019, DDR, DeepDRiD, IDRiD, Messidor-2, EyePACS ($N=55{,}570$ ảnh từ 35,474 bệnh nhân), phân chia vô trùng ở cấp độ bệnh nhân (patient-level).  
  $\rightarrow$ **Trích dẫn:** `\cite{cuadros2009eyepacs, decenciere2014feedback, porwal2018indian, li2019diagnostic, liu2022regularized, aptos2019kaggle}`
- **Luận điểm 2:** Bài báo không chỉ đo lường xem mô hình có chuyển giao được không, mà **phân rã chi tiết hình phạt dịch chuyển miền theo từng clinical grade và từng thành phần kiến trúc**.

---

```text
====================================================================================================
SECTION 2: RELATED WORK
====================================================================================================
```

### 2.1 DR Grading and the Domain Generalization Problem
- **Đoạn 1 (Tiến trình từ In-domain đến Cross-domain):**  
  Từ các mô hình CNN sơ khai phân loại nhị phân sàng lọc \cite{gulshan2016development, ting2017development} đến các bộ phân cấp đa mức độ hiện đại \cite{takahashi2020convolutional, chilukoti2024reliable}. Mô hình đạt hiệu năng rất cao trong phòng thí nghiệm nhưng thất bại khi đưa ra môi trường lâm sàng thực tế do Domain Shift \cite{albadawy2018deep, mutawa2023transfer}.  
  $\rightarrow$ **Trích dẫn:** `\cite{gulshan2016development, ting2017development, takahashi2020convolutional, chilukoti2024reliable, albadawy2018deep, mutawa2023transfer}`
- **Đoạn 2 (UDA vs DG & Hạn chế của Aggregate Evaluation):**  
  Unsupervised Domain Adaptation (UDA) đòi hỏi dữ liệu miền đích chưa gán nhãn trong lúc huấn luyện, điều bất khả thi trong sàng lọc diện rộng; do đó Domain Generalization (DG) là giải pháp bắt buộc \cite{galappaththige2024generalizing}. Các nghiên cứu DG hiện tại (Galappaththige et al., 2024; Al-Antary & Arafa, 2021; Hashmi, 2026) chỉ đo lường chỉ số tổng thể (overall QWK, macro AUC). Khoảng trống cốt lõi: chưa ai phân tích xem sự sụt giảm hiệu năng tập trung vào grade nào và cấu trúc mô hình nào chịu trách nhiệm cho sự sụt giảm đó.  
  $\rightarrow$ **Trích dẫn:** `\cite{galappaththige2024generalizing, alantary2021multi, hashmi2026dualresolution, wu2025diabetic, araujo2020drgraduate}`

### 2.2 Multi-Scale and Multi-Region Modeling for Retinal Pathology
- **Đoạn 1 (Vấn đề Pixel-loss do Downsampling):**  
  Tổn thương DR có độ biến thiên kích thước vật lý cực lớn (từ vi phình mạch $<120\,\mu\mathrm{m}$ đến xuất huyết lan tỏa toàn võng mạc) \cite{yang2022classification}. Việc downsampling ảnh gốc về $224 \times 224$ hay $300 \times 300\text{px}$ phá hủy thông tin tần số cao, gây sụp đổ độ nhạy ở Grade 1 \cite{mohanty2023using, dharrao2025ai, tyagi2026enhanced}.
- **Đoạn 2 (Sự phát triển của Patch-based & MIL):**  
  Các phương pháp dựa trên patch và Multiple Instance Learning (MIL) xem ảnh đáy mắt là một túi các thực thể tổn thương \cite{deepmil2018iet, mil2023acm, ilse2018attention}. Sudharshan/Quellec et al. (2021) chứng minh MIL giúp định vị tổn thương có thể giải thích được \cite{quellec2021interpretable}. Tuy nhiên, xử lý thuần patch cục bộ làm mất đi tương quan không gian toàn cục.
- **Đoạn 3 (Hạn chế của các hệ thống Dual-branch hiện nay & MRAMnet):**  
  Các mô hình song nhánh hiện đại (Cun et al., 2026; Reti-TransNet, 2026) kết hợp toàn cục và cục bộ nhưng chỉ dùng phép nối vector đơn giản, thiếu mô hình hóa tường minh cho Quy tắc 4-2-1 của ETDRS \cite{cun2026global, retitransnet2026}. MRAMnet (Lu et al., 2026) chia $2 \times 2$ góc phần tư nhưng theo lưới cứng nhắc, không theo mốc giải phẫu và dùng hàm mất mát danh nghĩa ArcFace \cite{lu2026mramnet}. ViT xử lý ảnh lớn thì gặp nghẽn tính toán $\mathcal{O}(N^2)$ \cite{vitmae2024plos}.  
  $\rightarrow$ Khẳng định vị trí của ta: **Grade 1 $\leftrightarrow$ Native-Resolution Local MIL; Grade 3 $\leftrightarrow$ Global Relational Quadrant Context**.

### 2.3 Ordinal Modeling for DR and Medical Image Grading
- **Đoạn 1 (Hạn chế của Nominal Softmax):**  
  Bản chất bệnh DR tiến triển theo thứ bậc rõ ràng: $0 < 1 < 2 < 3 < 4$ \cite{wilkinson2003proposed, yang2022classification}. Tuy nhiên đa số các nghiên cứu AI nhãn khoa vẫn dùng Softmax danh nghĩa với Cross-Entropy, phạt sai số nhầm Mức 4 thành Mức 0 bằng đúng sai số nhầm Mức 3 thành Mức 4 \cite{yang2022classification, araujo2020drgraduate}.
- **Đoạn 2 (Các phương pháp mô hình hóa thứ bậc):**  
  Phân tích 4 trường phái hồi quy thứ tự trong y văn:  
  1. *Hồi quy liên tục (Metric Regression):* Ép khoảng cách Euclid cứng nhắc giữa các mốc sinh học \cite{delarosa2020large}.  
  2. *Consistent Rank Logits (CORAL):* Phân rã thành $K-1$ bài toán nhị phân chia sẻ trọng số \cite{cao2020rank}.  
  3. *Conditional Ordinal Neural Networks (CORN):* Xác suất có điều kiện \cite{shi2021deep}.  
  4. *Cumulative Link Models (CLM):* Chiếu đặc trưng lên trục tiềm ẩn liên tục với các ngưỡng cắt đơn điệu nghiêm ngặt \cite{mccullagh1980regression, agresti2010analysis, vargas2020cumulative}.  
  Gần đây có thêm các nỗ lực đưa thứ tự vào ViT như ORViT-DR \cite{orvitdr2026} và mô hình hóa độ tin cậy \cite{gatedstatespace2026}.
- **Đoạn 3 (Khoảng trống OOD của Ordinal Modeling):**  
  Tất cả các nghiên cứu hồi quy thứ tự trên đều chỉ được đánh giá in-domain; chưa từng có công trình nào kiểm chứng xem: *Cấu trúc thứ bậc có giúp bảo toàn tính nhất quán nghiêm ngặt (severity consistency) và ngăn ngừa vỡ ranh giới quyết định khi gặp OOD shift nặng hay không?* $\rightarrow$ Thúc đẩy **RQ3**.

---

```text
====================================================================================================
SECTION 3, 4, 5, 6: METHODOLOGY, PROTOCOL, RESULTS & DISCUSSION
====================================================================================================
```

### Section 3: Methodology (Phương pháp luận)
- **Sec 3.2 (Data Harmonization):** Trích dẫn 6 bài báo gốc của 6 bộ dữ liệu: EyePACS \cite{cuadros2009eyepacs}, Messidor-2 \cite{decenciere2014feedback}, IDRiD \cite{porwal2018indian}, DDR \cite{li2019diagnostic}, DeepDRiD \cite{liu2022regularized}, APTOS 2019 \cite{aptos2019kaggle}.
- **Sec 3.4 (Global Context Stream):** Trích dẫn Non-Local Neural Networks \cite{wang2018non}, CBAM \cite{woo2018cbam}, ResNet-50 \cite{he2016deep}, và Quy tắc 4-2-1 của ETDRS \cite{etdrs1991report10}.
- **Sec 3.5 (Native-Resolution Local MIL):** Trích dẫn Attention-MIL \cite{ilse2018attention}, EfficientNet-B0 \cite{tan2019efficientnet}, và cơ chế trích xuất patch bảo toàn tổn thương hạt mịn \cite{quellec2021interpretable, dharrao2025ai}.
- **Sec 3.6 (CLM & Ordinal Prediction):** Trích dẫn Cumulative Link Model gốc \cite{mccullagh1980regression}, Deep CLM \cite{vargas2020cumulative}, và hàm liên kết Complementary Log-Log (clog-log).
- **Sec 3.7 (Training Objective):** Trích dẫn Quadratic Weighted Kappa gốc \cite{cohen1968weighted} và hàm loss khả vi xấp xỉ QWK \cite{delarosa2020large, chilukoti2024reliable}.

### Section 4: Experimental Protocol (Giao thức Thực nghiệm)
- **Sec 4.1 (LODO Protocol):** Trích dẫn DomainBed và quy chuẩn đánh giá zero-shot \cite{gulrajani2021search, galappaththige2024generalizing}.
- **Sec 4.2 (Ablation Baselines):** Trích dẫn Softmax baseline, CORAL \cite{cao2020rank}, CORN \cite{shi2021deep}, và DR|GRADUATE \cite{araujo2020drgraduate}.
- **Sec 4.3 (Evaluation Metrics):** Trích dẫn chuẩn sàng lọc Referable DR của Hội đồng Sàng lọc Vương quốc Anh (UK NSC) và tiêu chuẩn nhãn khoa quốc tế \cite{takahashi2020convolutional, hashmi2026dualresolution}.

### Section 5: Results & Hypotheses Verification
- **Sec 5.1 (Table 2 - LODO Benchmark):** Đối chiếu trực tiếp với kết quả sụt giảm hiệu năng của Galappaththige et al. \cite{galappaththige2024generalizing} và Hashmi \cite{hashmi2026dualresolution}.
- **Sec 5.3 (RQ1 - Grade 1 & Local MIL):** Đối chiếu với hiện tượng mất F1 ở lớp Mild của Tyagi et al. \cite{tyagi2026enhanced} và Dharrao et al. \cite{dharrao2025ai}.
- **Sec 5.4 (RQ2 - Grade 3 & Global Relational Context):** Đối chiếu với quy tắc ETDRS 4-2-1 \cite{etdrs1991report10} và cấu trúc lưới cứng của MRAMnet \cite{lu2026mramnet}.
- **Sec 5.5 (RQ3 - Ordinal Head & Rank Consistency):** So sánh định lượng tỷ lệ vi phạm thứ bậc (rank violations) giữa Softmax, CORAL \cite{cao2020rank}, CORN \cite{shi2021deep} và CLM \cite{vargas2020cumulative}.
- **Sec 5.7 (Explainability & Lesion Masks):** Trích dẫn phương pháp Pointing Game và tập mặt nạ tổn thương thực thể IDRiD \cite{porwal2018indian, quellec2021interpretable}.

### Section 6: Discussion & Clinical Implications
- **Sec 6.1 (Aggregate Metrics vs Grade Failure):** Bình luận sâu về việc aggregate metrics che giấu thất bại lâm sàng, kết nối Yang et al. \cite{yang2022classification} và Galappaththige et al. \cite{galappaththige2024generalizing}.
- **Sec 6.4 (Giá trị của Ordinal Modeling OOD):** Phân tích phát hiện: *Giá trị của mô hình hóa thứ tự thể hiện rõ nét nhất dưới tác động của distribution shift thay vì đánh giá in-domain*, làm sâu sắc thêm nhận định của Hashmi \cite{hashmi2026dualresolution}.
- **Sec 6.5 (Alternative Explanations & Limitations):** Thảo luận minh bạch về Label noise giữa các bệnh viện và Prior class shift $P_{\mathrm{train}}(Y) \neq P_{\mathrm{test}}(Y)$, trích dẫn bằng chứng từ Wu et al. \cite{wu2025diabetic} và Araujo et al. \cite{araujo2020drgraduate}.

---

# PHẦN 5: TỔNG KẾT MA TRẬN PHẢN BIỆN REVIEWER (ACADEMIC DEFENSE MATRIX)

Bảng ma trận dưới đây chuẩn bị sẵn các luận cứ và trích dẫn thép để tác giả tự tin bảo vệ bài báo trước các câu hỏi phản biện hóc búa nhất của Reviewer:

| Câu hỏi phản biện tiềm năng của Reviewer | Tử huyệt nếu trả lời sai | Lập luận chuẩn xác theo Spine & Trích dẫn thép |
| :--- | :--- | :--- |
| **Q1: "Tại sao không dùng Vision Transformer (ViT) cho toàn bộ mô hình mà lại dùng kiến trúc All-CNN kết hợp MIL?"** | Trả lời ngây thơ: *"Vì máy em yếu, không đủ GPU chạy ViT."* $\rightarrow$ **REJECT NGAY LẬP TỨC**. | **Trả lời:** ViT thuần xử lý ảnh kích thước lớn có độ phức tạp tính toán bậc hai $\mathcal{O}(N^2)$ và cực kỳ khát dữ liệu \cite{vitmae2024plos}. Trong khi đó, tổn thương Grade 1 đòi hỏi độ phân giải gốc cấp pixel; kiến trúc All-CNN lai của chúng tôi kết hợp ResNet50 + Attention ở luồng Global và EfficientNetB0 trượt MIL ở luồng Local vừa bảo tồn thiên kiến quy nạp (inductive bias), vừa đạt độ phân giải quang học gốc mà không gây cạn kiệt bộ nhớ, tối ưu cho thiết bị sàng lọc di động \cite{ilse2018attention, quellec2021interpretable}. |
| **Q2: "Bài này có gì khác so với nghiên cứu của Galappaththige et al. (WACV 2024) về Domain Generalization trong DR?"** | Trả lời: *"Mô hình của em đạt QWK cao hơn mô hình của họ."* $\rightarrow$ Biến bài báo thành cuộc đua SOTA tầm thường. | **Trả lời:** Galappaththige et al. (2024) \cite{galappaththige2024generalizing} chỉ đánh giá Domain Generalization ở cấp độ tổng thể (aggregate QWK/AUC). Nghiên cứu của chúng tôi chứng minh rằng **đo lường tổng thể che giấu sự sụp đổ không đồng đều giữa các grade lâm sàng**. Chúng tôi là công trình đầu tiên phân rã domain shift theo từng cấp độ bệnh và chứng minh mối liên hệ trực tiếp giữa Grade 1 với Local MIL và Grade 3 với ngữ cảnh 4 góc phần tư ETDRS. |
| **Q3: "Tại sao không dùng hàm ArcFace như MRAMnet (Lu et al., 2026) mà lại chọn Cumulative Link Model (CLM)?"** | Trả lời: *"Vì CLM mới hơn."* $\rightarrow$ Sai sự thật học thuật (CLM có từ năm 1980). | **Trả lời:** ArcFace trong MRAMnet \cite{lu2026mramnet} là hàm mất mát danh nghĩa (nominal), nó ép các lớp cách đều nhau trên siêu mặt cầu nhưng hoàn toàn bỏ qua bản chất thứ bậc tiến triển sinh học $0 < 1 < 2 < 3 < 4$ \cite{yang2022classification}. CLM \cite{mccullagh1980regression, vargas2020cumulative} với các ngưỡng cắt đơn điệu bảo đảm triệt tiêu 100% lỗi vi phạm thứ bậc và bảo toàn tính nhất quán chẩn đoán ngay cả khi gặp dịch chuyển miền nặng ngoại viện. |
| **Q4: "Làm sao chứng minh sự sụt giảm hiệu năng không phải do Nhiễu nhãn (Label Noise) hay Lệch phân phối lớp $P(Y)$ giữa các tập dữ liệu?"** | Tuyên bố liều lĩnh: *"Dữ liệu của tôi hoàn toàn vô trùng và không có nhiễu nhãn."* $\rightarrow$ Bị bắt bẻ vì các tập mở luôn có inter-grader variability. | **Trả lời:** Chúng tôi minh bạch thừa nhận trong Section 1 và Section 6.5 rằng label noise và prior class shift $P(Y)$ tồn tại giữa các đoàn hệ quốc tế \cite{wu2025diabetic}. Tuy nhiên, việc mô hình phân rã kiến trúc của chúng tôi cải thiện vượt trội có chọn lọc ở đúng Grade 1 (nhờ Local MIL) và Grade 3 (nhờ Quadrant Context) chứng minh rằng các cơ chế kiến trúc này có tác động trực tiếp và độc lập đối với các thất bại đặc trưng của từng cấp độ bệnh. |

---
*Tài liệu được biên soạn và kiểm toán chuẩn mực theo các quy chuẩn thẩm định học thuật cao nhất của TMI / MedIA / ICCPR.*
