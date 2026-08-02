# 1. Khởi tạo lại kiến trúc mạng (phải giống hệt lúc train)
# Lưu ý: Lúc test không cần nhánh phụ (use_aux=False) để chạy cho nhẹ
loaded_model = GoogLeNet(num_classes=5, use_aux=False)

# 2. Đưa kiến trúc vào đúng Device (CPU hoặc GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model = loaded_model.to(device)

# 3. Nạp file trọng số đã lưu vào kiến trúc mạng
# map_location giúp chống lỗi khi bạn train trên GPU nhưng lại mang máy khác chỉ có CPU ra test
loaded_model.load_state_dict(torch.load('best_googlenet_aptos.pth', map_location=device))

# 4. Bật chế độ Evaluation trước khi dự đoán
loaded_model.eval()

print("✅ Đã load model thành công và sẵn sàng dự đoán!")