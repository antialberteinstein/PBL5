import cv2
import os
import sys
import time

# Chèn đường dẫn để xài ké Utils trong thư mục Src của hệ thống
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from utils.pose_utils import POSES, get_pose_name
from utils.mask_utils import add_virtual_mask
import insightface

def capture_dataset():
    mssv = input("🪪 Nhập ID hoặc Tên Sinh viên (VD: 102210, Nguyen...): ").strip()
    if not mssv:
        print("❌ Lỗi: Mã định danh không được để trống!")
        return
        
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../dataset/faces/{mssv}"))
    os.makedirs(out_dir, exist_ok=True)
    
    print("⏳ Đang khởi động AI quét khuôn mặt...")
    # Thâm module landmark_3d_68 để tính toán góc nghiêng (Pitch, Yaw, Roll)
    app = insightface.app.FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'landmark_2d_106', 'landmark_3d_68'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    cap = cv2.VideoCapture(0)
    
    images_per_pose = 5
    last_capture_time = 0
    pose_idx = 0
    pose_count = 0
    
    print("\n" + "="*40)
    print("📸 HƯỚNG DẪN LẤY DATASET 📸")
    print("Màn hình sẽ yêu cầu góc mặt. Máy sẽ CHỈ CHỤP khi mày vào ĐÚNG GÓC.")
    print("Mỗi phát chụp máy tính tự nhân bản thêm 1 tấm rọi khẩu trang ảo!")
    print("Nhấn phím 'Q' nếu muốn thoát ngang.")
    print("="*40 + "\n")
    
    while cap.isOpened() and pose_idx < len(POSES):
        ret, frame = cap.read()
        if not ret:
            break
            
        display_frame = frame.copy()
        target_pose = POSES[pose_idx]
        
        faces = app.get(frame)
        current_pose = "Khong xac dinh"
        
        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
            face = faces[0]
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box
            
            # Tính Góc Mặt hiện tại
            current_pose = get_pose_name(face.pose)
            
            # Padding 50%
            pad_x = int((x2 - x1) * 0.5)
            pad_y = int((y2 - y1) * 0.5)
            
            cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            cx2, cy2 = min(frame.shape[1], x2 + pad_x), min(frame.shape[0], y2 + pad_y)
            
            # Màu HUD: Xanh khi đúng góc, Đỏ khi sai góc
            is_correct_pose = (current_pose == target_pose)
            color = (0, 255, 0) if is_correct_pose else (0, 0, 255)
            
            cv2.rectangle(display_frame, (cx1, cy1), (cx2, cy2), color, 2)
            
            # Cơ chế chụp tự động
            current_time = time.time()
            if is_correct_pose and (current_time - last_capture_time > 0.4):
                # 1. Chụp hình mặt gốc
                raw_crop = frame[cy1:cy2, cx1:cx2].copy()
                
                # 2. Rọi thêm Khẩu Trang Ảo
                # Đoạn này phải bọc lại cho đúng chuẩn cấu trúc mảng mà hàm của mày đòi hỏi (tránh vẽ viền trắng)
                class FakeFace: pass
                fake_face = FakeFace()
                fake_face.bbox = face.bbox
                fake_face.landmarks = face.landmark_2d_106
                
                frame_with_mask = add_virtual_mask(frame.copy(), fake_face)
                masked_crop = frame_with_mask[cy1:cy2, cx1:cx2]
                
                if raw_crop.size > 0:
                    p_name = target_pose.replace(' ', '_')
                    ts = int(current_time)
                    cv2.imwrite(os.path.join(out_dir, f"{p_name}_{pose_count}_{ts}.jpg"), raw_crop)
                    cv2.imwrite(os.path.join(out_dir, f"{p_name}_{pose_count}_{ts}_masked.jpg"), masked_crop)
                    
                    pose_count += 1
                    last_capture_time = current_time
                    
                    # Chuyển Sang Pose tiếp theo nếu đủ 5 tấm ảnh
                    if pose_count >= images_per_pose:
                        pose_idx += 1
                        pose_count = 0
                        time.sleep(1) # Nghỉ xíu chuyển góc
        
        # UI Hướng dẫn hiển thị trên màn hình
        cv2.putText(display_frame, f"Yeu cau: {target_pose} ({pose_count}/{images_per_pose})", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(display_frame, f"Goc hien tai: {current_pose}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
        cv2.imshow("AUTO DATASET (LORA)", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("\n❌ Thoát sớm.")
            break
            
    if pose_idx >= len(POSES):
        print(f"\n✅ FULL COMBO: Đã lấy đủ {len(POSES)*images_per_pose*2} tấm hình (Cả Khẩu trang) tại '{out_dir}'")
        
    cap.release()
    cv2.destroyAllWindows()
    
    import sys
    if sys.platform == "darwin":
        for _ in range(30): cv2.waitKey(1)

if __name__ == "__main__":
    capture_dataset()
