import os
import requests
import re
from duckduckgo_search import DDGS

# =========================================================================
# CẤU HÌNH GLOBAL
# =========================================================================

# Thư mục gốc để lưu tất cả các ảnh lấy về (cùng thư mục mà mày nói)
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/faces"))

# Điền danh sách các Idol mà mày muốn làm "đòn bẩy" cho Model ở đây
IDOL_LIST = [
    "Son Tung MTP",
    "Cristiano Ronaldo",
    "Lionel Messi",
    "Bill Gates",
    "Elon Musk",
    "Taylor Swift",
    "Blackpink Lisa",
    "Snoop Dogg",
    "Chau Tinh Tri",
    "Ly Tieu Long"
]

# Max số ảnh cần tải cho mỗi Idol (tầm 15 tấm là vừa đủ để đánh lừa model rồi)
NUM_IMAGES_PER_IDOL = 15

# =========================================================================

def sanitize_foldername(name):
    """Giúp chuyển tên Idol thành dạng an toàn để làm tên thư mục (Xóa dấu cách...)"""
    clean = re.sub(r'[^a-zA-Z0-9]', '', name.strip())
    return clean

def download_image(url, save_path):
    """Tải và lưu một ảnh nếu link chưa chết"""
    try:
        # Nhét thêm User-Agent để server DuckDuckGo tưởng là trình duyệt
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, timeout=5, headers=headers)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception:
        pass
    return False

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Khởi tạo súng bắn DuckDuckGo
    ddgs = DDGS()

    print(f"BẮT ĐẦU TẢI DỮ LIỆU FAKE CHO {len(IDOL_LIST)} IDOL...")
    print(f"Thư mục lưu trữ: {OUTPUT_DIR}\n" + "-"*50)

    for idol in IDOL_LIST:
        print(f"\n🚀 Đang săn ảnh cho Idol: {idol}")
        
        folder_name = sanitize_foldername(idol)
        idol_dir = os.path.join(OUTPUT_DIR, folder_name)
        
        if not os.path.exists(idol_dir):
            os.makedirs(idol_dir)

        # Keyword tao nhét thêm chữ "face portrait closeup" để máy nó ưu tiên tìm ảnh chụp cận mặt, dễ cho việc ArcFace crop
        try:
            results = ddgs.images(
                keywords=f"{idol} face portrait closeup",
                region="wt-wt",
                safesearch="moderate",
                max_results=NUM_IMAGES_PER_IDOL * 3 # Gấp 3 phòng trường hợp link chết
            )
        except Exception as e:
            print(f"❌ API DuckDuckGo chặn hoặc sập khi tìm {idol}: {e}")
            continue

        downloaded_count = 0
        for res in results:
            if downloaded_count >= NUM_IMAGES_PER_IDOL:
                break
                
            img_url = res.get('image')
            if not img_url:
                continue
                
            ext = img_url.split('.')[-1].lower()
            if ext not in ['jpg', 'jpeg', 'png']:
                ext = 'jpg'
                
            file_path = os.path.join(idol_dir, f"{downloaded_count + 1:02d}.{ext}")
            
            # Tiến hành cào
            if download_image(img_url, file_path):
                downloaded_count += 1
                print(f"   [+] Bắt được tấm {downloaded_count}/{NUM_IMAGES_PER_IDOL}")
                
        print(f"✅ Hoàn tất ID: {idol}. (Lưu tại thư mục '{folder_name}')")

if __name__ == "__main__":
    main()
