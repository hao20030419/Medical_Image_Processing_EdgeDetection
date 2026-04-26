import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_and_display(image_path, title):
    # 讀取灰階影像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read {image_path}")
        return

    # 高斯模糊 (有助於減少雜訊)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # 1. Sobel 邊緣檢測
    # 分別計算 X 與 Y 方向的梯度，然後計算其大小
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    sobel_combined = cv2.convertScaleAbs(sobel_combined, alpha=12.0, beta=0) # 將邊緣整體增亮 12 倍
    # 2. Laplacian of Gaussian (LoG) 邊緣檢測
    # 由於 Laplacian 對雜訊非常敏感，這裡特別先進行平滑處理 (Smoothing) 再取二階導數
    #lap_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = cv2.blur(laplacian, (5, 5))
    laplacian = cv2.normalize(np.absolute(laplacian), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    laplacian = cv2.convertScaleAbs(laplacian, alpha=20.0, beta=0) # 將邊緣整體增亮 20 倍
    
    # 3. 結合方法：先做 Laplacian 再做 Sobel (中間加入 5x5 平均平滑)
    # 先以 Laplacian 計算二階導數，接著做 5x5 的 Averaging blur，最後再利用 Sobel 取一階梯度
    # 這能有效減緩 Laplacian 產生的劇烈高頻雜訊，讓後續的 Sobel 捕捉到更平坦、連續的邊緣
    lap_raw = cv2.Laplacian(img, cv2.CV_64F)
    lap_avg = cv2.blur(lap_raw, (5, 5)) # 加入 5x5 Averaging
    lap_sobel_x = cv2.Sobel(lap_avg, cv2.CV_64F, 1, 0, ksize=3)
    lap_sobel_y = cv2.Sobel(lap_avg, cv2.CV_64F, 0, 1, ksize=3)
    lap_sobel_combined = cv2.magnitude(lap_sobel_x, lap_sobel_y)
    lap_sobel_combined = cv2.normalize(lap_sobel_combined, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    lap_sobel_combined = cv2.convertScaleAbs(lap_sobel_combined, alpha=10.0, beta=0)
    
    # 4. 改良版 Canny 邊緣檢測 (針對低對比度醫學影像)
    # 醫學影像邊緣梯度極小，原來的閾值 (50, 150) 太高，會把真正的組織邊緣當作雜訊丟棄。
    # 改善方法：大幅調低雙閾值至非常低，且不強加模糊，否則微弱邊緣會消失
    canny = cv2.Canny(img, threshold1=10, threshold2=75)
    
    # 繪圖
    plt.figure(figsize=(15, 10))
    plt.suptitle(f'Edge Detection: {title}', fontsize=16)
    
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('1. Sobel Edge Detection')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(laplacian, cmap='gray')
    plt.title('2. Laplacian (LoG)')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(canny, cmap='gray')
    plt.title('3. Canny Edge Detection')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(lap_sobel_combined, cmap='gray')
    plt.title('4. Laplacian + Sobel')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}_edges.png")
    print(f"Saved: {title.replace(' ', '_')}_edges.png")
    # plt.show() # 如果要在 VS code 中執行並儲存圖片，這裡先註解掉

if __name__ == "__main__":
    # 處理自然影像 (Natural Image)
    process_and_display('image/house_1.jpg', 'Natural Image (House)')
    
    # 處理醫學影像 (Medical Image)
    process_and_display('image/pneumothorax_1.jpeg', 'Medical Image (Pneumothorax X-Ray)')
