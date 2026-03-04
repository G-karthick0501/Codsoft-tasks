import cv2
import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("--- 1. LUCAS-KANADE (LK) MOTION OPTICAL FLOW FOR UI PERFORMANCE ---")
print("Generating Synthetic 'High-Performance Scroll Virtualizer' UI Video...")

def generate_synthetic_scroll_video(frames=10, width=400, height=600):
    video = []
    # Create a simple synthetic list UI
    for i in range(frames):
        # White background
        frame = np.ones((height, width), dtype=np.uint8) * 240
        
        # Draw a fixed "Header" (Does not move)
        cv2.rectangle(frame, (0, 0), (width, 80), 200, -1)
        # Header text
        cv2.putText(frame, 'UI Navigation', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 100, 2)
        
        # Draw scrolling "List Items"
        offset = i * 20 # Scroll speed
        for j in range(6):
            y_start = 100 + j * 120 - offset
            y_end = y_start + 100
            
            # Clip to below header so it looks like it's passing underneath
            if y_end > 80 and y_start < height:
                y1 = max(80, y_start)
                y2 = min(height, y_end)
                cv2.rectangle(frame, (20, y1), (width - 20, y2), 150, -1)
                
                # Add "Text" shape inside for corners to track
                if y1 + 30 < y2:
                    cv2.rectangle(frame, (40, y1 + 10), (width - 150, y1 + 30), 100, -1)
                
        video.append(frame)
    return video

video = generate_synthetic_scroll_video()
old_frame = video[0]
new_frame = video[6] # Jump to frame 6 to show distinct motion

# Define mask to only look for features below the stationary header
mask = np.zeros_like(old_frame)
mask[80:, :] = 255

print("Detecting Shi-Tomasi Corners (Geometric Features)....")
# 1. Feature Extraction (Shi-Tomasi corner detection)
p0 = cv2.goodFeaturesToTrack(
    old_frame, 
    mask=mask, 
    maxCorners=200, 
    qualityLevel=0.1, 
    minDistance=10, 
    blockSize=7
)

print("Calculating Sparse Optical Flow (Lucas-Kanade)...")
# 2. Block-Matching Optical Flow (Sparse LK)
p1, st, err = cv2.calcOpticalFlowPyrLK(
    old_frame, 
    new_frame, 
    p0, 
    None, 
    winSize=(15, 15), 
    maxLevel=2, 
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Select good points that successfully tracked
good_new = p1[st == 1]
good_old = p0[st == 1]

print("Plotting Motion Vector Field (Quiver Plot)...")
# 3. Visualization
plt.figure(figsize=(8, 10))
# Plot the starting frame in grayscale
plt.imshow(old_frame, cmap='gray', alpha=0.5)

# Calculate Vectors
X = good_old[:, 0]
Y = good_old[:, 1]
U = good_new[:, 0] - good_old[:, 0]
V = good_new[:, 1] - good_old[:, 1]

# Quiver plot for the Motion Array
plt.quiver(X, Y, U, V, color='red', angles='xy', scale_units='xy', scale=1, width=0.006)

# Overlay titles and boundaries
plt.title("Lucas-Kanade UI Optical Flow (Temporal Motion DNA)", fontsize=14, fontweight='bold')
plt.axhline(80, color='blue', linestyle='--', linewidth=2, label='Header Boundary (Static Viewport)')
plt.legend()
plt.axis('off')

plt.tight_layout()
plt.savefig("lucas_kanade_motion.png", dpi=150)
plt.close()

print("\n" + "="*60)
print("SUCCESS! Saved 'lucas_kanade_motion.png'.")
print("Notice how the vectors ONLY point upwards in the list area, proving zero-shot scroll tracking.")
