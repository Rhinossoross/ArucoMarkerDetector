import cv2
import numpy as np

output_path = "marker maker/chessboard_9x6.png"
square_size_px: int = 80          # Size of each square in pixels
num_squares_width: int = 10       # Total squares horizontally (inner corners = num_squares_width  - 1 = 9)
num_squares_height: int = 7       # Total squares vertically   (inner corners = num_squares_height - 1 = 6)
margin_px: int = 4                # White margin around the board (helps printing) --2px of this will be black border
"""
Generates a chessboard calibration pattern and saves it as PNG.

- Starts with white in top-left corner (common convention)
- Black squares are filled with (0,0,0), white with (255,255,255)
- Saved in high resolution for printing/scanning
"""
# Calculate total image dimensions
board_width_px = num_squares_width * square_size_px
board_height_px = num_squares_height * square_size_px

total_width = board_width_px + 2 * margin_px
total_height = board_height_px + 2 * margin_px

# Create white background
img = np.full((total_height, total_width, 3), 255, dtype=np.uint8)

# Draw the chessboard pattern inside the margin
for row in range(num_squares_height):
    for col in range(num_squares_width):
        # Alternate colors: start with white (255) in top-left
        color = 0 if (row + col) % 2 == 1 else 255
        x_start = margin_px + col * square_size_px
        y_start = margin_px + row * square_size_px
        x_end = x_start + square_size_px
        y_end = y_start + square_size_px
        
        img[y_start:y_end, x_start:x_end] = color

# Optional: add thin black border around the whole board for clarity when printing
cv2.rectangle(img, (margin_px-2, margin_px-2),
                    (margin_px + board_width_px + 1, margin_px + board_height_px + 1),
                    (0, 0, 0), thickness=2)

# Save the image


success = cv2.imwrite(str(output_path), img)
if success:
    print(f"Chessboard pattern saved successfully to: {output_path}")
    print(f"Dimensions: {total_width} × {total_height} pixels")
    print(f"Inner corners: 9 × 6 (use in calibration script)")
    print(f"Each square: {square_size_px}px → real size = {square_size_px / 25.4:.2f} inches")
else:
    print("Failed to save image. Check permissions or path.")
