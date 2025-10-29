import tkinter
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk

# ========================= Globals =========================
filePath = None
fig = None
canvas = None
plt.style.use('dark_background')


def check_image():
    """Checks if an image has been loaded and shows an error if not."""
    if not filePath:
        messagebox.showerror("Error", "Please upload an image first.")
        return False
    return True


def OpenImage():
    """Opens a file dialog and displays the image on the canvas."""
    global filePath, fig, canvas
    filePath = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
    )
    if not filePath:
        return
    try:
        img = Image.open(filePath).convert("RGB")
        
        # --- Draw on canvas ---
        fig.clear() # Clear the previous plot
        ax = fig.add_subplot(111) # Add a single new plot
        ax.imshow(img)
        ax.set_title(f"Original: {filePath.split('/')[-1]}", fontsize=10)
        ax.axis('off')
        canvas.draw() # Refresh the canvas in the GUI
        
        messagebox.showinfo("Success", "Image loaded successfully!")
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image: {e}")
        filePath = None

# ------------------- Basic Operations -------------------
def SeparateRGB():
    """(This function's layout is already multi-plot, no change needed)"""
    if not check_image(): return
    global fig, canvas
    img = Image.open(filePath).convert("RGB")
    arr = np.asarray(img) 
    
    fig.clear()
    
    ax_orig = fig.add_subplot(221)
    ax_orig.imshow(img)
    ax_orig.set_title("Original")
    ax_orig.axis('off')
    
    ax_r = fig.add_subplot(222)
    ax_r.imshow(arr[:, :, 0], cmap="gray")
    ax_r.set_title("Red Channel")
    ax_r.axis('off')
    
    ax_g = fig.add_subplot(223)
    ax_g.imshow(arr[:, :, 1], cmap="gray")
    ax_g.set_title("Green Channel")
    ax_g.axis('off')
    
    ax_b = fig.add_subplot(224)
    ax_b.imshow(arr[:, :, 2], cmap="gray")
    ax_b.set_title("Blue Channel")
    ax_b.axis('off')
    
    fig.tight_layout() 
    canvas.draw()

def ConvertRGBtoGray():
    """Converts to grayscale and displays Original vs. Grayscale."""
    if not check_image(): return
    global fig, canvas
    
    original_img = Image.open(filePath).convert("RGB")
    pixels = original_img.load()
    width, height = original_img.size
    
    gray_img = Image.new("RGB", (width, height))
    gray_pixels = gray_img.load()
    
    for x in range(width):
        for y in range(height):
            try:
                r, g, b = pixels[x, y][:3]
            except TypeError:
                r = g = b = pixels[x, y]
            gray = (r + g + b) // 3
            gray_pixels[x, y] = (gray, gray, gray)
            
    # --- Draw on canvas ---
    fig.clear()
    ax1 = fig.add_subplot(121)
    ax1.imshow(original_img)
    ax1.set_title("Original")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(122)
    ax2.imshow(gray_img)
    ax2.set_title("Grayscale (Average)")
    ax2.axis('off')
    
    fig.tight_layout()
    canvas.draw()

def Multiply():
    if not check_image(): return
    global fig, canvas
    
    original_img = Image.open(filePath)
    img_l = original_img.convert("L")
    arr = np.asarray(img_l).astype(np.int16)
    arr = arr * 2
    processed_arr = np.clip(arr, 0, 255).astype(np.uint8) 
    
    # --- Draw on canvas ---
    fig.clear()
    ax1 = fig.add_subplot(121)
    ax1.imshow(original_img, cmap='gray' if original_img.mode == 'L' else None)
    ax1.set_title("Original")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(122)
    ax2.imshow(processed_arr, cmap='gray')
    ax2.set_title("Brightness ×2")
    ax2.axis('off')
    
    fig.tight_layout()
    canvas.draw()

def Subtract():
    """Decreases brightness (-100) and displays Original vs. Processed."""
    if not check_image(): return
    global fig, canvas
    
    original_img = Image.open(filePath)
    img_l = original_img.convert("L")
    arr = np.asarray(img_l).astype(np.int16)
    arr = arr - 100
    processed_arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    # --- Draw on canvas ---
    fig.clear()
    ax1 = fig.add_subplot(121)
    ax1.imshow(original_img, cmap='gray' if original_img.mode == 'L' else None)
    ax1.set_title("Original")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(122)
    ax2.imshow(processed_arr, cmap='gray')
    ax2.set_title("Brightness -100")
    ax2.axis('off')
    
    fig.tight_layout()
    canvas.draw()

def ImageAddition():
    """Adds two images and displays Original 1 vs. Result."""
    if not check_image(): return
    global fig, canvas
    
    img1_path = filePath
    img2_path = filedialog.askopenfilename(title="Select Second Image to Add")
    if not img2_path:
        return
        
    try:
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB").resize(img1.size)
        
        arr1 = np.asarray(img1).astype(np.int16)
        arr2 = np.asarray(img2).astype(np.int16)
        
        result_arr = arr1 + arr2
        result_arr = np.clip(result_arr, 0, 255).astype(np.uint8)
        
        result_img = Image.fromarray(result_arr)
        
        # --- Draw on canvas ---
        fig.clear()
        ax1 = fig.add_subplot(121)
        ax1.imshow(img1)
        ax1.set_title("Original (Image 1)")
        ax1.axis('off')
        
        ax2 = fig.add_subplot(122)
        ax2.imshow(result_img)
        ax2.set_title("Added Image")
        ax2.axis('off')
        
        fig.tight_layout()
        canvas.draw()
        
    except Exception as e:
        messagebox.showerror("Error", f"Error during image addition: {e}")

def solarize_image():
    """Applies solarization and displays Original vs. Processed."""
    if not check_image(): return
    global fig, canvas
    
    original_img = Image.open(filePath).convert("RGB")
    arr = np.asarray(original_img).astype(np.uint8)
    solarized = arr.copy()
    solarized[arr == 255] = 0
    solarized[arr == 0] = 255
    
    # --- Draw on canvas ---
    fig.clear()
    ax1 = fig.add_subplot(121)
    ax1.imshow(original_img)
    ax1.set_title("Original")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(122)
    ax2.imshow(solarized)
    ax2.set_title("Solarized Image")
    ax2.axis('off')
    
    fig.tight_layout()
    canvas.draw()

def swap_red_blue():
    """Swaps R and B channels and displays Original vs. Processed."""
    if not check_image(): return
    global fig, canvas
    
    original_img = Image.open(filePath).convert("RGB")
    arr = np.asarray(original_img)
    swapped = arr[:, :, [2, 1, 0]]
    
    # --- Draw on canvas ---
    fig.clear()
    ax1 = fig.add_subplot(121)
    ax1.imshow(original_img)
    ax1.set_title("Original")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(122)
    ax2.imshow(swapped)
    ax2.set_title("Red ↔ Blue Swapped")
    ax2.axis('off')
    
    fig.tight_layout()
    canvas.draw()

def eliminate_channel(channel='R'):
    """Removes a color channel and displays Original vs. Processed."""
    if not check_image(): return
    global fig, canvas
    
    original_img = Image.open(filePath).convert("RGB")
    arr = np.asarray(original_img).copy()
    
    if channel.upper() == 'R': arr[:, :, 0] = 0
    elif channel.upper() == 'G': arr[:, :, 1] = 0
    elif channel.upper() == 'B': arr[:, :, 2] = 0
    
    # --- Draw on canvas ---
    fig.clear()
    ax1 = fig.add_subplot(121)
    ax1.imshow(original_img)
    ax1.set_title("Original")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(122)
    ax2.imshow(arr)
    ax2.set_title(f"{channel.upper()} Channel Removed")
    ax2.axis('off')
    
    fig.tight_layout()
    canvas.draw()

# ------------------- Filters -------------------
def median_filter():
    """Applies a 3x3 median filter and displays Original vs. Processed."""
    if not check_image(): return
    global fig, canvas
    
    original_img = Image.open(filePath).convert("RGB")
    arr = np.asarray(original_img)
    filtered = arr.copy()
    h, w, _ = arr.shape
    
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            for c in range(3):
                region = arr[y - 1:y + 2, x - 1:x + 2, c].flatten()
                filtered[y, x, c] = np.median(region)
                
    # --- Draw on canvas ---
    fig.clear()
    ax1 = fig.add_subplot(121)
    ax1.imshow(original_img)
    ax1.set_title("Original")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(122)
    ax2.imshow(filtered.astype(np.uint8))
    ax2.set_title("3x3 Median Filter")
    ax2.axis('off')
    
    fig.tight_layout()
    canvas.draw()

def average_filter():
    """Applies a 3x3 average filter and displays Original vs. Processed."""
    if not check_image(): return
    global fig, canvas
    
    original_img = Image.open(filePath).convert("RGB")
    arr = np.asarray(original_img).astype(np.float64)
    filtered = arr.copy()
    h, w, _ = arr.shape
    kernel = np.ones((3, 3)) / 9.0
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            for c in range(3):
                region = arr[y - 1:y + 2, x - 1:x + 2, c]
                total = np.sum(region * kernel)
                filtered[y, x, c] = int(total)
                
    # --- Draw on canvas ---
    fig.clear()
    ax1 = fig.add_subplot(121)
    ax1.imshow(original_img)
    ax1.set_title("Original")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(122)
    ax2.imshow(filtered.astype(np.uint8))
    ax2.set_title("3x3 Average Filter")
    ax2.axis('off')
    
    fig.tight_layout()
    canvas.draw()

def weighted_average_filter():
    """Applies a 3x3 weighted average filter and displays Original vs. Processed."""
    if not check_image(): return
    global fig, canvas
    
    original_img = Image.open(filePath).convert("RGB")
    arr = np.asarray(original_img).astype(np.float64)
    filtered = arr.copy()
    h, w, _ = arr.shape
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    sumk = 16.0
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            for c in range(3):
                region = arr[y - 1:y + 2, x - 1:x + 2, c]
                total = np.sum(region * kernel)
                filtered[y, x, c] = int(total / sumk)
                
    # --- Draw on canvas ---
    fig.clear()
    ax1 = fig.add_subplot(121)
    ax1.imshow(original_img)
    ax1.set_title("Original")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(122)
    ax2.imshow(filtered.astype(np.uint8))
    ax2.set_title("Weighted Avg Filter")
    ax2.axis('off')
    
    fig.tight_layout()
    canvas.draw()

def gaussian_filter():
    weighted_average_filter()

# ------------------- Histogram -------------------
def show_histogram():
    """(This function's layout is already multi-plot, no change needed)"""
    if not check_image(): return
    global fig, canvas
    img = Image.open(filePath).convert("L") 
    arr = np.asarray(img)
    hist, bins = np.histogram(arr.flatten(), 256, [0, 256])

    fig.clear()
    
    ax1 = fig.add_subplot(211) 
    ax1.imshow(arr, cmap='gray')
    ax1.set_title("Grayscale Image")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(212) 
    ax2.bar(range(256), hist, color='gray')
    ax2.set_title("Histogram")
    ax2.set_xlabel("Gray Level")
    ax2.set_ylabel("Count")
    
    fig.tight_layout()
    canvas.draw()

def histogram_stretching():
    """(This function's layout is already 1x2, no change needed)"""
    if not check_image(): return
    global fig, canvas
    img = Image.open(filePath).convert("L")
    arr = np.asarray(img)
    
    r_min, r_max = arr.min(), arr.max()
    if r_max == r_min:
        stretched = arr.copy()
    else:
        stretched = (arr - r_min) * (255.0 / (r_max - r_min))
        stretched = np.clip(stretched, 0, 255).astype(np.uint8)
        
    fig.clear()
    
    ax1 = fig.add_subplot(121)
    ax1.imshow(arr, cmap='gray')
    ax1.set_title("Original")
    ax1.axis("off")
    
    ax2 = fig.add_subplot(122)
    ax2.imshow(stretched, cmap='gray')
    ax2.set_title("Stretched")
    ax2.axis("off")
    
    fig.tight_layout()
    canvas.draw()

def histogram_equalization():
    """(This function's layout is already 1x2, no change needed)"""
    if not check_image(): return
    global fig, canvas
    img = Image.open(filePath).convert("L")
    arr = np.asarray(img)
    h, w = arr.shape
    
    hist = np.histogram(arr.flatten(), 256, [0, 256])[0]
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0][0]
    eq_map = np.floor((cdf - cdf_min) / (h * w - cdf_min) * 255).astype(np.uint8)
    equalized = eq_map[arr]
    
    fig.clear()
    
    ax1 = fig.add_subplot(121)
    ax1.imshow(arr, cmap='gray')
    ax1.set_title("Original")
    ax1.axis("off")
    
    ax2 = fig.add_subplot(122)
    ax2.imshow(equalized, cmap='gray')
    ax2.set_title("Equalized")
    ax2.axis("off")
    
    fig.tight_layout()
    canvas.draw()

# =============================================================================
# ============= GUI SECTION (UNCHANGED FROM PREVIOUS VERSION) =================
# =============================================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

window = ctk.CTk()
window.title("Modern Image Processing Studio")
window.geometry("1100x700")
window.minsize(700, 500)

# --- Configure the main window's grid layout ---
window.rowconfigure(1, weight=1) 
window.columnconfigure(0, weight=2)
window.columnconfigure(1, weight=1)

# --- Title ---
title_label = ctk.CTkLabel(window, text="🖼️ Image Processing Studio", 
                           font=ctk.CTkFont(size=24, weight="bold"))
title_label.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10))

# --- Left Frame (Preview Area) ---
preview_frame = ctk.CTkFrame(window)
preview_frame.grid(row=1, column=0, padx=(20, 10), pady=10, sticky="nsew")

preview_frame.rowconfigure(1, weight=1)
preview_frame.columnconfigure(0, weight=1)

load_button = ctk.CTkButton(preview_frame, text="Load Image", 
                            command=OpenImage, height=40,
                            font=ctk.CTkFont(size=14, weight="bold"))
load_button.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="n")

# --- Frame to hold the plot ---
plot_frame = ctk.CTkFrame(preview_frame, fg_color=("#DBDBDB", "#2B2B2B"))
plot_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(10, 20))
plot_frame.grid_rowconfigure(0, weight=1)
plot_frame.grid_columnconfigure(0, weight=1)

# --- Initialize Matplotlib Figure and Canvas ---
fig = Figure(figsize=(5, 4), dpi=100)
fig.patch.set_facecolor(plot_frame.cget('fg_color')[1]) 
fig.set_tight_layout(True)

ax = fig.add_subplot(111)
ax.set_title("Load an image to begin", color="gray")
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor(plot_frame.cget('fg_color')[1])
for spine in ax.spines.values():
    spine.set_edgecolor("gray")

canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.draw()
canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)


# --- Right Frame (Scrollable Controls) ---
scroll_frame = ctk.CTkScrollableFrame(window, label_text="Processing Tools",
                                      label_font=ctk.CTkFont(size=16, weight="bold"))
scroll_frame.grid(row=1, column=1, padx=(10, 20), pady=10, sticky="nsew")

sections = [
    ("Basic Operations", [
        ("Separate RGB", SeparateRGB),
        ("Convert to Gray", ConvertRGBtoGray),
        ("Add a Second Image", ImageAddition),
        ("Multiply (Brightness)", Multiply),
        ("Subtract (Brightness)", Subtract)
    ]),
    ("Color Manipulation", [
        ("Solarize Effect", solarize_image),
        ("Swap Red & Blue", swap_red_blue),
        ("Remove Red Channel", lambda: eliminate_channel('R')),
        ("Remove Green Channel", lambda: eliminate_channel('G')),
        ("Remove Blue Channel", lambda: eliminate_channel('B'))
    ]),
    ("Filtering", [
        ("Median Filter (3x3)", median_filter),
        ("Average Filter (3x3)", average_filter),
        ("Weighted Avg Filter", weighted_average_filter),
        ("Gaussian Filter (Approx)", gaussian_filter)
    ]),
    ("Histogram Tools", [
        ("Show Histogram", show_histogram),
        ("Histogram Stretching", histogram_stretching),
        ("Histogram Equalization", histogram_equalization)
    ])
]

# --- Populate the Scroll Frame ---
for section, buttons in sections:
    section_label = ctk.CTkLabel(scroll_frame, text=section, 
                                 font=ctk.CTkFont(size=14, weight="bold"),
                                 anchor="w")
    section_label.pack(fill="x", padx=10, pady=(15, 5))
    
    sep = ctk.CTkFrame(scroll_frame, height=2, fg_color="gray25")
    sep.pack(fill="x", padx=10, pady=(0, 5))

    for text, func in buttons:
        btn = ctk.CTkButton(scroll_frame, text=text, command=func, height=35)
        btn.pack(fill="x", padx=10, pady=4)

# --- Exit Button ---
exit_button = ctk.CTkButton(window, text="Exit", command=window.destroy, 
                            fg_color="#D32F2F", hover_color="#B71C1C",
                            height=40, font=ctk.CTkFont(size=14, weight="bold"))
exit_button.grid(row=2, column=0, columnspan=2, padx=20, pady=(10, 20), sticky="s")


window.mainloop()