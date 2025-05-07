import win32gui
import win32con
import win32ui
from PIL import Image


import win32gui
import win32con
import win32ui
from PIL import Image

import time

# TL coordinates for 500x500 box: width=505, height=505, topleft=(255, 278)
# to compensate for bad window TL = (10, 55), width = 990, height = 945


def screenshot_window_by_title(title, save_path,  width=495, height=475, topleft=(10, 50)):
    time.sleep(0.1)
    
    hwnd = win32gui.FindWindow(None, title)
    if not hwnd:
        print(f"Window '{title}' not found.")
        return

    # Bring window to foreground (optional)
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    #win32gui.SetForegroundWindow(hwnd)

    # Get window dimensions
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (left, top))
    right, bottom = win32gui.ClientToScreen(hwnd, (right, bottom))

    # Calculate the starting point (topleft) relative to the window's client area
    start_x, start_y = topleft
    start_x += left  # Add window's left offset
    start_y += top   # Add window's top offset

    # Get the window's device context (DC)
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    # Create a bitmap to save the screenshot
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)

    # Capture from (start_x, start_y) with the specified width and height
    saveDC.BitBlt((0, 0), (width, height), mfcDC, (start_x - left, start_y - top), win32con.SRCCOPY)

    # Save bitmap to file
    saveBitMap.SaveBitmapFile(saveDC, save_path)
    #print(f"Screenshot saved at: {save_path}")

    # Cleanup
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

screenshot_window_by_title("pygame window", r"C:\Users\Benson\OPP\ML\screenshot\image.png")