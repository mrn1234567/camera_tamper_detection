"""Non-blocking Tk popup for camera tamper alarms."""

from __future__ import annotations

import tkinter as tk


class CameraAlarmPopup:
    """Show or hide a topmost alarm while the video loop keeps running."""

    def __init__(self, message: str = "摄像头正在被遮挡") -> None:
        self._root: tk.Tk | None = None
        self._visible = False
        try:
            root = tk.Tk()
        except tk.TclError:
            # Headless environments can still run the detector without a popup.
            return

        self._root = root
        root.withdraw()
        root.title("摄像头报警")
        root.geometry("520x130+620+80")
        root.configure(bg="#8b0000")
        root.attributes("-topmost", True)
        root.resizable(False, False)
        tk.Label(
            root,
            text=message,
            bg="#8b0000",
            fg="white",
            font=("Microsoft YaHei UI", 20, "bold"),
            padx=28,
            pady=18,
        ).pack()

    def set_alarm(self, active: bool) -> None:
        """Update popup state; safe to call once per video frame."""
        if self._root is None:
            return
        active = bool(active)
        if active and not self._visible:
            self._root.deiconify()
            self._root.lift()
            self._root.focus_force()
            self._visible = True
        elif not active and self._visible:
            self._root.withdraw()
            self._visible = False
        # Process window messages without waiting for user input.
        self._root.update_idletasks()
        self._root.update()

    def close(self) -> None:
        """Close the popup window."""
        if self._root is not None:
            self._root.destroy()
            self._root = None
            self._visible = False
