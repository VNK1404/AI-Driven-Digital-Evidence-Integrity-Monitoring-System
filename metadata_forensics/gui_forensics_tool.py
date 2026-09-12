#!/usr/bin/env python3
"""
=============================================================================
  gui_forensics_tool.py — Visual Interface for Metadata Forensics
=============================================================================
Provides a clean, desktop graphical user interface (GUI) to upload/select
an evidence file and instantly view the forensic score, anomaly detection
results, and extracted metadata flags.
"""

import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading

from pipeline.metadata_pipeline import run_metadata_pipeline

class ForensicsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Metadata Forensics System")
        self.root.geometry("800x600")
        self.root.configure(padx=20, pady=20)
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", padding=6, font=('Segoe UI', 10))
        style.configure("Header.TLabel", font=('Segoe UI', 16, 'bold'))
        style.configure("Score.TLabel", font=('Segoe UI', 24, 'bold'))
        style.configure("Alert.TLabel", font=('Segoe UI', 14, 'bold'), foreground='red')
        style.configure("Safe.TLabel", font=('Segoe UI', 14, 'bold'), foreground='green')

        # Header
        self.header = ttk.Label(root, text="Metadata Forensics Analysis Hub", style="Header.TLabel")
        self.header.pack(pady=(0, 20))

        # Upload Frame
        self.upload_frame = ttk.LabelFrame(root, text=" 1. Select Evidence File ", padding=15)
        self.upload_frame.pack(fill="x", pady=10)

        self.btn_select = ttk.Button(self.upload_frame, text="Browse File...", command=self.select_file)
        self.btn_select.pack(side="left", padx=10)

        self.btn_analyze = ttk.Button(self.upload_frame, text="Run Analysis", command=self.run_analysis, state="disabled")
        self.btn_analyze.pack(side="right", padx=10)

        self.lbl_filepath = ttk.Label(self.upload_frame, text="No file selected", foreground="gray", anchor="w")
        self.lbl_filepath.pack(side="left", padx=10, fill="x", expand=True)

        # Progress bar
        self.progress = ttk.Progressbar(root, mode='indeterminate')
        self.progress.pack(fill="x", pady=10)

        # Results Frame
        self.results_frame = ttk.LabelFrame(root, text=" 2. Analysis Results ", padding=15)
        self.results_frame.pack(fill="both", expand=True, pady=10)

        # Score Area (Top of results)
        self.score_frame = ttk.Frame(self.results_frame)
        self.score_frame.pack(fill="x", pady=10)

        self.lbl_score_text = ttk.Label(self.score_frame, text="Integrity Score:", font=('Segoe UI', 12))
        self.lbl_score_text.grid(row=0, column=0, sticky="w", padx=10)
        self.lbl_score_val = ttk.Label(self.score_frame, text="---", style="Score.TLabel")
        self.lbl_score_val.grid(row=0, column=1, sticky="w", padx=10)



        # Details Area (Bottom of results)
        self.details_notebook = ttk.Notebook(self.results_frame)
        self.details_notebook.pack(fill="both", expand=True, pady=10)

        # Tab 1: Flags
        self.tab_flags = ttk.Frame(self.details_notebook)
        self.details_notebook.add(self.tab_flags, text="Suspicious Flags")
        self.txt_flags = tk.Text(self.tab_flags, wrap="word", height=8, font=('Consolas', 10), bg="#f5f5f5")
        self.txt_flags.pack(fill="both", expand=True, padx=5, pady=5)

        # Tab 2: Raw Metadata
        self.tab_meta = ttk.Frame(self.details_notebook)
        self.details_notebook.add(self.tab_meta, text="Raw Extracted Metadata")
        self.txt_meta = tk.Text(self.tab_meta, wrap="none", height=8, font=('Consolas', 10), bg="#f5f5f5")
        
        # Scrollbars for raw metadata
        v_scroll = ttk.Scrollbar(self.tab_meta, orient="vertical", command=self.txt_meta.yview)
        h_scroll = ttk.Scrollbar(self.tab_meta, orient="horizontal", command=self.txt_meta.xview)
        self.txt_meta.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        v_scroll.pack(side="right", fill="y")
        h_scroll.pack(side="bottom", fill="x")
        self.txt_meta.pack(side="left", fill="both", expand=True)

        self.selected_file = None

    def select_file(self):
        filetypes = (
            ('All supported media', '*.jpg *.jpeg *.png *.tif *.bmp *.mp4 *.avi *.mov *.mkv *.wav *.mp3 *.flac *.pdf'),
            ('Images', '*.jpg *.jpeg *.png *.tif *.bmp'),
            ('Videos', '*.mp4 *.avi *.mov *.mkv'),
            ('Audio', '*.wav *.mp3 *.flac'),
            ('Documents', '*.pdf'),
            ('All files', '*.*')
        )
        
        filename = filedialog.askopenfilename(
            title='Select Digital Evidence',
            initialdir=os.path.join(os.path.dirname(__file__), "datasets"),
            filetypes=filetypes
        )
        
        if filename:
            self.selected_file = filename
            self.lbl_filepath.config(text=filename, foreground="black")
            self.btn_analyze.config(state="normal")
            
            # Reset UI
            self.lbl_score_val.config(text="---", foreground="black")
            self.txt_flags.delete(1.0, tk.END)
            self.txt_meta.delete(1.0, tk.END)

    def run_analysis(self):
        if not self.selected_file:
            return
            
        self.btn_analyze.config(state="disabled")
        self.btn_select.config(state="disabled")
        self.progress.start(15)
        self.lbl_score_val.config(text="Analyzing...", foreground="gray")
        
        # Run in a separate thread to keep UI responsive
        threading.Thread(target=self._process_file, daemon=True).start()

    def _process_file(self):
        try:
            # The core processing!
            result = run_metadata_pipeline(self.selected_file)
            
            # Update UI on the main thread safely
            self.root.after(0, self._show_results, result)
        except Exception as e:
            self.root.after(0, self._show_error, str(e))

    def _show_results(self, result):
        self.progress.stop()
        self.btn_analyze.config(state="normal")
        self.btn_select.config(state="normal")
        
        # Display Score
        score = result.get('metadata_score', 0)
        self.lbl_score_val.config(text=f"{score} / 100")
        if score >= 80:
            self.lbl_score_val.config(foreground="green")
        elif score >= 50:
            self.lbl_score_val.config(foreground="orange")
        else:
            self.lbl_score_val.config(foreground="red")
            
        # Display Anomaly status — removed from UI (metadata forensics cannot
        # detect AI-generated video content; anomaly penalty still counts in score)

        # Display Flags
        self.txt_flags.delete(1.0, tk.END)
        all_flags = result.get("flags", []) + result.get("timeline_flags", [])
        if all_flags:
            flag_text = "\n".join([f"🚩 {f.replace('_', ' ').capitalize()}" for f in all_flags])
            self.txt_flags.insert(tk.END, flag_text)
        else:
            self.txt_flags.insert(tk.END, "✅ No suspicious flags detected.\nAll rule-based checks passed.")
            
        # Display Raw Metadata
        self.txt_meta.delete(1.0, tk.END)
        try:
            meta_str = json.dumps(result.get("metadata", {}), indent=4, default=str)
            self.txt_meta.insert(tk.END, meta_str)
        except Exception as e:
            self.txt_meta.insert(tk.END, f"Error displaying metadata: {e}")

    def _show_error(self, error_msg):
        self.progress.stop()
        self.btn_analyze.config(state="normal")
        self.btn_select.config(state="normal")
        self.lbl_score_val.config(text="ERROR", foreground="red")
        messagebox.showerror("Analysis Error", f"An error occurred during analysis:\n\n{error_msg}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ForensicsApp(root)
    root.mainloop()
