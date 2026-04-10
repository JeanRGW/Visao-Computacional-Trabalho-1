import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List

import customtkinter as ctk
from PIL import Image

from panorama import PanoramaResult, run_all_combinations


class VisionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Visao Computacional - Trabalho Pratico 1")
        self.geometry("1200x760")
        self.minsize(1050, 650)

        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.default_input_dir = os.path.join(self.base_dir, "assets", "input")
        self.default_output_dir = os.path.join(self.base_dir, "assets", "output")

        os.makedirs(self.default_input_dir, exist_ok=True)
        os.makedirs(self.default_output_dir, exist_ok=True)

        self.img1_path_var = tk.StringVar(value=os.path.join(self.default_input_dir, "img1.jpg"))
        self.img2_path_var = tk.StringVar(value=os.path.join(self.default_input_dir, "img2.jpg"))
        self.output_dir_var = tk.StringVar(value=self.default_output_dir)
        self.status_var = tk.StringVar(value="Pronto. Selecione as imagens e execute a panoramica.")

        self.panorama_results: List[PanoramaResult] = []
        self.preview_images: List[ctk.CTkImage] = []

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._build_layout()

    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, width=320, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(14, weight=1)

        title_label = ctk.CTkLabel(
            sidebar,
            text="Painel de Controle",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        title_label.grid(row=0, column=0, padx=18, pady=(22, 12), sticky="w")

        appearance_label = ctk.CTkLabel(sidebar, text="Tema")
        appearance_label.grid(row=1, column=0, padx=18, pady=(8, 4), sticky="w")

        self.appearance_menu = ctk.CTkOptionMenu(
            sidebar,
            values=["Dark", "Light", "System"],
            command=self._on_appearance_change,
        )
        self.appearance_menu.set("Dark")
        self.appearance_menu.grid(row=2, column=0, padx=18, pady=(0, 12), sticky="ew")

        ctk.CTkLabel(sidebar, text="Imagem 1").grid(row=3, column=0, padx=18, pady=(8, 2), sticky="w")
        self.img1_entry = ctk.CTkEntry(sidebar, textvariable=self.img1_path_var)
        self.img1_entry.grid(row=4, column=0, padx=18, pady=2, sticky="ew")
        ctk.CTkButton(sidebar, text="Selecionar Imagem 1", command=self._browse_img1).grid(
            row=5, column=0, padx=18, pady=(2, 8), sticky="ew"
        )

        ctk.CTkLabel(sidebar, text="Imagem 2").grid(row=6, column=0, padx=18, pady=(4, 2), sticky="w")
        self.img2_entry = ctk.CTkEntry(sidebar, textvariable=self.img2_path_var)
        self.img2_entry.grid(row=7, column=0, padx=18, pady=2, sticky="ew")
        ctk.CTkButton(sidebar, text="Selecionar Imagem 2", command=self._browse_img2).grid(
            row=8, column=0, padx=18, pady=(2, 8), sticky="ew"
        )

        ctk.CTkLabel(sidebar, text="Diretorio de saida").grid(row=9, column=0, padx=18, pady=(4, 2), sticky="w")
        self.output_dir_entry = ctk.CTkEntry(sidebar, textvariable=self.output_dir_var)
        self.output_dir_entry.grid(row=10, column=0, padx=18, pady=2, sticky="ew")
        ctk.CTkButton(sidebar, text="Selecionar Pasta de Saida", command=self._browse_output_dir).grid(
            row=11, column=0, padx=18, pady=(2, 10), sticky="ew"
        )

        self.run_btn = ctk.CTkButton(
            sidebar,
            text="Gerar Panoramicas (4 combinacoes)",
            height=38,
            command=self._run_panorama,
        )
        self.run_btn.grid(row=12, column=0, padx=18, pady=(4, 8), sticky="ew")

        self.gesture_btn = ctk.CTkButton(
            sidebar,
            text="Abrir Interface Gestual",
            height=36,
            fg_color="#2f8f46",
            hover_color="#236b34",
            command=self._run_gesture,
        )
        self.gesture_btn.grid(row=13, column=0, padx=18, pady=(0, 12), sticky="ew")

        self.quit_btn = ctk.CTkButton(
            sidebar,
            text="Fechar Aplicacao",
            height=34,
            fg_color="#c6403a",
            hover_color="#9f312d",
            command=self.destroy,
        )
        self.quit_btn.grid(row=15, column=0, padx=18, pady=18, sticky="ew")

        main = ctk.CTkFrame(self)
        main.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(2, weight=1)

        header = ctk.CTkFrame(main, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Resultados",
            font=ctk.CTkFont(size=26, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            header,
            text="Panoramica e controle gestual em uma interface unica",
            text_color=("#4b5563", "#9ca3af"),
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        result_card = ctk.CTkFrame(main)
        result_card.grid(row=1, column=0, sticky="ew", padx=14, pady=(2, 8))
        result_card.grid_columnconfigure(0, weight=1)

        columns = [
            "Combinacao",
            "Tempo(s)",
            "KP img1",
            "KP img2",
            "Matches",
            "Inliers",
            "Inlier Ratio",
            "Arquivo",
        ]

        self.results_table = ctk.CTkTextbox(result_card, height=230, wrap="none")
        self.results_table.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.results_table.insert("1.0", " | ".join(columns) + "\n")
        self.results_table.configure(state="disabled")

        preview_card = ctk.CTkFrame(main)
        preview_card.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 8))
        preview_card.grid_columnconfigure(0, weight=1)
        preview_card.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            preview_card,
            text="Visualizacao dos panoramas",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 6))

        self.preview_frame = ctk.CTkScrollableFrame(preview_card, height=250)
        self.preview_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.preview_frame.grid_columnconfigure((0, 1), weight=1)

        controls_card = ctk.CTkFrame(main)
        controls_card.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 8))
        controls_card.grid_columnconfigure((0, 1), weight=1)

        self.open_output_btn = ctk.CTkButton(
            controls_card,
            text="Abrir Pasta de Saida",
            height=36,
            command=self._open_output_dir,
        )
        self.open_output_btn.grid(row=0, column=0, padx=(12, 8), pady=12, sticky="ew")

        self.clear_btn = ctk.CTkButton(
            controls_card,
            text="Limpar Resultados",
            height=36,
            fg_color="#7c8693",
            hover_color="#616b76",
            command=self._clear_results,
        )
        self.clear_btn.grid(row=0, column=1, padx=(0, 12), pady=12, sticky="ew")

        status_bar = ctk.CTkFrame(main)
        status_bar.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 10))
        status_bar.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(status_bar, textvariable=self.status_var, anchor="w").grid(
            row=0, column=0, padx=10, pady=8, sticky="ew"
        )

    def _on_appearance_change(self, mode: str):
        ctk.set_appearance_mode(mode.lower())

    def _browse_img1(self):
        path = filedialog.askopenfilename(
            title="Selecione a Imagem 1",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
        )
        if path:
            self.img1_path_var.set(path)

    def _browse_img2(self):
        path = filedialog.askopenfilename(
            title="Selecione a Imagem 2",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
        )
        if path:
            self.img2_path_var.set(path)

    def _browse_output_dir(self):
        path = filedialog.askdirectory(title="Selecione a pasta de saida")
        if path:
            self.output_dir_var.set(path)

    def _set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        self.run_btn.configure(state=state)
        self.gesture_btn.configure(state=state)

    def _run_panorama(self):
        img1_path = self.img1_path_var.get().strip()
        img2_path = self.img2_path_var.get().strip()
        output_dir = self.output_dir_var.get().strip() or self.default_output_dir

        if not img1_path or not img2_path:
            messagebox.showwarning("Dados incompletos", "Selecione as duas imagens antes de executar.")
            return

        self.status_var.set("Processando panoramicas... aguarde.")
        self._set_busy(True)

        def worker():
            try:
                os.makedirs(output_dir, exist_ok=True)
                results = run_all_combinations(img1_path, img2_path, output_dir)
                self.after(0, lambda: self._on_panorama_success(results))
            except Exception as exc:
                self.after(0, lambda: self._on_error(f"Falha ao gerar panoramicas: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_panorama_success(self, results: List[PanoramaResult]):
        self.panorama_results = results
        self._refresh_table()
        self._refresh_previews()
        self._print_report_data()
        self._set_busy(False)
        self.status_var.set("Panoramicas geradas com sucesso.")
        messagebox.showinfo("Concluido", "As 4 combinacoes de panoramica foram processadas.")

    def _run_gesture(self):
        self._set_busy(True)
        self.status_var.set("Abrindo interface gestual. Use Q para encerrar a janela da webcam.")

        def worker():
            try:
                gesture_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture.py")
                completed = subprocess.run(
                    [sys.executable, gesture_script],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if completed.returncode != 0:
                    error_msg = completed.stderr.strip() or completed.stdout.strip() or "erro desconhecido"
                    self.after(0, lambda: self._on_error(f"Falha na interface gestual: {error_msg}"))
                    return

                self.after(0, self._on_gesture_closed)
            except Exception as exc:
                self.after(0, lambda: self._on_error(f"Falha na interface gestual: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_gesture_closed(self):
        self._set_busy(False)
        self.status_var.set("Interface gestual encerrada.")

    def _refresh_previews(self):
        for child in self.preview_frame.winfo_children():
            child.destroy()

        self.preview_images = []
        if not self.panorama_results:
            ctk.CTkLabel(self.preview_frame, text="Nenhum panorama gerado ainda.").grid(
                row=0, column=0, padx=10, pady=10, sticky="w"
            )
            return

        for index, result in enumerate(self.panorama_results):
            row = index // 2
            col = index % 2

            card = ctk.CTkFrame(self.preview_frame)
            card.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
            card.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                card,
                text=f"{result.combo_name} ({result.elapsed_seconds:.4f}s)",
                font=ctk.CTkFont(size=14, weight="bold"),
            ).grid(row=0, column=0, padx=8, pady=(8, 4), sticky="w")

            preview = self._build_preview_image(result.output_path, (430, 220))
            if preview is None:
                ctk.CTkLabel(card, text="Falha ao carregar imagem.").grid(
                    row=1, column=0, padx=8, pady=8, sticky="w"
                )
            else:
                self.preview_images.append(preview)
                ctk.CTkLabel(card, text="", image=preview).grid(row=1, column=0, padx=8, pady=8, sticky="nsew")

            ctk.CTkLabel(
                card,
                text=f"Inlier ratio: {result.inlier_ratio:.4f}",
                text_color=("#4b5563", "#9ca3af"),
            ).grid(row=2, column=0, padx=8, pady=(0, 8), sticky="w")

    def _build_preview_image(self, image_path: str, max_size):
        try:
            pil_image = Image.open(image_path)
            pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
            return ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=pil_image.size)
        except Exception:
            return None

    def _print_report_data(self):
        print("\n" + "=" * 90)
        print("DADOS PARA RELATORIO (MONTAGEM MANUAL)")
        print("=" * 90)
        print(f"Imagem 1: {self.img1_path_var.get().strip()}")
        print(f"Imagem 2: {self.img2_path_var.get().strip()}")
        print(f"Diretorio de saida: {self.output_dir_var.get().strip()}")
        print("\nTabela comparativa:")
        print("Combinacao     | Tempo(s) | KP1    | KP2    | Matches | Inliers | InlierRatio | Arquivo")
        print("-" * 128)
        for r in self.panorama_results:
            print(
                f"{r.combo_name:<14}| {r.elapsed_seconds:>8.4f} | {r.num_keypoints_img1:>6} | {r.num_keypoints_img2:>6} | "
                f"{r.num_good_matches:>7} | {r.num_inliers:>7} | {r.inlier_ratio:>10.4f} | {r.output_path}"
            )
        print("=" * 90 + "\n")

    def _refresh_table(self):
        self.results_table.configure(state="normal")
        self.results_table.delete("1.0", "end")

        header = (
            f"{'Combinacao':<13} | {'Tempo(s)':>8} | {'KP1':>6} | {'KP2':>6} | {'Matches':>8} | "
            f"{'Inliers':>8} | {'Ratio':>7} | Arquivo\n"
            + "-" * 120
            + "\n"
        )
        self.results_table.insert("1.0", header)

        for r in self.panorama_results:
            row = (
                f"{r.combo_name:<13} | {r.elapsed_seconds:>8.4f} | {r.num_keypoints_img1:>6} | {r.num_keypoints_img2:>6} | "
                f"{r.num_good_matches:>8} | {r.num_inliers:>8} | {r.inlier_ratio:>7.4f} | {r.output_path}\n"
            )
            self.results_table.insert("end", row)

        self.results_table.configure(state="disabled")

    def _open_output_dir(self):
        path = self.output_dir_var.get().strip() or self.default_output_dir
        os.makedirs(path, exist_ok=True)
        try:
            os.startfile(path)
            self.status_var.set(f"Pasta aberta: {path}")
        except Exception as exc:
            self._on_error(f"Nao foi possivel abrir a pasta: {exc}")

    def _clear_results(self):
        self.panorama_results = []
        self.results_table.configure(state="normal")
        self.results_table.delete("1.0", "end")
        self.results_table.insert(
            "1.0",
            "Combinacao | Tempo(s) | KP img1 | KP img2 | Matches | Inliers | Inlier Ratio | Arquivo\n",
        )
        self.results_table.configure(state="disabled")
        self._refresh_previews()
        self.status_var.set("Resultados limpos.")

    def _on_error(self, message: str):
        self._set_busy(False)
        self.status_var.set(message)
        messagebox.showerror("Erro", message)


def run_gui():
    app = VisionApp()
    app.mainloop()
