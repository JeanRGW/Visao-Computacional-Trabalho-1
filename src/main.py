import os
from typing import List, Optional

from gesture import GestureSlideController
from gui_app import run_gui
from panorama import PanoramaResult, format_results_table, run_all_combinations


def _ask_path(prompt: str, default: Optional[str] = None) -> str:
    if default:
        value = input(f"{prompt} [{default}]: ").strip()
        return value or default
    return input(f"{prompt}: ").strip()


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def run_cli():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_input_dir = os.path.join(base_dir, "assets", "input")
    default_output_dir = os.path.join(base_dir, "assets", "output")

    _ensure_dir(default_input_dir)
    _ensure_dir(default_output_dir)

    img1_path = ""
    img2_path = ""
    panorama_results: List[PanoramaResult] = []

    while True:
        print("\n" + "=" * 70)
        print("1o Trabalho Pratico - Visao Computacional")
        print("=" * 70)
        print("1) Definir imagens para panoramica")
        print("2) Executar panoramica (4 combinacoes)")
        print("3) Exibir tabela de resultados")
        print("4) Executar interface gestual (webcam)")
        print("0) Sair")

        option = input("Escolha uma opcao: ").strip()

        if option == "1":
            print("\nInforme os caminhos das duas imagens.")
            print("Dica: coloque em assets/input e informe o caminho completo ou relativo.")
            img1_path = _ask_path("Caminho da imagem 1", os.path.join(default_input_dir, "img1.jpg"))
            img2_path = _ask_path("Caminho da imagem 2", os.path.join(default_input_dir, "img2.jpg"))
            print(f"Imagem 1: {img1_path}")
            print(f"Imagem 2: {img2_path}")

        elif option == "2":
            try:
                if not img1_path or not img2_path:
                    print("Defina as imagens primeiro (opcao 1).")
                    continue

                output_dir = _ask_path("Diretorio de saida", default_output_dir)
                _ensure_dir(output_dir)

                panorama_results = run_all_combinations(img1_path, img2_path, output_dir)

                print("\nPanoramicas geradas com sucesso:")
                for r in panorama_results:
                    print(f"- {r.combo_name}: {r.output_path} (tempo: {r.elapsed_seconds:.4f}s)")

            except Exception as exc:
                print(f"Erro ao gerar panoramicas: {exc}")

        elif option == "3":
            if not panorama_results:
                print("Nenhum resultado disponivel. Execute a opcao 2 primeiro.")
                continue
            print("\n" + format_results_table(panorama_results))

        elif option == "4":
            print("\nAbrindo interface gestual.")
            print("Faca gestos para direita/esquerda para navegar slides. Pressione Q para sair.")
            try:
                controller = GestureSlideController()
                controller.run()
            except Exception as exc:
                print(f"Erro na interface gestual: {exc}")

        elif option == "0":
            print("Encerrando programa.")
            break

        else:
            print("Opcao invalida. Tente novamente.")


if __name__ == "__main__":
    import sys

    if "--cli" in sys.argv:
        run_cli()
    else:
        run_gui()
