#!/usr/bin/env python3
import json
import os
import tempfile
from pathlib import Path

import gradio as gr

from occupancy_cpu import run_occupancy


def process_video(video_file, line_y, max_distance, max_missed):
    if not video_file:
        raise gr.Error("Sube un video para procesar.")

    input_path = Path(video_file)
    if not input_path.exists():
        raise gr.Error("El archivo de video no existe.")

    with tempfile.TemporaryDirectory(prefix="occupancy_") as tmpdir:
        tmp = Path(tmpdir)
        output_path = tmp / "output.mp4"
        report_path = tmp / "report.json"

        report = run_occupancy(
            input_path=str(input_path),
            output_path=str(output_path),
            report_path=str(report_path),
            line_y=int(line_y),
            max_distance=float(max_distance),
            max_missed=int(max_missed),
        )

        permanent_output = Path("cpu_fallback") / f"web_output_{os.getpid()}.mp4"
        permanent_report = Path("cpu_fallback") / f"web_report_{os.getpid()}.json"
        permanent_output.parent.mkdir(parents=True, exist_ok=True)

        output_bytes = output_path.read_bytes()
        report_bytes = report_path.read_bytes()
        permanent_output.write_bytes(output_bytes)
        permanent_report.write_bytes(report_bytes)

    summary = (
        f"Entradas: {report['in']}\n"
        f"Salidas: {report['out']}\n"
        f"Ocupacion: {report['occupancy']}\n"
        f"Frames: {report['frames']}"
    )

    return str(permanent_output), summary, json.dumps(report, indent=2)


def build_app():
    with gr.Blocks(title="Occupancy Analytics CPU") as demo:
        gr.Markdown("# Occupancy Analytics (CPU)")
        gr.Markdown(
            "Sube un video, define la linea horizontal de cruce y ejecuta el conteo de entrada/salida."
        )

        with gr.Row():
            video_input = gr.Video(label="Video de entrada", sources=["upload"])
            video_output = gr.Video(label="Video procesado")

        with gr.Row():
            line_y = gr.Slider(0, 1080, value=540, step=1, label="Linea de cruce (Y)")
            max_distance = gr.Slider(20, 200, value=90, step=1, label="Distancia max de tracking")
            max_missed = gr.Slider(1, 40, value=12, step=1, label="Frames perdidos max")

        run_btn = gr.Button("Procesar video", variant="primary")
        summary_box = gr.Textbox(label="Resumen", lines=6)
        json_box = gr.Code(label="Reporte JSON", language="json")

        run_btn.click(
            fn=process_video,
            inputs=[video_input, line_y, max_distance, max_missed],
            outputs=[video_output, summary_box, json_box],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
