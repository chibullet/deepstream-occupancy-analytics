#!/usr/bin/env python3
import json
import os
import tempfile
from pathlib import Path

import gradio as gr

from occupancy_cpu import run_occupancy


def adjust_angle(current_angle, delta):
    v = float(current_angle) + float(delta)
    v = max(-90.0, min(90.0, v))
    return int(round(v / 5.0) * 5)


def process_video(
    video_file,
    line_angle_deg,
    line_center_x_pct,
    line_center_y_pct,
    flow_direction,
    max_distance,
    max_missed,
):
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
            line_angle_deg=float(line_angle_deg),
            line_center_x_ratio=float(line_center_x_pct) / 100.0,
            line_center_y_ratio=float(line_center_y_pct) / 100.0,
            flow_direction=str(flow_direction),
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
        f"Frames: {report['frames']}\n"
        f"Angulo: {report['line_angle_deg']} grados\n"
        f"Flujo: {report['flow_direction']}"
    )

    return str(permanent_output), summary, json.dumps(report, indent=2)


def build_app():
    with gr.Blocks(title="Occupancy Analytics CPU") as demo:
        gr.Markdown("# Occupancy Analytics (CPU)")
        gr.Markdown(
            "Sube un video, configura una linea (horizontal, vertical o inclinada) y define la direccion de flujo."
        )

        with gr.Row():
            video_input = gr.Video(label="Video de entrada", sources=["upload"])
            video_output = gr.Video(label="Video procesado")

        with gr.Row():
            line_angle_deg = gr.Slider(
                -90,
                90,
                value=0,
                step=5,
                label="Angulo de linea (grados): 0=horizontal, 90=vertical",
            )
            flow_direction = gr.Dropdown(
                choices=[
                    "left_to_right",
                    "right_to_left",
                    "top_to_bottom",
                    "bottom_to_top",
                ],
                value="left_to_right",
                label="Direccion de flujo (se cuenta como IN)",
            )

        with gr.Row():
            angle_minus_5 = gr.Button("-5 grados")
            angle_plus_5 = gr.Button("+5 grados")
            angle_h = gr.Button("Horizontal (0)")
            angle_v = gr.Button("Vertical (90)")
            angle_d1 = gr.Button("Diagonal (+45)")
            angle_d2 = gr.Button("Diagonal (-45)")

        with gr.Row():
            flow_lr = gr.Button("IN: izquierda -> derecha")
            flow_rl = gr.Button("IN: derecha -> izquierda")
            flow_tb = gr.Button("IN: arriba -> abajo")
            flow_bt = gr.Button("IN: abajo -> arriba")

        with gr.Row():
            line_center_x_pct = gr.Slider(
                0,
                100,
                value=50,
                step=1,
                label="Posicion X de la linea (%)",
            )
            line_center_y_pct = gr.Slider(
                0,
                100,
                value=50,
                step=1,
                label="Posicion Y de la linea (%)",
            )

        with gr.Row():
            max_distance = gr.Slider(20, 200, value=90, step=1, label="Distancia max de tracking")
            max_missed = gr.Slider(1, 40, value=12, step=1, label="Frames perdidos max")

        run_btn = gr.Button("Procesar video", variant="primary")
        summary_box = gr.Textbox(label="Resumen", lines=6)
        json_box = gr.Code(label="Reporte JSON", language="json")

        run_btn.click(
            fn=process_video,
            inputs=[
                video_input,
                line_angle_deg,
                line_center_x_pct,
                line_center_y_pct,
                flow_direction,
                max_distance,
                max_missed,
            ],
            outputs=[video_output, summary_box, json_box],
        )

        angle_minus_5.click(
            fn=lambda a: adjust_angle(a, -5),
            inputs=[line_angle_deg],
            outputs=[line_angle_deg],
        )
        angle_plus_5.click(
            fn=lambda a: adjust_angle(a, 5),
            inputs=[line_angle_deg],
            outputs=[line_angle_deg],
        )
        angle_h.click(fn=lambda: 0, outputs=[line_angle_deg])
        angle_v.click(fn=lambda: 90, outputs=[line_angle_deg])
        angle_d1.click(fn=lambda: 45, outputs=[line_angle_deg])
        angle_d2.click(fn=lambda: -45, outputs=[line_angle_deg])

        flow_lr.click(fn=lambda: "left_to_right", outputs=[flow_direction])
        flow_rl.click(fn=lambda: "right_to_left", outputs=[flow_direction])
        flow_tb.click(fn=lambda: "top_to_bottom", outputs=[flow_direction])
        flow_bt.click(fn=lambda: "bottom_to_top", outputs=[flow_direction])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
