# CPU fallback (sin DeepStream)

Esta alternativa permite probar conteo de entrada/salida en CPU usando OpenCV,
para equipos sin GPU NVIDIA.

## 1) Instalar dependencias

```bash
cd cpu_fallback
python3 -m pip install -r requirements.txt
```

## 2) Ejecutar sobre un video

```bash
python3 occupancy_cpu.py \
  --input /ruta/a/video.mp4 \
  --output cpu_output.mp4 \
  --report cpu_report.json \
  --line-angle-deg 0 \\
  --line-center-x-ratio 0.5 \\
  --line-center-y-ratio 0.5 \\
  --flow-direction left_to_right
```

## 3) Pagina web para subir videos

```bash
python3 -m pip install -r requirements.txt
python3 web_app.py
```

Abre esta URL en tu navegador:

`http://localhost:7860`

Si corres esto en Codespaces, usa el puerto reenviado 7860.

## Resultado

- Video con detecciones y contador: `cpu_output.mp4`
- Resumen JSON: `cpu_report.json`

## Notas

- Usa detector HOG de OpenCV (CPU), por lo que la precisión/rendimiento es menor
  que DeepStream + TensorRT.
- La línea de cruce se puede configurar con ángulo:
  - `0` = horizontal
  - `90` = vertical
  - valores intermedios = inclinada
- En la web, el ángulo cambia en pasos de 5 grados.
- La dirección elegida en `--flow-direction` se cuenta como `IN`.
  El movimiento contrario se cuenta como `OUT`.
