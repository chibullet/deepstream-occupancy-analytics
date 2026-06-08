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
  --line-y 540
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
- El conteo se hace al cruzar la línea horizontal `line-y`:
  - Arriba -> abajo: `IN`
  - Abajo -> arriba: `OUT`
