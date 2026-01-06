# YOLOv8 Segmentation

## Przygotowanie Danych
Przed rozpoczęciem treningu modelu YOLOv8, należy koniecznie przygotować dane do odpowiedniego formatu (YOLO polygon format).

Aby to zrobić, uruchom skrypt przygotowawczy z katalogu `src/segmentation/yolo_8_seg`:

```bash
python datasets_prepair/prepare.py
```

Skrypt ten przetworzy obrazy i maski, tworząc strukturę katalogów wymaganą przez YOLOv8 w `datasets/yolo_seg_data`.

## Trening
Po przygotowaniu danych można rozpocząć trening modelu:

```bash
python train.py
```
