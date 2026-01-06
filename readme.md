# System Analizy Pęknięć Powierzchni Budowlanych - Praca Inżynierska
## Streszczenie
Praca przedstawia projekt i implementację zintegrowanego systemu analizy pęknięć powierzchni budowlanych, wykorzystującego metody komputerowego przetwarzania obrazu oraz uczenia maszynowego. Głównym celem pracy jest automatyzacja procesu inspekcji technicznej poprzez zastosowanie technik uczenia zespołowego (Ensemble Learning) w celu redukcji wariancji błędu pojedynczych estymatorów.

Architektura systemu opiera się na czteroelementowym potoku przetwarzania:
1. **Kontroler Domeny**: Autorska sieć CNN filtrująca obrazy bez uszkodzeń.
2. **Segmentacja**: Fuzja wyników (U-Net, SegFormer, YOLOv8-seg) metodą Soft-Voting.
3. **Klasyfikacja**: Zespół sieci (EfficientNet-B0, ConvNeXt) określający stopień degradacji.
4. **Analiza Geometryczna**: Obliczanie metryk (długość, szerokość, pole, krętość) algorytmami szkieletyzacji i EDT.

System zaimplementowano w Pythonie (PyTorch, CUDA), a interfejs użytkownika w architekturze klient-serwer (Flask). Badania wykazały, że podejście hybrydowe przewyższa skutecznością pojedyncze rozwiązania SOTA.

W celu pobrania pracy naukowej należy przejść [Tutaj](assets/paper/paper.pdf)

## Instalacja i Konfiguracja

Aby uruchomić projekt, należy przygotować środowisko Python oraz zainstalować zależności.

```bash
# 1. Tworzenie wirtualnego środowiska (zalecane)
python3 -m venv venv
source venv/bin/activate

# 2. Instalacja wymaganych bibliotek
pip install -r requirements.txt
```

### Konfiguracja Ścieżek
Domyślna konfiguracja zakłada standardową strukturę katalogów. W przypadku zmiany lokalizacji modeli lub danych:
- **Modele**: Edytuj `src/final_prediction_pipeline/prediction.py` i zaktualizuj zmienne `SEGFORMER_PATH`, `UNET_PATH`, `YOLO_PATH` itp.
- **Dane/Inferencja**: Zaktualizuj odpowiednie pliki `hparams.py` w podkatalogach modułów.

## Uruchomienie Aplikacji

Aplikacja posiada interfejs webowy umożliwiający łatwą analizę zdjęć.

```bash
# Uruchomienie serwera deweloperskiego
python3 src/server/app.py
```

Po uruchomieniu interfejs dostępny jest pod adresem: [http://localhost:5000](http://localhost:5000)

## Struktura Projektu i Zasoby

| Katalog | Opis | Zasoby (Wymagane pobranie) |
| :--- | :--- | :--- |
| `src/` | Kod źródłowy aplikacji i skrypty treningowe. | |
| `models/` | Wagi wytrenowanych sieci neuronowych. | [Instrukcja i Pobieranie](models/readme.md) |
| `datasets/` | Zbiory danych treningowych i testowych. | [Instrukcja i Pobieranie](datasets/readme.md) |
| `assets/` | Materiały dodatkowe (paper, logi). | |
| `assets/training_logs/` | Logi z procesu uczenia (TensorBoard). | [Instrukcja i Pobieranie](assets/training_logs/readme.md) |

---
***Autor: Mateusz Szczęśniak***
