# Praca Inżynierska

Praca przedstawia projekt i implementację zintegrowanego systemu analizy pęknięć powierzchni budowlanych, wykorzystującego metody komputerowego przetwarzania obrazu oraz uczenia maszynowego. Głównym celem pracy jest automatyzacja procesu inspekcji technicznej poprzez zastosowanie technik uczenia zespołowego (Ensemble Learning) w celu redukcji wariancji błędu pojedynczych estymatorów. Architektura systemu opiera się na czteroelementowym potoku przetwarzania. Pierwszy etap stanowi Kontroler Domeny, realizowany przez autorską konwolucyjną sieć neuronową (CNN), filtrującą obrazy niezawierające uszkodzeń. Etap segmentacji wykorzystuje fuzję wyników modeli U-Net, SegFormer oraz YOLOv8-seg przy użyciu metody głosowania miękkiego (soft-voting). Klasyfikacja stopnia degradacji (włosowe, małe, średnie, duże) realizowana jest przez zespół sieci EfficientNet-B0 oraz ConvNeXt. Ostatni moduł wykonuje analizę geometryczną (długość, szerokość, pole powierzchni, krętość) z wykorzystaniem algorytmów szkieletyzacji oraz Euklidesowej Transformaty Odległościowej (EDT). Rozwiązanie zaimplementowano w języku Python z użyciem biblioteki PyTorch i środowiska CUDA, a interfejs użytkownika opracowano w architekturze klient-serwer przy użyciu frameworka Flask. Przeprowadzone badania wykazały, że zastosowanie podejścia hybrydowego zwiększa precyzję detekcji oraz stabilność predykcji względem pojedynczych architektur referencyjnych (SOTA).

## Instalacja i Uruchomienie

Aby projekt działał poprawnie, należy utworzyć odpowiednie środowisko Python (zalecane wirtualne środowisko) i zainstalować wymagane biblioteki.

```bash
# Utworzenie wirtualnego środowiska
python3 -m venv venv
source venv/bin/activate

# Instalacja zależności
pip install -r requirements.txt
```

### Zmiana lokalizacji plików modeli
W przypadku zmiany lokalizacji plików modeli należy edytować plik `src/final_prediction_pipeline/prediction.py` oraz zmienić parametr ścieżek dostępu (zmienne 
SEGFORMER_PATH,UNET_PATH,YOLO_PATH itp.).
Podobnie należy zmienić parametry ścieżek w przypadku zmiany lokalizacji danych (zdjęć testowych) oraz inferencji pojedynczych modeli (pliki hparams.py).

### Uruchomienie Serwera
Aby uruchomić interfejs webowy aplikacji:

```bash
python3 src/server/app.py
```
Serwer będzie dostępny pod adresem: `http://localhost:5000`

## Struktura Projektu

### Kod Źródłowy (`src`)
Główny kod źródłowy aplikacji i skrypty treningowe znajdują się w katalogu `src/`.

### Modele (`models`)
Modele sieci neuronowych niezbędne do działania systemu znajdują się w katalogu `models/`.
**Ważne:** Modele należy pobrać i rozpakować ręcznie. Szczegółowe instrukcje znajdują się w pliku [models/readme.md](models/readme.md).



### Materiały Dodatkowe (`assets`)
Folder `assets/` zawiera dodatkowe materiały, w tym pełny tekst pracy inżynierskiej (paper).

#### Logi Treningowe
W katalogu `assets/training_logs/` przechowywane są logi z procesu uczenia modeli.
Instrukcja pobierania i obsługi logów znajduje się w pliku [assets/training_logs/readme.md](assets/training_logs/readme.md).

--
