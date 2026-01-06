# Logi Treningowe

## Pobieranie Danych
W celu analizy procesu uczenia modeli należy pobrać archiwum z logami.

1. Przejdź na stronę: [LINK DO DANYCH](https://drive.google.com/file/d/1oNHk2cY6F8rapmSGEBRtuidnHzdy-gQh/view?usp=sharing)
2. Pobierz plik `training_logs.zip`.
3. Rozpakuj archiwum w katalogu `assets/training_logs`.

## Uruchomienie TensorBoard
Aby wyświetlić wykresy i metryki treningowe, należy użyć narzędzia TensorBoard.

1. Aktywuj środowisko wirtualne Python.
   ```bash
   source venv/bin/activate
   ```
2. Przejdź do katalogu z logami:
   ```bash
   cd assets/training_logs
   ```
3. Uruchom serwer TensorBoard:
   ```bash
   tensorboard --logdir=.
   ```
4. Otwórz przeglądarkę pod adresem wskazanym w terminalu (zazwyczaj `http://localhost:6006`).
