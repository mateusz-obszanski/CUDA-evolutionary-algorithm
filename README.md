# Wyspowy algorytm ewolucyjny

Działający kod znajduje się w folderze `dev/`.

## Kompilacja

Wymagany zainstalowany toolkit CUDA 12.1 oraz skonfigurowana zmienna `PATH`.

Kompilacja działa na systemie Ubuntu 22.04 z kompilatorem GCC 11.3.0. Kompilacja na systemie Windows nie działa, nawet po modyfikacjach kodu usuwających błędy kompilatora - kompilator hosta wywołuje `abort`.

Należy przejść do folderu `dev/` i uruchomić skrypt `./build-dev.sh` (wersja CPU) lub `./build-dev-cuda.sh`. Można też uruchomić skrypty z przyrostkiem `-optim`, które kompilują z optymalizacją.

## Uruchomienie aplikacji

W folderze `dev/build/` należy uruchomić program `dev` lub `dev-gpu` z argumentem - należy podać ścieżkę do pliku konfiguracyjnego, przykładowy znajduje się w pliku `algo_cfg.txt`.
Parametry:
- liczba iteracji - pomiędzy migracjami
- liczba epok - liczba migracji
- liczba epok od ostatniej poprawy - jedno z kryterium stopu

Wyniki zapisywane są w folderze `results/` lub `results-gpu/`.

## Rezultaty

Przebieg działania algorytmu można obejrzeć w formie wykresu po uruchomieniu skryptu `./read_data.py`. Należy podać ścieżkę do folderu z danymi wynikowymi konkretnego eksperymentu (zapisywane są z datą i godziną uruchomienia) lub ścieżkę do folderu ze wszystkimi wynikami (np `results-gpu`) z flagą `--latest` - wtedy skrypt wyświetli dane ostatnio przeprowadzonego eksperymentu.

Przykład:

```
python3 ./read_data.py ./results-gpu --latest
```

Wymagana jest dostępność bibliotek numpy, pandas, matplotlib i seaborn.
Skrypt czyta binarne dane i generuje wykresy dla każdej z wysp, po czym wyświetla je po kolei (aby przejść do następnego, należy nacisnąć klawisz `q`), a następnie zapisuje obok danych.

