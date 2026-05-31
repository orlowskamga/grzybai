# Automatyczna klasyfikacja mikroskopowych obrazow grzybow (CNN)

Klasyfikacja mikroskopowych obrazow grzybow na podstawie zbiorow **DeFungi**
i **OpenFungi** (czesc mikroskopowa). Projekt porownuje siec konwolucyjna
trenowana **od zera** z **transfer learningiem** (kilka backbone'ow), z pelna
ewaluacja i interfejsem testowym.

## Struktura

```
src/fungi/        wspolny kod (pipeline danych, modele, trening, ewaluacja, Grad-CAM)
  config.py       konfiguracja eksperymentu (YAML)
  seed.py         powtarzalnosc
  data.py         manifest -> podzial grupowo-stratyfikowany -> DataLoadery
  models.py       wlasny CNN + backbone'y z timm, dwufazowy fine-tuning
  engine.py       petla treningu (wybor modelu wg val macro-F1)
  metrics.py      accuracy/precision/recall/F1, macierz pomylek, ROC/AUC, t-SNE
  explain.py      Grad-CAM (wlasna implementacja, bez zewnetrznych bibliotek)
scripts/          prepare_data.py, train.py, evaluate.py
configs/          po jednym YAML na eksperyment + class_mapping.yaml
notebooks/        cienkie notatniki opakowujace pipeline (do raportu/wykresow)
app.py            interfejs Gradio (predykcja + Grad-CAM)
```

## Decyzje projektowe (skrot)

- **Wspolny klasyfikator** dla obu zbiorow. Etykiety sa ujednolicane przez
  `configs/class_mapping.yaml`. Poniewaz zbiory roznia sie warunkami obrazowania,
  model moglby rozpoznawac *zrodlo* zamiast morfologii -- dlatego:
  - wybor modelu i raporty oparte na **macro-F1** (nie samej accuracy),
  - manifest przechowuje kolumne `source`, a `evaluate.py` liczy wyniki
    **w rozbiciu na zrodlo** (duza dysproporcja = sygnal ostrzegawczy),
  - *Aspergillus niger* (DeFungi) i *A.* sekcja Nigri (OpenFungi) sa domyslnie
    scalone w jedna klase.
- **Podzial grupowo-stratyfikowany** (train/val/test): patche z jednego
  preparatu nie trafiaja do roznych zbiorow (brak przecieku). Wymaga ustawienia
  `group_regex` dla DeFungi w `class_mapping.yaml`.
- **Powtarzalnosc**: jeden seed steruje podzialem, inicjalizacja i DataLoaderami.

## Instalacja

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Uzycie

1. Pobierz zbiory do `data/raw/defungi/` i `data/raw/openfungi/micro/`.
2. Sprawdz nazwy folderow i przyklady plikow, potem popraw `class_mapping.yaml`:
   ```bash
   python scripts/prepare_data.py --inspect
   ```
3. Zbuduj manifest:
   ```bash
   python scripts/prepare_data.py
   ```
4. Trenuj (jeden eksperyment = jeden config):
   ```bash
   python scripts/train.py --config configs/resnet50.yaml
   python scripts/train.py --config configs/customcnn.yaml
   ```
5. Ewaluacja na zbiorze testowym:
   ```bash
   python scripts/evaluate.py --config configs/resnet50.yaml \
       --checkpoint outputs/resnet50/best.pth --tsne
   ```
6. Interfejs testowy:
   ```bash
   python app.py --config configs/resnet50.yaml --checkpoint outputs/resnet50/best.pth
   ```

## Sprzet

Trening na serwerze NVIDIA (CUDA). Dwie karty pozwalaja uruchamiac dwa
eksperymenty rownolegle (po jednym na GPU): `CUDA_VISIBLE_DEVICES=0` i `=1`.
