"""Pakiet pomocniczy do projektu klasyfikacji mikroskopowych obrazow grzybow.

Wspolny kod uzywany przez skrypty (scripts/), notatniki (notebooks/) oraz
interfejs (app.py). Wszystkie eksperymenty korzystaja z tego samego pipeline'u
danych, treningu i ewaluacji -- rozni je tylko plik konfiguracyjny
(configs/*.yaml), co czyni porownania uczciwymi i odtwarzalnymi.
"""
from .config import Config
from .seed import set_seed

__all__ = ["Config", "set_seed"]
