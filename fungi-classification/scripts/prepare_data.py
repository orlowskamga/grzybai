"""Buduje ujednolicony manifest z dwoch zbiorow (DeFungi + OpenFungi-micro).

Manifest (data/manifest.csv) ma kolumny: path, unified_label, source, group.
Nie kopiuje obrazow -- wskazuje na oryginalne pliki (oszczedza miejsce).

Mapowanie folderow na wspolne etykiety jest w configs/class_mapping.yaml.
Najpierw uruchom z --inspect, by zobaczyc nazwy folderow i przyklady plikow,
potem uzupelnij mapping (zwlaszcza nazwy klas OpenFungi i regex grupy DeFungi).

Przyklady:
    python scripts/prepare_data.py --inspect
    python scripts/prepare_data.py --mapping configs/class_mapping.yaml --out data/manifest.csv
"""
import _bootstrap  # noqa: F401
import argparse
import os
import re
import csv
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(root):
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in IMG_EXT:
                yield os.path.join(dirpath, fn)


def class_folder(root, path):
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    return parts[0] if len(parts) > 1 else "_root_"


def extract_group(path, regex):
    stem = os.path.splitext(os.path.basename(path))[0]
    if not regex:
        return stem
    m = re.search(regex, stem)
    return m.group(1) if (m and m.groups()) else stem


def inspect(mapping):
    for source, spec in mapping.items():
        root = spec.get("root", "")
        print(f"\n=== {source}  (root: {root}) ===")
        if not root or not os.path.isdir(root):
            print("  [!] katalog nie istnieje -- popraw 'root' w mapping")
            continue
        folders = {}
        for p in iter_images(root):
            folders.setdefault(class_folder(root, p), []).append(p)
        for folder, paths in sorted(folders.items()):
            ex = [os.path.basename(x) for x in paths[:3]]
            print(f"  {folder:<28} {len(paths):>6} obrazow   przyklady: {ex}")


def build(mapping, out_path):
    rows = []
    stats = {}
    for source, spec in mapping.items():
        root = spec.get("root", "")
        regex = spec.get("group_regex")
        classes = spec.get("classes", {}) or {}
        if not os.path.isdir(root):
            print(f"[{source}] pomijam -- brak katalogu: {root}")
            continue
        unmapped = set()
        for p in iter_images(root):
            folder = class_folder(root, p)
            label = classes.get(folder)
            if label is None:
                unmapped.add(folder)
                continue
            # Klucz grupy zawiera folder klasy -> patche z jednego preparatu sa
            # razem, ale nazwy plikow powtarzajace sie miedzy klasami nie koliduja.
            rows.append({
                "path": os.path.abspath(p),
                "unified_label": label,
                "source": source,
                "group": f"{source}:{folder}:{extract_group(p, regex)}",
            })
            stats[label] = stats.get(label, 0) + 1
        if unmapped:
            print(f"[{source}] foldery bez mapowania (pominiete): {sorted(unmapped)}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "unified_label", "source", "group"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nZapisano {len(rows)} obrazow -> {out_path}")
    print("Rozklad klas (po scaleniu):")
    for label, n in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<28} {n:>6}")
    n_groups = len({r["group"] for r in rows})
    print(f"Liczba grup (preparatow/obrazow zrodlowych): {n_groups}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default="configs/class_mapping.yaml")
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--inspect", action="store_true",
                    help="tylko wypisz foldery i przyklady plikow")
    args = ap.parse_args()

    with open(args.mapping, "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)

    if args.inspect:
        inspect(mapping)
    else:
        build(mapping, args.out)


if __name__ == "__main__":
    main()
