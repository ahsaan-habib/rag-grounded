#!/usr/bin/env bash
# Pull the docs this project is built around into data/corpus/.
set -euo pipefail
dest="${1:-data/corpus}"
mkdir -p "$dest"

clone() {  # repo branch subdir name
  tmp=$(mktemp -d)
  git clone --depth 1 --branch "$2" --filter=blob:none --sparse "https://github.com/$1.git" "$tmp" >/dev/null
  git -C "$tmp" sparse-checkout set "$3" >/dev/null
  rm -rf "$dest/$4" && cp -r "$tmp/$3" "$dest/$4"
  rm -rf "$tmp"
  echo "$4: $(find "$dest/$4" -name '*.md' | wc -l) files"
}

clone laravel/docs 12.x . laravel
clone filamentphp/filament 3.x docs filament
clone livewire/livewire main docs livewire
rm -rf "$dest/laravel/.git"
