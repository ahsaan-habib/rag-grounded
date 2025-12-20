#!/usr/bin/env bash
# Pull the docs this project is built around into data/corpus/.
#   laravel/   laravel/docs @ 12.x
#   filament/  filamentphp/filament @ 3.x, packages/<pkg>/docs -> filament/<pkg>/
#   livewire/  livewire/livewire @ 3.x, docs/
set -euo pipefail
dest="${1:-data/corpus}"
mkdir -p "$dest"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

sparse() {  # repo branch dir paths...
  local repo=$1 branch=$2 dir=$3; shift 3
  git clone -q --depth 1 --branch "$branch" --filter=blob:none --sparse \
    "https://github.com/$repo.git" "$dir"
  git -C "$dir" sparse-checkout set "$@"
}

git clone -q --depth 1 --branch 12.x https://github.com/laravel/docs.git "$tmp/laravel"
rm -rf "$dest/laravel" && mkdir -p "$dest/laravel" && cp "$tmp"/laravel/*.md "$dest/laravel/"

pkgs=(panels forms tables actions infolists notifications widgets)
sparse filamentphp/filament 3.x "$tmp/filament" "${pkgs[@]/#/packages/}"
rm -rf "$dest/filament"
for p in "${pkgs[@]}"; do
  mkdir -p "$dest/filament/$p" && cp -r "$tmp/filament/packages/$p/docs/." "$dest/filament/$p/"
done

sparse livewire/livewire 3.x "$tmp/livewire" docs
rm -rf "$dest/livewire" && cp -r "$tmp/livewire/docs" "$dest/livewire"

for d in laravel filament livewire; do
  echo "$d: $(find "$dest/$d" -name '*.md' | wc -l) files"
done
