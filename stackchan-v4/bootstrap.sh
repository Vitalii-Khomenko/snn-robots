#!/usr/bin/env bash
# One-time setup of a clean clone for the firmware (2026-10-28, an audit: build.sh sourced an ESP-IDF that only an
# earlier machine's cache had): the builder image, then ESP-IDF v5.5.4 and its esp32s3 tools in ../.tooling (~10 GB,
# resumable). Safe to run again: what is there is reused.
#   ./bootstrap.sh            then: ./build.sh build && ./flash.sh
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
repo_root="$(cd .. && pwd)"
idf_version="v5.5.4"

if ! docker image inspect stackchan-firmware-builder:local >/dev/null 2>&1; then
  echo "building the builder image (firmware-builder/Dockerfile)..."
  docker build -t stackchan-firmware-builder:local "$repo_root/firmware-builder"
fi
mkdir -p "$repo_root/.tooling"

docker run --rm \
  -v "$repo_root/.tooling:/cache" \
  -e IDF_TOOLS_PATH=/cache/espressif \
  -e IDF_VERSION="$idf_version" \
  --dns "${FIRMWARE_DNS_SERVER:-192.168.8.1}" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  stackchan-firmware-builder:local \
  bash -c '
    set -Eeuo pipefail
    root=/cache/esp-idf-$IDF_VERSION
    zip=/cache/esp-idf-$IDF_VERSION.zip
    url=${IDF_SOURCE_ARCHIVE_URL:-https://dl.espressif.com/github_assets/espressif/esp-idf/releases/download/$IDF_VERSION/esp-idf-$IDF_VERSION.zip}
    if [[ ! -f $root/tools/idf_tools.py ]]; then
      if ! unzip -tq "$zip" >/dev/null 2>&1; then
        echo "downloading ESP-IDF $IDF_VERSION (resumable): $url"
        curl --location --fail --show-error --retry 20 --retry-delay 5 --retry-all-errors --continue-at - --output "$zip" "$url"
      fi
      tmp=$(mktemp -d /cache/esp-idf-extract.XXXXXX)
      unzip -q "$zip" -d "$tmp"
      mv "$tmp/esp-idf-$IDF_VERSION" "$root"
      rmdir "$tmp"
    fi
    echo "ESP-IDF source: $root"
    "$root/install.sh" esp32s3
  '
echo "ready: ./build.sh build"
