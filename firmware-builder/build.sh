#!/usr/bin/env bash
set -Eeuo pipefail

idf_version="v5.5.4"
idf_root="/cache/esp-idf-${idf_version}"
idf_archive="/cache/esp-idf-${idf_version}.zip"
idf_archive_url="${IDF_SOURCE_ARCHIVE_URL:-https://dl.espressif.com/github_assets/espressif/esp-idf/releases/download/${idf_version}/esp-idf-${idf_version}.zip}"
tools_root="${IDF_TOOLS_PATH:-/cache/espressif}"
assets_mirror="${IDF_GITHUB_ASSETS:-dl.espressif.com/github_assets}"

export IDF_TOOLS_PATH="$tools_root"
export IDF_GITHUB_ASSETS="$assets_mirror"

mkdir -p /cache "$tools_root"

if [[ ! -f "$idf_root/tools/idf_tools.py" ]]; then
    if [[ -e "$idf_root" ]]; then
        incomplete_root="${idf_root}.incomplete.$(date +%s)"
        echo "Found an incomplete ESP-IDF directory; preserving it as $incomplete_root"
        mv "$idf_root" "$incomplete_root"
    fi

    if ! unzip -tq "$idf_archive" >/dev/null 2>&1; then
        echo "Downloading resumable ESP-IDF source archive $idf_version..."
        echo "Source URL: $idf_archive_url"
        curl --location --fail --show-error \
            --retry 20 --retry-delay 5 --retry-all-errors \
            --continue-at - --output "$idf_archive" "$idf_archive_url"
    else
        echo "Reusing complete ESP-IDF source archive: $idf_archive"
    fi

    if ! unzip -tq "$idf_archive" >/dev/null 2>&1; then
        echo "ESP-IDF source archive is incomplete or corrupt: $idf_archive" >&2
        exit 3
    fi

    extract_root="$(mktemp -d /cache/esp-idf-extract.XXXXXX)"
    echo "Extracting ESP-IDF source..."
    unzip -q "$idf_archive" -d "$extract_root"
    extracted_root="$extract_root/esp-idf-${idf_version}"
    if [[ ! -f "$extracted_root/tools/idf_tools.py" ]]; then
        echo "Unexpected ESP-IDF archive layout: $extracted_root" >&2
        exit 4
    fi
    mv "$extracted_root" "$idf_root"
    echo "ESP-IDF source is ready at $idf_root"
fi

echo "Installing or reusing ESP-IDF tools in $tools_root..."
"$idf_root/install.sh" esp32s3

source "$idf_root/export.sh"

echo "Fetching the official StackChan firmware dependencies..."
python3 ./fetch_repos.py

echo "Building the official StackChan firmware..."
if [[ -f /project/sdkconfig.defaults.local ]]; then
    echo "Applying local sdkconfig overlay for the laptop gateway..."
    # Keep the user's normal sdkconfig untouched. ESP-IDF preserves values
    # already present in sdkconfig, so a separate generated config is needed
    # for the local URL to override the factory default deterministically.
    export SDKCONFIG=/project/sdkconfig.local
    idf.py reconfigure
fi
idf.py build
