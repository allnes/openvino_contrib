#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  collect_license_inputs.sh validate-submodules
  collect_license_inputs.sh openvino-header-files
  collect_license_inputs.sh full-repo
  collect_license_inputs.sh thirdparty-focused
EOF
}

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

mode="${1:-}"

ensure_policy() {
    if [[ ! -f .github/scancode/policies.yml ]]; then
        echo "Missing .github/scancode/policies.yml" >&2
        exit 1
    fi
}

validate_submodules() {
    if [[ ! -f .gitmodules ]]; then
        echo "No .gitmodules file found; nothing to initialize."
        return 0
    fi

    git submodule sync --recursive
    git submodule update --init --recursive

    if git submodule status --recursive | awk '$1 ~ /^-/ { found = 1 } END { exit found ? 0 : 1 }'; then
        echo "One or more submodules are not initialized:" >&2
        git submodule status --recursive >&2
        exit 1
    fi
}

write_openvino_header_files() {
    mkdir -p license-inputs
    git ls-files | awk '
        $0 ~ /(^|\/)(thirdparty|3rdparty|external|vendor|vendors|deps|submodules)\// { next }
        $0 ~ /^modules\/ollama_openvino\/llama\/llama\.cpp\// { next }
        $0 ~ /^modules\/ollama_openvino\/ml\/backend\/ggml\/ggml\// { next }
        { print }
    ' > license-inputs/openvino-header-files.txt
    echo "Wrote license-inputs/openvino-header-files.txt"
}

rsync_common_excludes=(
    --exclude '.git/'
    --exclude '.scancodeio/'
    --exclude 'license-inputs/'
    --exclude 'scancode-inputs/'
    --exclude 'scancode-thirdparty-inputs/'
    --exclude 'build/'
    --exclude '**/build/'
    --exclude 'cmake-build-*/'
    --exclude '**/cmake-build-*/'
    --exclude '.cache/'
    --exclude '**/.cache/'
    --exclude '.ccache/'
    --exclude '**/.ccache/'
    --exclude '**/__pycache__/'
    --exclude '**/node_modules/'
)

prepare_full_repo() {
    ensure_policy
    rm -rf scancode-inputs
    mkdir -p scancode-inputs/repository

    rsync -a --delete "${rsync_common_excludes[@]}" ./ scancode-inputs/repository/
    cp .github/scancode/policies.yml scancode-inputs/policies.yml
    echo "Prepared full repository ScanCode input at scancode-inputs/"
}

copy_relative_path() {
    local source_path="$1"
    local destination_root="$2"
    local normalized="${source_path#./}"

    if [[ ! -e "$normalized" ]]; then
        return 0
    fi

    mkdir -p "$destination_root/$(dirname "$normalized")"
    rsync -a --delete "${rsync_common_excludes[@]}" "$normalized" "$destination_root/$(dirname "$normalized")/"
}

prepare_thirdparty_focused() {
    ensure_policy
    rm -rf scancode-thirdparty-inputs
    mkdir -p scancode-thirdparty-inputs/repository

    cp .github/scancode/policies.yml scancode-thirdparty-inputs/policies.yml

    for root_file in LICENSE NOTICE third-party-programs.txt; do
        copy_relative_path "$root_file" scancode-thirdparty-inputs/repository
    done

    mapfile -t vendored_roots < <(
        find . \
            -path './.git' -prune -o \
            -path './scancode-inputs' -prune -o \
            -path './scancode-thirdparty-inputs' -prune -o \
            -path './license-inputs' -prune -o \
            -type d \( \
                -name thirdparty -o \
                -name 3rdparty -o \
                -name external -o \
                -name vendor -o \
                -name vendors -o \
                -name deps -o \
                -name submodules \
            \) -print | sort
    )

    repo_specific_roots=(
        "modules/ollama_openvino/llama/llama.cpp"
        "modules/ollama_openvino/ml/backend/ggml/ggml"
    )

    for path in "${repo_specific_roots[@]}"; do
        if [[ -d "$path" ]]; then
            vendored_roots+=("$path")
        fi
    done

    if [[ ${#vendored_roots[@]} -eq 0 ]]; then
        echo "No vendored roots found; focused input contains repository-level license documents only."
    else
        printf '%s\n' "${vendored_roots[@]}" | sort -u > scancode-thirdparty-inputs/vendored-roots.txt
        while IFS= read -r path; do
            copy_relative_path "$path" scancode-thirdparty-inputs/repository
        done < scancode-thirdparty-inputs/vendored-roots.txt
    fi

    echo "Prepared third-party ScanCode input at scancode-thirdparty-inputs/"
}

case "$mode" in
    validate-submodules)
        validate_submodules
        ;;
    openvino-header-files)
        write_openvino_header_files
        ;;
    full-repo)
        prepare_full_repo
        ;;
    thirdparty-focused)
        prepare_thirdparty_focused
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
