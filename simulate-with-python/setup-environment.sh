#!/bin/bash

# Black        0;30     Dark Gray     1;30
# Red          0;31     Light Red     1;31
# Green        0;32     Light Green   1;32
# Brown/Orange 0;33     Yellow        1;33
# Blue         0;34     Light Blue    1;34
# Purple       0;35     Light Purple  1;35
# Cyan         0;36     Light Cyan    1;36
# Light Gray   0;37     White         1;37

COLOR='\033[1;33m'
IMPORTANT='\033[1;31m'
RESET='\033[0m' # No Color

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
VENV_DEST=$(cd "$SCRIPT_DIR/.." && pwd)/python-virtualenv
LIBOQS_RS=$(cd "$SCRIPT_DIR/.." && pwd)/dependencies/liboqs-rs-bindings
LIBOQS_C="$LIBOQS_RS/liboqs"
PROTOGRAPH_DIR="$SCRIPT_DIR/ProtographLDPC"

pushd "$SCRIPT_DIR"

if [ ! -d "$VENV_DEST" ]; then
    echo -e "${COLOR}Couldn't find virtual environment. Creating one...${RESET}"
    python3 -m venv $VENV_DEST
fi

echo -e "${COLOR}Activating python virtual envionment...${RESET}"
source $VENV_DEST/bin/activate

echo -e "${COLOR}Making sure all specified python packages are installed...${RESET}"
pip install -r requirements.txt

# Build ProtographLDPC library if missing
if [ ! -d "$PROTOGRAPH_DIR" ] || [ -z "$(ls -A $PROTOGRAPH_DIR 2>/dev/null)" ]; then
    echo -e "${COLOR}Cloning ProtographLDPC submodule...${RESET}"
    git submodule init
    git submodule update
    echo -e "${COLOR}Building ProtographLDPC library...${RESET}"
    pushd "$PROTOGRAPH_DIR"
    git pull --recurse-submodules
    git submodule update --init --recursive
    pushd LDPC-codes
    make
    popd
    pushd peg
    make
    popd
    popd
fi

if [ ! -f "$LIBOQS_C/README.md" ]; then
    echo -e "${COLOR}Checking out submodules if not already done so...${RESET}"
    git submodule update --init --recursive
fi

if [ ! -f "$LIBOQS_C/build/lib/liboqs.a" ]; then
    echo -e "${COLOR}C-library not built. Building now (manually check correct system dependencies are installed, then press enter)...${RESET}"
    pushd $LIBOQS_RS
    bash build-oqs.sh --yes
    popd
fi

echo -e "${COLOR}Building local rust package...${RESET}"
pushd simulate_rs
maturin develop --release
popd #simulate_rs

popd #SCRIPT_DIR


echo -e "${COLOR}Done!${RESET}"

if [[ "$0" == *setup-environment.sh ]]; then
    echo
    echo -e "${IMPORTANT}This script should be run as: ${RESET}"
    echo -e "${IMPORTANT}\tsource $0${RESET}"
    echo -e "${IMPORTANT}if it wasn't you need to run the following to activate the virtual environment for the current shell:${RESET}"
    echo -e "${IMPORTANT}\tsource $VENV_DEST/bin/activate${RESET}"
fi