#!/usr/bin/env bash
set -e

# Set the versions of IPOPT and HSL's builder to use - from the tag releases
# https://github.com/coin-or/Ipopt/tags
# https://github.com/coin-or-tools/ThirdParty-HSL/tags
IPOPT_VERSION="3.14.10"
HSL_BUILDER_VERSION="2.2.1"

# Defaults
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
INSTALL_HOME="${HOME}/.local/share"
IPOPT_HOME="${INSTALL_HOME}/ipopt"
HSL_HOME="${INSTALL_HOME}/hsl"

# Check if the script is run with sudo, and update HSL and IPOPT paths accordingly
if [ "$(id -u)" -eq 0 ]; then
  echo "Warning: Running this script with sudo is not recommended unless installing for all users."
  echo "Press enter to continue or Ctrl+C to cancel."
  read -n 1 -s
  echo
  HSL_HOME="/usr/local"
  IPOPT_HOME="/usr/local"
fi

function yes_no_prompt() {
  local prompt="$1"
  local response
  while true; do
    read -r -p "$prompt (y/n): " response
    case "$response" in
      [Yy]* ) return 0 ;;
      [Nn]* ) return 1 ;;
      * ) echo "Please answer y(es) or n(o)." ;;
    esac
  done
}

function install_system_prerequisites() {
  echo "Requesting sudo access for installing dependencies"
  sudo apt-get update
  sudo apt-get install libgomp1 liblapack3 libblas3 libmetis5 \
    build-essential gcc g++ gfortran git patch wget pkg-config \
    liblapack-dev libblas-dev libmetis-dev apt-utils
}

function install_conda_prerequisites() {
  echo "Installing prerequisites using conda"
  conda install libgomp conda-forge::liblapack libblas metis gcc_linux-64 \
    gxx_linux-64 gfortran_linux-64 wget pkg-config conda-build patch libcxx
}

function set_system_paths_loop() {
  echo "Enter the path to install IPOPT (and HSL):"
  read -r install_path
  if [[ -z "$install_path" ]]; then
    export INSTALL_HOME="$(realpath -smq ".")"
  else
    export INSTALL_HOME="$(realpath -smq "${install_path}")"
  fi
  
  if yes_no_prompt "Install to subdirectory [eg ${INSTALL_HOME}/ipopt, ${INSTALL_HOME}/hsl]"; then
    export IPOPT_HOME="$INSTALL_HOME/ipopt"
    export HSL_HOME="$INSTALL_HOME/hsl"
  else
    export IPOPT_HOME="$INSTALL_HOME"
    export HSL_HOME="$INSTALL_HOME"
  fi
  if [[ ! -d "$INSTALL_HOME" ]]; then
    echo "Directory ${INSTALL_HOME} does not exist. Creating it now."
    mkdir -p "$INSTALL_HOME"
  fi
}

function set_system_paths() {
  echo "By default, IPOPT will be installed this directory and a script will be created to source the variables."
  echo "If run with sudo, (/usr/local) will be set to make it available for all users."
  if [[ "$HSL_FOUND" = true ]]; then
    echo "Currently (${IPOPT_HOME}) and (${HSL_HOME}) are set."
  else
    echo "Currently (${IPOPT_HOME}) is set."
  fi
  if ! yes_no_prompt "Is this okay?"; then
    while true; do
      set_system_paths_loop
      echo "IPOPT-> ${IPOPT_HOME}"
      if [[ "$HSL_FOUND" = true ]]; then
        echo "HSL-> ${HSL_HOME}"
      fi
      if yes_no_prompt "Is this correct?"; then
        break
      fi
    done
    return
  fi
}

# Quick check for presence of coinhsl
HSL_FOUND=true
if [ -n "$COINHSL_PATH" ] && [ -f "$COINHSL_PATH" ]; then
  COINHSL_ARCHIVE="$COINHSL_PATH"
elif [ -f "$SCRIPT_DIR/coinhsl.tar.gz" ]; then
  COINHSL_ARCHIVE="$SCRIPT_DIR/coinhsl.tar.gz"
elif [ -f "./coinhsl-2023.11.17.tar.gz" ]; then
  COINHSL_ARCHIVE="$SCRIPT_DIR/coinhsl-2023.11.17.tar.gz"
else
  echo "Warning: CoinHSL archive not found! CoinHSL is required for the MA27, MA57, MA77, HSL_MA77, HSL_MA86, and HSL_MA97 solvers."
  echo "We _strongly_ recommend using HSL and offer no guarantees that our methods will work without it."
  if ! yes_no_prompt "Do you want to continue without it (not recommended!!)?"; then
    echo "Please obtain a copy of CoinHSL from https://licences.stfc.ac.uk/product/coin-hsl"
    echo "Either place the file in the same directory as this script and name it 'coinhsl.tar.gz',"
    echo "or set the environment variable COINHSL_PATH to point to the archive (e.g., export COINHSL_PATH=/path/to/coinhsl-*.tar.gz)."
    echo "Exiting..."
    exit 1
  fi
  HSL_FOUND=false
fi

if [ -n "$COINHSL_ARCHIVE" ]; then
  if [[ "$COINHSL_ARCHIVE" != *.tar.gz ]]; then
    echo "Error: CoinHSL archive must be a .tar.gz file. Found: $COINHSL_ARCHIVE"
    echo "Please follow the instructions above to download the correct .tar.gz file"
    exit 1
  fi
fi

# Make sure the script is not run from a conda environment
if [ -n "${CONDA_PREFIX}" ]; then
  echo "Continuing to install IPOPT (and HSL) in the current conda environment: ${CONDA_PREFIX}"
  if ! yes_no_prompt "Is this okay?"; then
    echo "Exiting... Please run this script outside of the conda environment to install for the system level."
    exit 1
  fi
  IPOPT_HOME="${CONDA_PREFIX}"
  HSL_HOME="${CONDA_PREFIX}"
  install_conda_prerequisites
else
  echo "Warning: You are not inside a conda environment."
  echo "It is _strongly_ recommended to run this script inside the 'splanning' conda environment."
  echo "This ensures that IPOPT and HSL are installed in an isolated and reproducible workspace."
  if ! yes_no_prompt "Do you want to continue with a system-level installation anyway?"; then
    echo "Exiting. Please activate the 'splanning' conda environment and re-run this script."
    exit 1
  fi
  echo "Continuing to install for the system level..."
  echo ""
  set_system_paths
  install_system_prerequisites
fi


# ----------------------------------------------------
# BEGIN ACTUAL INSTALLATION
# Required prerequisites are installed above
# Required variables for this section are set above
# $IPOPT_VERSION
# $HSL_BUILDER_VERSION
# $IPOPT_HOME
# $HSL_HOME
# $SCRIPT_DIR
# ----------------------------------------------------

# Setup IPOPT Builder and Prerequisites
echo "Setting up the IPOPT Builder..."
# Warn if ipopt_builder directory already exists
if [ -d "${SCRIPT_DIR}/ipopt_builder" ]; then
  echo "Warning: ipopt_builder directory already exists. It will be removed and recreated."
  if ! yes_no_prompt "Okay to proceed?"; then
    echo "Exiting... Please remove the ipopt_builder directory manually or run from a different directory."
    exit 1
  fi
  rm -rf "${SCRIPT_DIR}/ipopt_builder"
fi
wget https://github.com/coin-or/Ipopt/archive/refs/tags/releases/${IPOPT_VERSION}.tar.gz
tar xvf ./${IPOPT_VERSION}.tar.gz && mv Ipopt-releases-${IPOPT_VERSION} ipopt_builder
cd ipopt_builder

# Setup HSL
if [ "$HSL_FOUND" != true ]; then
  echo "Skipping HSL installation..."
  HSL_HOME=""
else
  echo "Setting up HSL..."
  rm -rf ThirdParty-HSL
  git clone --depth 1 --branch releases/${HSL_BUILDER_VERSION} https://github.com/coin-or-tools/ThirdParty-HSL.git
  cd "${SCRIPT_DIR}/ipopt_builder/ThirdParty-HSL"
  tar xvf $COINHSL_ARCHIVE
  mv $(ls -d coinhsl*/) coinhsl
  ./configure --prefix="${HSL_HOME}"
  make -j$(nproc)
  make install

  # Add link aliases for CoinHSL and HSL
  find "${HSL_HOME}/lib" -name "libcoinhsl.*" | xargs -I{} echo {} {} | sed "s/coin//2" | xargs -n 2 ln -s
  ln -s "${HSL_HOME}/lib/pkgconfig/coinhsl.pc" "${HSL_HOME}/lib/pkgconfig/hsl.pc"
fi

# Setup IPOPT
cd "${SCRIPT_DIR}/ipopt_builder"
mkdir build
cd build
../configure --prefix="${IPOPT_HOME}"
make -j$(nproc)
make install

cd "${SCRIPT_DIR}"
# Ask if want to delete build files
if yes_no_prompt "Do you want to delete the build files?"; then
  rm -rf "${SCRIPT_DIR}/ipopt_builder"
  rm -rf ${IPOPT_VERSION}.tar.gz
fi

# Make a script to source the IPOPT environment
if [ -z "${CONDA_PREFIX}" ]; then
  if [ "${IPOPT_HOME}" != "/usr/local" ]; then
    rm -f "${IPOPT_HOME}/ipopt_env.sh"
    echo "export IPOPT_HOME=${IPOPT_HOME}" >> "${IPOPT_HOME}/ipopt_env.sh"
    echo "export LD_LIBRARY_PATH=\${IPOPT_HOME}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" >> "${IPOPT_HOME}/ipopt_env.sh"
    echo "export PKG_CONFIG_PATH=\${IPOPT_HOME}/lib/pkgconfig\${PKG_CONFIG_PATH:+:\${PKG_CONFIG_PATH}}" >> "${IPOPT_HOME}/ipopt_env.sh"
    echo "export C_INCLUDE_PATH=\${IPOPT_HOME}/include/coin-or\${C_INCLUDE_PATH:+:\${C_INCLUDE_PATH}}" >> "${IPOPT_HOME}/ipopt_env.sh"
    if [ "$HSL_FOUND" = true ]; then
      echo "export HSL_HOME=${HSL_HOME}" >> "${IPOPT_HOME}/ipopt_env.sh"
      echo "export LD_LIBRARY_PATH=\${HSL_HOME}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" >> "${IPOPT_HOME}/ipopt_env.sh"
      echo "export PKG_CONFIG_PATH=\${HSL_HOME}/lib/pkgconfig\${PKG_CONFIG_PATH:+:\${PKG_CONFIG_PATH}}" >> "${IPOPT_HOME}/ipopt_env.sh"
      echo "export C_INCLUDE_PATH=\${HSL_HOME}/include/coin-or/hsl\${C_INCLUDE_PATH:+:\${C_INCLUDE_PATH}}" >> "${IPOPT_HOME}/ipopt_env.sh"

      echo "IPOPT and HSL have been installed successfully."
    else
      echo "IPOPT has been installed successfully."
    fi

    echo "Add it to your path by running the following command:"
    echo "source ${IPOPT_HOME}/ipopt_env.sh"
    if yes_no_prompt "Do you want to add this to your bashrc?"; then
        echo "source ${IPOPT_HOME}/ipopt_env.sh" >> ~/.bashrc
        echo "Added to bashrc. Please run 'source ~/.bashrc' to apply the changes."
    fi
  else
    echo "IPOPT (and HSL) have been installed successfully at the system level for all users."
  fi
else
  echo "IPOPT (and HSL) have been installed successfully."
  echo "Note: You can only use them in your conda environment, and they will not be listed in conda export."
fi