#!/bin/bash

# Auto-Wall Linux Installation Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="/opt/auto-wall"
BIN_DIR="/usr/local/bin"
DESKTOP_DIR="/usr/share/applications"
ICONS_DIR="/usr/share/icons/hicolor/256x256/apps"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_error "This script should not be run as root. Please run as a regular user."
        print_error "The script will ask for sudo permissions when needed."
        exit 1
    fi
}

check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check if we're on a supported distribution
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        print_status "Detected OS: $NAME"
        
        case "$ID" in
            ubuntu|debian|linuxmint|elementary|pop)
                PKG_MANAGER="apt"
                ;;
            fedora|centos|rhel)
                PKG_MANAGER="dnf"
                ;;
            arch|manjaro)
                PKG_MANAGER="pacman"
                ;;
            *)
                print_warning "Unsupported distribution: $ID"
                print_warning "Installation may not work correctly"
                PKG_MANAGER="unknown"
                ;;
        esac
    else
        print_warning "Could not detect OS distribution"
        PKG_MANAGER="unknown"
    fi
}

install_dependencies() {
    print_status "Installing system dependencies..."
    
    case "$PKG_MANAGER" in
        apt)
            sudo apt-get update
            sudo apt-get install -y python3 python3-tk desktop-file-utils
            ;;
        dnf)
            sudo dnf install -y python3 python3-tkinter desktop-file-utils
            ;;
        pacman)
            sudo pacman -Sy --noconfirm python python-tkinter desktop-file-utils
            ;;
        *)
            print_warning "Unknown package manager. Please install Python 3 and tkinter manually."
            ;;
    esac
}

install_from_appimage() {
    local appimage_file="$1"
    
    print_status "Installing from AppImage: $appimage_file"
    
    # Create installation directory
    sudo mkdir -p "$INSTALL_DIR"
    
    # Copy AppImage
    sudo cp "$appimage_file" "$INSTALL_DIR/auto-wall"
    sudo chmod +x "$INSTALL_DIR/auto-wall"
    
    # Create symlink in bin directory
    sudo ln -sf "$INSTALL_DIR/auto-wall" "$BIN_DIR/auto-wall"
    
    # Extract icon from AppImage if possible
    if command -v 7z >/dev/null 2>&1; then
        temp_dir=$(mktemp -d)
        cd "$temp_dir"
        7z x "$SCRIPT_DIR/$appimage_file" auto-wall.png >/dev/null 2>&1 || true
        if [[ -f auto-wall.png ]]; then
            sudo mkdir -p "$ICONS_DIR"
            sudo cp auto-wall.png "$ICONS_DIR/auto-wall.png"
        fi
        cd - >/dev/null
        rm -rf "$temp_dir"
    fi
    
    create_desktop_file
    print_success "AppImage installation completed"
}

install_from_deb() {
    local deb_file="$1"
    
    print_status "Installing from Debian package: $deb_file"
    
    sudo dpkg -i "$deb_file"
    
    # Fix any missing dependencies
    if ! sudo apt-get install -f -y; then
        print_error "Failed to install dependencies"
        return 1
    fi
    
    print_success "Debian package installation completed"
}

install_from_executable() {
    local exe_file="$1"
    
    print_status "Installing from standalone executable: $exe_file"
    
    # Create installation directory
    sudo mkdir -p "$INSTALL_DIR"
    
    # Copy executable
    sudo cp "$exe_file" "$INSTALL_DIR/auto-wall"
    sudo chmod +x "$INSTALL_DIR/auto-wall"
    
    # Create symlink in bin directory
    sudo ln -sf "$INSTALL_DIR/auto-wall" "$BIN_DIR/auto-wall"
    
    create_desktop_file
    print_success "Standalone executable installation completed"
}

create_desktop_file() {
    print_status "Creating desktop entry..."
    
    sudo tee "$DESKTOP_DIR/auto-wall.desktop" > /dev/null <<EOF
[Desktop Entry]
Name=Auto-Wall
Comment=Battle Map Wall Detection Tool
Exec=auto-wall
Icon=auto-wall
Type=Application
Categories=Graphics;Photography;
Terminal=false
StartupWMClass=Auto-Wall
EOF

    # Update desktop database
    if command -v update-desktop-database >/dev/null 2>&1; then
        sudo update-desktop-database "$DESKTOP_DIR"
    fi
    
    # Update icon cache
    if command -v gtk-update-icon-cache >/dev/null 2>&1; then
        sudo gtk-update-icon-cache -q "$ICONS_DIR/../../../" 2>/dev/null || true
    fi
}

uninstall() {
    print_status "Uninstalling Auto-Wall..."
    
    # Remove from package manager if installed
    if dpkg -l | grep -q auto-wall 2>/dev/null; then
        print_status "Removing Debian package..."
        sudo apt-get remove -y auto-wall
    fi
    
    # Remove manual installation
    if [[ -f "$INSTALL_DIR/auto-wall" ]]; then
        print_status "Removing manual installation..."
        sudo rm -rf "$INSTALL_DIR"
    fi
    
    # Remove symlink
    if [[ -L "$BIN_DIR/auto-wall" ]]; then
        sudo rm -f "$BIN_DIR/auto-wall"
    fi
    
    # Remove desktop file
    if [[ -f "$DESKTOP_DIR/auto-wall.desktop" ]]; then
        sudo rm -f "$DESKTOP_DIR/auto-wall.desktop"
        if command -v update-desktop-database >/dev/null 2>&1; then
            sudo update-desktop-database "$DESKTOP_DIR"
        fi
    fi
    
    # Remove icon
    if [[ -f "$ICONS_DIR/auto-wall.png" ]]; then
        sudo rm -f "$ICONS_DIR/auto-wall.png"
        if command -v gtk-update-icon-cache >/dev/null 2>&1; then
            sudo gtk-update-icon-cache -q "$ICONS_DIR/../../../" 2>/dev/null || true
        fi
    fi
    
    print_success "Auto-Wall has been uninstalled"
}

main() {
    echo "Auto-Wall Linux Installation Script"
    echo "==================================="
    echo
    
    case "${1:-}" in
        --uninstall|-u)
            uninstall
            exit 0
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo
            echo "Options:"
            echo "  --uninstall, -u    Uninstall Auto-Wall"
            echo "  --help, -h         Show this help message"
            echo
            echo "Installation:"
            echo "  Run without arguments to auto-detect and install available packages"
            exit 0
            ;;
    esac
    
    check_root
    check_dependencies
    install_dependencies
    
    cd "$SCRIPT_DIR"
    
    # Find available packages
    appimage_files=(*.AppImage)
    deb_files=(*.deb)
    exe_files=()
    
    # Look for standalone executable
    if [[ -f "dist/Auto-Wall" ]]; then
        exe_files+=("dist/Auto-Wall")
    elif [[ -f "Auto-Wall" ]]; then
        exe_files+=("Auto-Wall")
    fi
    
    # Install in order of preference: .deb > AppImage > standalone
    installed=false
    
    # Try .deb first
    for deb_file in "${deb_files[@]}"; do
        if [[ -f "$deb_file" && "$deb_file" != "*.deb" ]]; then
            if install_from_deb "$deb_file"; then
                installed=true
                break
            fi
        fi
    done
    
    # Try AppImage if .deb failed or not available
    if [[ "$installed" == "false" ]]; then
        for appimage_file in "${appimage_files[@]}"; do
            if [[ -f "$appimage_file" && "$appimage_file" != "*.AppImage" ]]; then
                if install_from_appimage "$appimage_file"; then
                    installed=true
                    break
                fi
            fi
        done
    fi
    
    # Try standalone executable as last resort
    if [[ "$installed" == "false" ]]; then
        for exe_file in "${exe_files[@]}"; do
            if [[ -f "$exe_file" ]]; then
                if install_from_executable "$exe_file"; then
                    installed=true
                    break
                fi
            fi
        done
    fi
    
    if [[ "$installed" == "false" ]]; then
        print_error "No suitable package found for installation"
        print_error "Please download a package from the releases page:"
        print_error "https://github.com/ThreeHats/auto-wall/releases"
        exit 1
    fi
    
    echo
    print_success "Auto-Wall installation completed!"
    print_status "You can now run 'auto-wall' from the command line"
    print_status "Or find 'Auto-Wall' in your application menu"
}

main "$@"