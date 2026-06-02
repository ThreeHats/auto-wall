import json
import re
from urllib.request import Request, urlopen
from urllib.error import URLError

from PyQt6.QtGui import QDesktopServices
from PyQt6.QtCore import QUrl

def parse_version(version_str):
    """Parse a version string into a tuple for comparison.

    Returns a 5-tuple (major, minor, patch, stable_flag, pre_num) so that
    SemVer pre-release ordering works correctly:
      1.4.0          -> (1, 4, 0, 1, 0)  # stable, sorts highest
      1.4.0-beta.2   -> (1, 4, 0, 0, 2)
      1.4.0-beta.1   -> (1, 4, 0, 0, 1)
    """
    match = re.search(r'(\d+)\.(\d+)\.(\d+)', version_str)
    if not match:
        return (0, 0, 0, 0, 0)
    major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))
    remainder = version_str[match.end():]
    if not remainder or not remainder.startswith('-'):
        return (major, minor, patch, 1, 0)
    beta_match = re.search(r'-beta\.(\d+)', remainder)
    if beta_match:
        return (major, minor, patch, 0, int(beta_match.group(1)))
    return (major, minor, patch, 0, 0)

def fetch_version(current_version, github_repo):
    """
    Check if updates are available from GitHub releases.

    - Dev/PR builds (contain '-dev-' or '-pr-'): skipped entirely.
    - Beta builds (contain '-beta.'): checks all releases so newer betas
      and stable releases are both detected.
    - Stable builds: checks /releases/latest (stable only).

    Returns:
        (bool, str, str): (update_available, latest_version, download_url)
    """
    if '-dev-' in current_version or '-pr-' in current_version:
        return False, current_version, ""

    is_beta = '-beta.' in current_version

    try:
        headers = {'User-Agent': f'Auto-Wall/{current_version}'}
        latest_version = None
        download_url = ""

        if is_beta:
            api_url = f"https://api.github.com/repos/{github_repo}/releases"
            request = Request(api_url, headers=headers)
            with urlopen(request, timeout=5) as response:
                if response.getcode() == 200:
                    releases = json.loads(response.read().decode('utf-8'))
                    if releases:
                        # First entry is the newest non-draft release (stable or pre-release)
                        latest = releases[0]
                        latest_version = latest.get('tag_name', '').lstrip('v')
                        download_url = latest.get('html_url', '')
        else:
            api_url = f"https://api.github.com/repos/{github_repo}/releases/latest"
            request = Request(api_url, headers=headers)
            with urlopen(request, timeout=5) as response:
                if response.getcode() == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    latest_version = data.get('tag_name', '').lstrip('v')
                    download_url = data.get('html_url', '')

        if latest_version and parse_version(latest_version) > parse_version(current_version):
            return True, latest_version, download_url

    except URLError as e:
        print(f"Error checking for updates: {e}")
    except Exception as e:
        print(f"Unexpected error checking for updates: {e}")

    return False, current_version, ""

def check_for_updates(self):
    """Check for updates and show notification if available."""
    try:
        is_update_available, latest_version, download_url = fetch_version(
            self.app_version, self.github_repo
        )
        
        if is_update_available:
            self.update_available = True
            self.update_url = download_url
            self.update_text.setText(f"Update {latest_version} Available!")
            # Ensure dismiss button is still visible and connected
            if hasattr(self, 'dismiss_button'):
                self.dismiss_button.show()
            self.update_notification.show()
            # Position the notification after it's shown and sized
            self.position_update_notification()
            print(f"Update available: version {latest_version}")
    except Exception as e:
        print(f"Error checking for updates: {e}")

def open_update_url(self, event):
    """Open the update URL when the notification is clicked."""
    if self.update_url:
        QDesktopServices.openUrl(QUrl(self.update_url))