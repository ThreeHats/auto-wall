import json
import re
from urllib.request import Request, urlopen
from urllib.error import URLError

def parse_version(version_str):
    """Parse a version string into a tuple for comparison."""
    # Extract version numbers from the string
    match = re.search(r'(\d+)\.(\d+)\.(\d+)', version_str)
    if match:
        return tuple(map(int, match.groups()))
    return (0, 0, 0)  # Default if parsing fails

def check_for_updates(current_version, github_repo):
    """
    Check if updates are available from GitHub releases.
    
    Args:
        current_version: Current app version string (e.g. "1.0.0")
        github_repo: GitHub repository name (e.g. "username/repo")
        
    Returns:
        (bool, str, str): Tuple containing:
            - Whether an update is available
            - Latest version string
            - Download URL
    """
    try:
        # Form the GitHub API URL for releases
        api_url = f"https://api.github.com/repos/{github_repo}/releases/latest"
        
        # Set up the request with a user agent (GitHub API requires this)
        headers = {
            'User-Agent': f'Auto-Wall/{current_version}'
        }
        request = Request(api_url, headers=headers)
        
        # Fetch the latest release info
        with urlopen(request, timeout=5) as response:
            if response.getcode() == 200:
                data = json.loads(response.read().decode('utf-8'))
                latest_version = data.get('tag_name', '').lstrip('v')
                download_url = data.get('html_url', '')
                
                # Compare versions
                current_version_tuple = parse_version(current_version)
                latest_version_tuple = parse_version(latest_version)
                
                if latest_version_tuple > current_version_tuple:
                    return True, latest_version, download_url
                    
    except URLError as e:
        print(f"Error checking for updates: {e}")
    except Exception as e:
        print(f"Unexpected error checking for updates: {e}")
        
    return False, current_version, ""
