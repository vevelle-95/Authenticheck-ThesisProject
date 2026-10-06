"""Download buyer photos once; unavailable photos are explicitly missing."""

import hashlib
import ipaddress
import socket
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image


def public_image_url(url):
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
            return False
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80))
        return bool(addresses) and all(ipaddress.ip_address(address[4][0]).is_global for address in addresses)
    except (OSError, ValueError):
        return False


class ImageStore:
    def __init__(self, directory, max_bytes=5 * 1024 * 1024):
        self.directory = Path(directory)
        self.max_bytes = max_bytes

    def path_for(self, url):
        return self.directory / (hashlib.sha256(url.encode()).hexdigest() + '.img')

    def load(self, url, download=True):
        path = self.path_for(url)
        if not path.is_file() and download:
            if not public_image_url(url):
                return None
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            try:
                with requests.get(url, stream=True, timeout=(10, 20), allow_redirects=False) as response:
                    response.raise_for_status()
                    if response.status_code != 200:
                        return None
                    size = 0
                    with temporary.open('wb') as stream:
                        for chunk in response.iter_content(65536):
                            size += len(chunk)
                            if size > self.max_bytes:
                                raise ValueError('Image exceeds the 5 MiB limit')
                            stream.write(chunk)
                with Image.open(temporary) as image:
                    image.verify()
                temporary.replace(path)
            except (requests.RequestException, OSError, ValueError, Image.DecompressionBombError):
                temporary.unlink(missing_ok=True)
                return None
        if not path.is_file():
            return None
        try:
            with Image.open(path) as image:
                return image.convert('RGB').copy()
        except (OSError, ValueError, Image.DecompressionBombError):
            return None
