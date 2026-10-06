"""Download buyer photos once; unavailable photos are explicitly missing."""

import hashlib
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image


class ImageStore:
    def __init__(self, directory, max_bytes=5 * 1024 * 1024):
        self.directory = Path(directory)
        self.max_bytes = max_bytes

    def path_for(self, url):
        return self.directory / (hashlib.sha256(url.encode()).hexdigest() + '.img')

    def load(self, url, download=True):
        path = self.path_for(url)
        if not path.is_file() and download:
            parsed = urlparse(url)
            if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
                return None
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            try:
                with requests.get(url, stream=True, timeout=(10, 20)) as response:
                    response.raise_for_status()
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
