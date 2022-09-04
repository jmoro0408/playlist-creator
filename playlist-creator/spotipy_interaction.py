import configparser
from pathlib import Path

import spotipy
from spotipy.oauth2 import SpotifyOAuth


def import_config() -> tuple:
    config = configparser.ConfigParser()
    root_dir = Path(__file__).resolve().parents[0]
    config.read(Path(root_dir, "config.ini"))
    _client_id = config["SPOTIFY"]["SPOTIPY_CLIENT_ID"]
    _client_secret = config["SPOTIFY"]["SPOTIPY_CLIENT_SECRET"]
    return _client_id, _client_secret


class SpotipyClient(object):
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.authorize = self.spotify_auth()

    def spotify_auth(self):
        redirect_uri = "http://localhost:8888/callback"
        scope = "user-library-read"
        return spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                scope=scope,
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=redirect_uri,
            )
        )


if __name__ == "__main__":
    pass
