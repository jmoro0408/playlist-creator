import configparser
from pathlib import Path

import spotipy # type: ignore
from spotipy.oauth2 import SpotifyOAuth # type: ignore

#TODO Authorization flow is messy

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

    def spotify_auth(self, scope = "user-library-read"):
        redirect_uri = "http://localhost:8888/callback"
        return spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                scope=scope,
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=redirect_uri,
            )
        )


    def get_users_liked_tracks(self, limit):
        return self.authorize.current_user_saved_tracks(limit= limit)


    def get_users_playlists(self):
        pass


if __name__ == "__main__":
    client_id, client_secret = import_config()
    sp = SpotipyClient(client_id, client_secret)

    liked_songs = (sp.get_users_liked_tracks(10))
    print("test")