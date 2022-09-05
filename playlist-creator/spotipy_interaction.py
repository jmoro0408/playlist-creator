import configparser
from pathlib import Path

import spotipy  # type: ignore
from spotipy.oauth2 import SpotifyOAuth  # type: ignore

# TODO Authorization flow is messy


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

    def spotify_auth(self, scope="user-library-read"):
        redirect_uri = "http://localhost:8888/callback"
        return spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                scope=scope,
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=redirect_uri,
            )
        )

    def get_users_liked_tracks(self, limit: int):
        return self.authorize.current_user_saved_tracks(limit=limit)

    def get_users_playlists_names(self) -> dict:
        _sp = self.authorize
        playlists = _sp.current_user_playlists()
        user_id = _sp.me()["id"]
        user_playlists_name_id_dict = {}
        for playlist in playlists["items"]:
            if playlist["owner"]["id"] == user_id:
                user_playlists_name_id_dict[playlist["name"]] = playlist["id"]
        return user_playlists_name_id_dict

    def get_user_playlist_track_info(self, playlist_id: str):
        # TODO named tuples for track info would be clearer
        _sp = self.authorize
        playlist_info = (
            []
        )  # to hold a tuple of (track_name, artist_name, track_id) for eac htrack in playlist
        results = _sp.playlist(playlist_id, fields="tracks,artists")
        num_tracks = len(results["tracks"]["items"])
        for i in range(num_tracks):
            _track_name = results["tracks"]["items"][i]["track"]["name"]
            _track_id = results["tracks"]["items"][i]["track"]["id"]
            _artist_name = results["tracks"]["items"][i]["track"]["artists"][0][
                "name"
            ]  # 2nd 0 might fail if more than one artist on track
            _result = (_track_name, _artist_name, _track_id)
            playlist_info.append(_result)
        return playlist_info

    def get_track_features(self, track_id: str):
        return self.authorize.audio_features(tracks=track_id)


if __name__ == "__main__":
    client_id, client_secret = import_config()
    sp = SpotipyClient(client_id, client_secret)

    first_playlist_name = list(sp.get_users_playlists_names())[0]
    test_playlist_id = sp.get_users_playlists_names()[first_playlist_name]
    test_track = sp.get_user_playlist_track_info(test_playlist_id)[0]
    test_track_id = test_track[2]
    print(sp.get_track_features(test_track_id))
