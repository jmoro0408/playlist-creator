import configparser
import itertools
from pathlib import Path
from typing import Any

import spotipy  # type: ignore
from spotipy.oauth2 import SpotifyOAuth  # type: ignore

# TODO Authorization flow is messy


def read_client_id_and_secret(config_file: str = "config.ini") -> tuple:
    """read in the spotify client ID and secret.
    These should be stored in a config.ini file with the structure:
    [SPOTIFY]
    SPOTIPY_CLIENT_ID=xxxxxxxxxxxxxxxxxxxxxxx
    SPOTIPY_CLIENT_SECRET=xxxxxxxxxxxxxxxxxxxxxxx
    Args:
        config_file (str, optional): path to config file. Defaults to "config.ini".

    Returns:
        tuple: client_id and client_secret as read from the config file
    """
    config = configparser.ConfigParser()
    root_dir = Path(__file__).resolve().parents[0]
    config.read(Path(root_dir, config_file))
    _client_id = config["SPOTIFY"]["SPOTIPY_CLIENT_ID"]
    _client_secret = config["SPOTIFY"]["SPOTIPY_CLIENT_SECRET"]
    return _client_id, _client_secret


def split_into_chunks(list_to_split: list, chunk_size: int):
    """
    This is stolen from a SO post:
    https://stackoverflow.com/questions/2130016/
    splitting-a-list-into-n-parts-of-approximately-equal-length
    and just splits a list into equal chunks of a specified size
    """
    k, m = divmod(len(list_to_split), chunk_size)
    return (
        list_to_split[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(chunk_size)
    )


class SpotipyClient(object):
    def __init__(self, client_id: str, client_secret: str) -> Any:
        self.client_id = client_id
        self.client_secret = client_secret
        self.authorize = self.spotify_auth()

    def spotify_auth(self, scope: str = "user-library-read"):
        """Returns an authorized spotipy object for the given scope.

        Args:
            scope (str, optional): authorization scope. Defaults to "user-library-read".
            Available scopes can be found at:
            https://developer.spotify.com/documentation/general/guides/authorization/scopes/

        Returns:
            Any: Authorized spotify client
        """
        redirect_uri = "http://localhost:8888/callback"
        return spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                scope=scope,
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=redirect_uri,
            )
        )

    def get_users_all_liked_tracks(self) -> list[dict]:
        """retries a list of all the user's liked tracks.

        Returns:
            list[dict]: users liked tracks with track information
        """
        sp = self.authorize
        results = sp.current_user_saved_tracks()
        tracks = results["items"]
        while results["next"]:
            results = sp.next(results)
            tracks.extend(results["items"])
        return tracks

    def get_user_liked_tracks_limited(self, limit: int = 10) -> list[dict]:
        """gets a subset of the users liked tracks up to a specified limit

        Args:
            limit (int, optional): how many tracks to grab. Defaults to 10.

        Returns:
            list[dict]: users liked tracks with track information
        """
        sp = self.authorize
        track_limit = 50  # set by spotify API
        num_batches, leftovers = divmod(limit, track_limit)  # 50 song paginated result
        offset = 0
        user_saved_tracks_chunked = []
        for i in range(num_batches):
            offset = offset + (i * track_limit)
            results = sp.current_user_saved_tracks(limit=track_limit, offset=offset)[
                "items"
            ]
            user_saved_tracks_chunked.append(results)
        if leftovers > 0:
            user_saved_tracks_chunked.append(
                sp.current_user_saved_tracks(limit=leftovers, offset=offset)["items"]
            )
        return list(itertools.chain.from_iterable(user_saved_tracks_chunked))

    def parse_users_liked_tracks(
        self, user_liked_songs_json: list[dict[Any, Any]]
    ) -> list[tuple[Any, Any, Any]]:
        # TODO lots of repeated code from playlist track info func
        liked_song_info: list[tuple[Any, Any, Any]] = []
        num_tracks = len(user_liked_songs_json)
        for i in range(num_tracks):
            _track_name = user_liked_songs_json[i]["track"]["name"]
            _track_id = user_liked_songs_json[i]["track"]["id"]
            _artist_name = user_liked_songs_json[i]["track"]["artists"][0][
                "name"
            ]  # 0 may fail if more than 1 artist
            _result = (_track_name, _artist_name, _track_id)
            liked_song_info.append(_result)
        return liked_song_info

    def get_users_playlists_info(self) -> dict:
        """grabs the playlist info for each playlist
        that the user has created

        Returns:
            dict: dict of playlist info
        """
        _sp = self.authorize
        playlists = _sp.current_user_playlists()
        user_id = _sp.me()["id"]
        user_playlists_name_id_dict = {}
        for playlist in playlists["items"]:
            if playlist["owner"]["id"] == user_id:
                user_playlists_name_id_dict[playlist["name"]] = playlist["id"]
        return user_playlists_name_id_dict

    def get_user_playlist_track_info(
        self, playlist_id: str
    ) -> list[tuple[Any, Any, Any]]:
        # TODO named tuples for track info would be clearer
        _sp = self.authorize
        results = _sp.playlist(playlist_id, fields="tracks,next")
        playlist_info = []
        tracks = results["tracks"]
        while tracks["next"]:
            num_tracks = len(tracks["items"])
            for i in range(num_tracks):
                _track_name = tracks["items"][i]["track"]["name"]
                _track_id = tracks["items"][i]["track"]["id"]
                _artist_name = tracks["items"][i]["track"]["artists"][0][
                    "name"
                ]  # 2nd 0 might fail if more than one artist on track
                _result = (_track_name, _artist_name, _track_id)
                playlist_info.append(_result)  # type: ignore
            tracks = _sp.next(tracks)
        num_tracks = len(tracks["items"])
        for i in range(num_tracks):
            _track_name = tracks["items"][i]["track"]["name"]
            _track_id = tracks["items"][i]["track"]["id"]
            _artist_name = tracks["items"][i]["track"]["artists"][0][
                "name"
            ]  # 2nd 0 might fail if more than one artist on track
            _result = (_track_name, _artist_name, _track_id)
            playlist_info.append(_result)  # type: ignore
        return playlist_info

    def get_track_features(self, track_ids: list):
        """grabs spotify features

        Args:
            track_ids (list): track ids to get info for

        Returns:
            _type_: list of audio features for each track
        """
        return self.authorize.audio_features(tracks=track_ids)


if __name__ == "__main__":
    # _client_id, _client_secret = import_config()
    # sp = SpotipyClient(_client_id, _client_secret)
    pass
