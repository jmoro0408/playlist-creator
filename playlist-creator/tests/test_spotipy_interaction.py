import configparser
from pathlib import Path

import pytest
from spotipy_interaction import SpotipyClient


@pytest.fixture
def client_fixture():
    config = configparser.ConfigParser()
    root_dir = Path(__file__).resolve().parents[1]
    config.read(Path(root_dir, "config.ini"))
    _client_id = config["SPOTIFY"]["SPOTIPY_CLIENT_ID"]
    _client_secret = config["SPOTIFY"]["SPOTIPY_CLIENT_SECRET"]
    return SpotipyClient(client_id=_client_id, client_secret=_client_secret)


def test_spotipy_login(client_fixture):
    client_fixture = client_fixture.authorize
    """This method is not actually
    implemented in my SpotipyClient Class, so need to call authorize here"""
    artist = "The Beatles"
    track = "Yellow Submarine"
    track_id = client_fixture.search(
        q="artist:" + artist + " track:" + track, type="track"
    )
    track_id = track_id["tracks"]["items"][0]["id"]
    track_json = client_fixture.tracks([track_id])
    assert track_json["tracks"][0]["artists"][0]["name"] == artist


def test_get_users_liked_tracks(client_fixture):
    limit = 10
    assert len(client_fixture.get_users_liked_tracks(limit = limit)["items"]) == 10

def test_get_users_playlists_names_type(client_fixture):
    assert isinstance(client_fixture.get_users_playlists_names(), dict)

def test_get_users_playlists_names_not_empty(client_fixture):
    assert len(client_fixture.get_users_playlists_names()) > 0

def test_get_user_playlist_track_info(client_fixture):
    errors = []
    first_playlist_name  = list(client_fixture.get_users_playlists_names())[0]
    test_playlist_id = client_fixture.get_users_playlists_names()[first_playlist_name]
    playlist_track_info = client_fixture.get_user_playlist_track_info(test_playlist_id)
    if not isinstance(playlist_track_info, list):
        errors.append("track info object not a list")
    if not isinstance(playlist_track_info[0], tuple):
        errors.append("first item in playlist track info not a tuple")
    assert len(errors) == 0

def test_get_track_features(client_fixture):
    errors = []
    track_id = "46lFttIf5hnUZMGvjK0Wxo" #galantis - Runaway (U&I)
    track_features = client_fixture.get_track_features(track_id)
    if len(track_features) != 1:
        errors.append(f"{len(track_features)} track returned, expected 1")
    if track_features[0]["danceability"] != 0.506:
        _msg = f"Danceability of {track_features[0]['danceability']} returned, expected 0.506"
        errors.append(_msg)
    if track_features[0]["duration_ms"] != 227074:
        _msg = f"Duration of {track_features[0]['duration_ms']} ms returned, expected 227074 ms"
        errors.append(_msg)
    assert len(errors) == 0