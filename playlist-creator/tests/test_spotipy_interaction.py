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
    return SpotipyClient(client_id=_client_id, client_secret=_client_secret).authorize


def test_spotipy_login(client_fixture):
    artist = "The Beatles"
    track = "Yellow Submarine"
    track_id = client_fixture.search(
        q="artist:" + artist + " track:" + track, type="track"
    )
    track_id = track_id["tracks"]["items"][0]["id"]
    track_json = client_fixture.tracks([track_id])
    assert track_json["tracks"][0]["artists"][0]["name"] == artist
