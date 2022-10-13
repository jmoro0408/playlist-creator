import configparser
from pathlib import Path

import pandas as pd
import pytest

from playlist_recommender.build_song_df import (build_liked_song_df,
                                                build_playlists_df)
from playlist_recommender.spotipy_interaction import SpotipyClient


@pytest.fixture
def client_fixture():
    config = configparser.ConfigParser()
    root_dir = Path(__file__).resolve().parents[1]
    config.read(Path(root_dir, "config.ini"))
    _client_id = config["SPOTIFY"]["SPOTIPY_CLIENT_ID"]
    _client_secret = config["SPOTIFY"]["SPOTIPY_CLIENT_SECRET"]
    return SpotipyClient(client_id=_client_id, client_secret=_client_secret)


def test_build_liked_song_df(client_fixture):
    results_df = build_liked_song_df(client_fixture)
    errors = []
    if not isinstance(results_df, pd.DataFrame):
        _msg = f"{type(results_df)} returned, expected Dataframe"
        errors.append(_msg)
    expected_columns = [
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "type",
        "id",
        "uri",
        "track_href",
        "analysis_url",
        "duration_ms",
        "time_signature",
        "artist_names",
        "track_names",
    ]
    missing_columns = [x for x in expected_columns if x not in results_df.columns]
    extra_columns = [x for x in results_df.columns if x not in expected_columns]
    if len(missing_columns) != 0:
        _msg = f"{missing_columns} columns not found in dataframe"
        errors.append(_msg)
    if len(extra_columns) != 0:
        _msg = f"{extra_columns} columns were found in the dataframe that were not expected"
        errors.append(_msg)
    assert len(errors) == 0


def test_build_playlists_df(client_fixture):
    playlist_df = build_playlists_df(client_fixture)
    errors = []
    if not isinstance(playlist_df, pd.DataFrame):
        _msg = f"{playlist_df} returned, expected dataframe"
        errors.append(_msg)
    if playlist_df.empty:
        _msg = "empty dataframe returned"
        errors.append(_msg)
    assert len(errors) == 0
