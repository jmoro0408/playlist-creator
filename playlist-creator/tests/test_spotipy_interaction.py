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
    assert len(client_fixture.get_users_liked_tracks(limit=limit)["items"]) == 10


def test_get_users_playlists_names_type(client_fixture):
    assert isinstance(client_fixture.get_users_playlists_names(), dict)


def test_get_users_playlists_names_not_empty(client_fixture):
    assert len(client_fixture.get_users_playlists_names()) > 0


def test_get_user_playlist_track_info(client_fixture):
    errors = []
    first_playlist_name = list(client_fixture.get_users_playlists_names())[0]
    test_playlist_id = client_fixture.get_users_playlists_names()[first_playlist_name]
    playlist_track_info = client_fixture.get_user_playlist_track_info(test_playlist_id)
    if not isinstance(playlist_track_info, list):
        errors.append("track info object not a list")
    if not isinstance(playlist_track_info[0], tuple):
        errors.append("first item in playlist track info not a tuple")
    assert len(errors) == 0


def test_get_track_features_single_track(client_fixture):
    errors = []
    track_id = "46lFttIf5hnUZMGvjK0Wxo"  # galantis - Runaway (U&I)
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


def test_get_track_features_multiple_tracks(client_fixture):
    errors = []
    first_playlist_name = list(client_fixture.get_users_playlists_names())[0]
    first_playlist_id = client_fixture.get_users_playlists_names()[first_playlist_name]
    first_track_id = client_fixture.get_user_playlist_track_info(first_playlist_id)[0][
        2
    ]
    second_track_id = client_fixture.get_user_playlist_track_info(first_playlist_id)[1][
        2
    ]
    track_ids = [first_track_id, second_track_id]
    track_features = client_fixture.get_track_features(track_ids)
    if not isinstance(track_features, list):
        _msg = f"{type(track_features)} returned, expected list"
        errors.append(_msg)
    if len(track_features) != 2:
        _msg = f"{len(track_features)} tracks returned, expected 2"
        errors.append(_msg)
    assert len(errors) == 0


def test_parse_users_liked_tracks(client_fixture):
    _example_ = {
        "href": "https://api.spotify.com/v1/me/tracks?offset=0&limit=1",
        "items": [
            {
                "added_at": "2022-09-05T13:34:19Z",
                "track": {
                    "album": {
                        "album_type": "album",
                        "artists": [
                            {
                                "external_urls": {
                                    "spotify": "https://open.spotify.com/artist/3PhoLpVuITZKcymswpck5b"
                                },
                                "href": "https://api.spotify.com/v1/artists/3PhoLpVuITZKcymswpck5b",
                                "id": "3PhoLpVuITZKcymswpck5b",
                                "name": "Elton John",
                                "type": "artist",
                                "uri": "spotify:artist:3PhoLpVuITZKcymswpck5b",
                            }
                        ],
                        "available_markets": [
                            "AD",
                            "AE",
                            "AG",
                            "AL",
                            "AM",
                            "AO",
                            "AR",
                            "AT",
                            "AU",
                            "AZ",
                            "BA",
                            "BB",
                            "BD",
                            "BE",
                            "BF",
                            "BG",
                            "BH",
                            "BI",
                            "BJ",
                            "BN",
                            "BO",
                            "BR",
                            "BS",
                            "BT",
                            "BW",
                            "BY",
                            "BZ",
                            "CA",
                            "CD",
                            "CG",
                            "CH",
                            "CI",
                            "CL",
                            "CM",
                            "CO",
                            "CR",
                            "CV",
                            "CW",
                            "CY",
                            "CZ",
                            "DE",
                            "DJ",
                            "DK",
                            "DM",
                            "DO",
                            "DZ",
                            "EC",
                            "EE",
                            "EG",
                            "ES",
                            "FI",
                            "FJ",
                            "FM",
                            "FR",
                            "GA",
                            "GB",
                            "GD",
                            "GE",
                            "GH",
                            "GM",
                            "GN",
                            "GQ",
                            "GR",
                            "GT",
                            "GW",
                            "GY",
                            "HK",
                            "HN",
                            "HR",
                            "HT",
                            "HU",
                            "ID",
                            "IE",
                            "IL",
                            "IN",
                            "IQ",
                            "IS",
                            "IT",
                            "JM",
                            "JO",
                            "JP",
                            "KE",
                            "KG",
                            "KH",
                            "KI",
                            "KM",
                            "KN",
                            "KR",
                            "KW",
                            "KZ",
                            "LA",
                            "LB",
                            "LC",
                            "LI",
                            "LK",
                            "LR",
                            "LS",
                            "LT",
                            "LU",
                            "LV",
                            "LY",
                            "MA",
                            "MC",
                            "MD",
                            "ME",
                            "MG",
                            "MH",
                            "MK",
                            "ML",
                            "MN",
                            "MO",
                            "MR",
                            "MT",
                            "MU",
                            "MV",
                            "MW",
                            "MX",
                            "MY",
                            "MZ",
                            "NA",
                            "NE",
                            "NG",
                            "NI",
                            "NL",
                            "NO",
                            "NP",
                            "NR",
                            "NZ",
                            "OM",
                            "PA",
                            "PE",
                            "PG",
                            "PH",
                            "PK",
                            "PL",
                            "PS",
                            "PT",
                            "PW",
                            "PY",
                            "QA",
                            "RO",
                            "RS",
                            "RW",
                            "SA",
                            "SB",
                            "SC",
                            "SE",
                            "SG",
                            "SI",
                            "SK",
                            "SL",
                            "SM",
                            "SN",
                            "SR",
                            "ST",
                            "SV",
                            "SZ",
                            "TD",
                            "TG",
                            "TH",
                            "TJ",
                            "TL",
                            "TN",
                            "TO",
                            "TR",
                            "TT",
                            "TV",
                            "TW",
                            "TZ",
                            "UA",
                            "UG",
                            "US",
                            "UY",
                            "UZ",
                            "VC",
                            "VE",
                            "VN",
                            "VU",
                            "WS",
                            "XK",
                            "ZA",
                            "ZM",
                            "ZW",
                        ],
                        "external_urls": {
                            "spotify": "https://open.spotify.com/album/5WupqgR68HfuHt3BMJtgun"
                        },
                        "href": "https://api.spotify.com/v1/albums/5WupqgR68HfuHt3BMJtgun",
                        "id": "5WupqgR68HfuHt3BMJtgun",
                        "images": [
                            {
                                "height": 640,
                                "url": "https://i.scdn.co/image/ab67616d0000b273f72f1e38e9bd48f18a17ed9b",
                                "width": 640,
                            },
                            {
                                "height": 300,
                                "url": "https://i.scdn.co/image/ab67616d00001e02f72f1e38e9bd48f18a17ed9b",
                                "width": 300,
                            },
                            {
                                "height": 64,
                                "url": "https://i.scdn.co/image/ab67616d00004851f72f1e38e9bd48f18a17ed9b",
                                "width": 64,
                            },
                        ],
                        "name": "Goodbye Yellow Brick Road (Remastered)",
                        "release_date": "1973-10-05",
                        "release_date_precision": "day",
                        "total_tracks": 17,
                        "type": "album",
                        "uri": "spotify:album:5WupqgR68HfuHt3BMJtgun",
                    },
                    "artists": [
                        {
                            "external_urls": {
                                "spotify": "https://open.spotify.com/artist/3PhoLpVuITZKcymswpck5b"
                            },
                            "href": "https://api.spotify.com/v1/artists/3PhoLpVuITZKcymswpck5b",
                            "id": "3PhoLpVuITZKcymswpck5b",
                            "name": "Elton John",
                            "type": "artist",
                            "uri": "spotify:artist:3PhoLpVuITZKcymswpck5b",
                        }
                    ],
                    "available_markets": [
                        "AD",
                        "AE",
                        "AG",
                        "AL",
                        "AM",
                        "AO",
                        "AR",
                        "AT",
                        "AU",
                        "AZ",
                        "BA",
                        "BB",
                        "BD",
                        "BE",
                        "BF",
                        "BG",
                        "BH",
                        "BI",
                        "BJ",
                        "BN",
                        "BO",
                        "BR",
                        "BS",
                        "BT",
                        "BW",
                        "BY",
                        "BZ",
                        "CA",
                        "CD",
                        "CG",
                        "CH",
                        "CI",
                        "CL",
                        "CM",
                        "CO",
                        "CR",
                        "CV",
                        "CW",
                        "CY",
                        "CZ",
                        "DE",
                        "DJ",
                        "DK",
                        "DM",
                        "DO",
                        "DZ",
                        "EC",
                        "EE",
                        "EG",
                        "ES",
                        "FI",
                        "FJ",
                        "FM",
                        "FR",
                        "GA",
                        "GB",
                        "GD",
                        "GE",
                        "GH",
                        "GM",
                        "GN",
                        "GQ",
                        "GR",
                        "GT",
                        "GW",
                        "GY",
                        "HK",
                        "HN",
                        "HR",
                        "HT",
                        "HU",
                        "ID",
                        "IE",
                        "IL",
                        "IN",
                        "IQ",
                        "IS",
                        "IT",
                        "JM",
                        "JO",
                        "JP",
                        "KE",
                        "KG",
                        "KH",
                        "KI",
                        "KM",
                        "KN",
                        "KR",
                        "KW",
                        "KZ",
                        "LA",
                        "LB",
                        "LC",
                        "LI",
                        "LK",
                        "LR",
                        "LS",
                        "LT",
                        "LU",
                        "LV",
                        "LY",
                        "MA",
                        "MC",
                        "MD",
                        "ME",
                        "MG",
                        "MH",
                        "MK",
                        "ML",
                        "MN",
                        "MO",
                        "MR",
                        "MT",
                        "MU",
                        "MV",
                        "MW",
                        "MX",
                        "MY",
                        "MZ",
                        "NA",
                        "NE",
                        "NG",
                        "NI",
                        "NL",
                        "NO",
                        "NP",
                        "NR",
                        "NZ",
                        "OM",
                        "PA",
                        "PE",
                        "PG",
                        "PH",
                        "PK",
                        "PL",
                        "PS",
                        "PT",
                        "PW",
                        "PY",
                        "QA",
                        "RO",
                        "RS",
                        "RW",
                        "SA",
                        "SB",
                        "SC",
                        "SE",
                        "SG",
                        "SI",
                        "SK",
                        "SL",
                        "SM",
                        "SN",
                        "SR",
                        "ST",
                        "SV",
                        "SZ",
                        "TD",
                        "TG",
                        "TH",
                        "TJ",
                        "TL",
                        "TN",
                        "TO",
                        "TR",
                        "TT",
                        "TV",
                        "TW",
                        "TZ",
                        "UA",
                        "UG",
                        "US",
                        "UY",
                        "UZ",
                        "VC",
                        "VE",
                        "VN",
                        "VU",
                        "WS",
                        "XK",
                        "ZA",
                        "ZM",
                        "ZW",
                    ],
                    "disc_number": 1,
                    "duration_ms": 308353,
                    "explicit": False,
                    "external_ids": {"isrc": "GBUM71304964"},
                    "external_urls": {
                        "spotify": "https://open.spotify.com/track/4gad1qhsqNL63n9OyCDsjL"
                    },
                    "href": "https://api.spotify.com/v1/tracks/4gad1qhsqNL63n9OyCDsjL",
                    "id": "4gad1qhsqNL63n9OyCDsjL",
                    "is_local": False,
                    "name": "All The Girls Love Alice - Remastered 2014",
                    "popularity": 45,
                    "preview_url": "https://p.scdn.co/mp3-preview/fdcce545f4ce057413266a15cdebbdd3459c443b?cid=e5c4075706824aa19d96b62e5fd1f059",
                    "track_number": 12,
                    "type": "track",
                    "uri": "spotify:track:4gad1qhsqNL63n9OyCDsjL",
                },
            }
        ],
        "limit": 1,
        "next": "https://api.spotify.com/v1/me/tracks?offset=1&limit=1",
        "offset": 0,
        "previous": None,
        "total": 669,
    }
    _expected = [
        (
            "All The Girls Love Alice - Remastered 2014",
            "Elton John",
            "4gad1qhsqNL63n9OyCDsjL",
        )
    ]
    _result = client_fixture.parse_users_liked_tracks(_example_)
    assert _result == _expected
