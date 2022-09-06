from typing import Any

import pandas as pd
from spotipy_interaction import SpotipyClient, import_config


def explode_results_list(result_list: list[tuple[Any, Any, Any]]) -> tuple:
    track_names = [x[0] for x in result_list]
    artist_names = [x[1] for x in result_list]
    track_ids = [x[2] for x in result_list]
    return track_names, artist_names, track_ids


def build_liked_song_df(sp: SpotipyClient, limit: int = 10) -> pd.DataFrame:
    """Return a dataframe with the users liked tracks,
    with track name, artist, duration, and audio features for each track

    Args:
        sp (SpotipyClient): authorised spotipy client object

    Returns:
        pd.DataFrame: dataframe with track details
    """
    liked_tracks_json = sp.get_users_liked_tracks(limit=limit)
    # build df for one song first
    parsed_liked_tracks_json = sp.parse_users_liked_tracks(liked_tracks_json)
    # get audio features for tracks
    track_names, artist_names, track_ids = explode_results_list(
        parsed_liked_tracks_json
    )
    audio_features = sp.get_track_features(track_ids)
    liked_song_df = pd.DataFrame(audio_features)
    liked_song_df["artist_names"] = artist_names
    liked_song_df["track_names"] = track_names
    return liked_song_df


def build_playlists_df(sp: SpotipyClient) -> pd.DataFrame:
    # get users playlists
    user_playlists = sp.get_users_playlists_info()
    playlist_dfs_list = []
    for playlist_name, playlist_id in user_playlists.items():
        playlist_tracks = sp.get_user_playlist_track_info(playlist_id)  # type: ignore
        track_names, artist_names, track_ids = explode_results_list(playlist_tracks)
        try:
            audio_features = sp.get_track_features(track_ids)
            # Audio features fails when track_id list is all None.
            # This happens when a playlist is made up entirely of local files
        except AttributeError:
            continue
        playlist_song_df = pd.DataFrame(audio_features)
        playlist_song_df["artist_names"] = artist_names
        playlist_song_df["track_names"] = track_names
        playlist_song_df["playlist_name"] = playlist_name
        playlist_dfs_list.append(playlist_song_df)
    return pd.concat(playlist_dfs_list)


if __name__ == "__main__":
    client_id, client_secret = import_config()
    sp = SpotipyClient(client_id, client_secret)
    liked_songs_df = build_liked_song_df(sp, 20)
    playlist_df = build_playlists_df(sp)
    print("test")
