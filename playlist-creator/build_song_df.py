import itertools
from typing import Any

import pandas as pd
from spotipy_interaction import SpotipyClient, import_config


def explode_results_list(result_list: list[tuple[Any, Any, Any]]) -> tuple:
    track_names = [x[0] for x in result_list]
    artist_names = [x[1] for x in result_list]
    track_ids = [x[2] for x in result_list]
    return track_names, artist_names, track_ids


def build_liked_song_df(sp: SpotipyClient) -> pd.DataFrame:
    """Return a dataframe with the users liked tracks,
    with track name, artist, duration, and audio features for each track.
    The web API only allows audio features requests of up to 100 tracks
    at a time, so the full liked songs json has to be split into 100
    track chunks before querying, then combined together.

    Args:
        sp (SpotipyClient): authorised spotipy client object

    Returns:
        pd.DataFrame: dataframe with track details
    """

    def split_into_chunks(list_to_split, chunk_size):
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

    liked_tracks_json = sp.get_users_liked_tracks()
    parsed_liked_tracks_json = sp.parse_users_liked_tracks(liked_tracks_json)
    track_names, artist_names, track_ids = explode_results_list(
        parsed_liked_tracks_json
    )
    batch_size = 100  # API only allows up to 100 requests per batch
    track_id_chunks = list(split_into_chunks(track_ids, batch_size))
    audio_features = []
    for chunk in track_id_chunks:
        audio_features.append(sp.get_track_features(chunk))
    audio_features_flattened = list(
        itertools.chain.from_iterable(audio_features)
    )  # flattening list for df creation
    liked_song_df = pd.DataFrame(audio_features_flattened)
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
    playlists_df = pd.concat(playlist_dfs_list)
    playlists_df = playlists_df.reset_index(drop=True)
    return playlists_df


if __name__ == "__main__":
    client_id, client_secret = import_config()
    sp = SpotipyClient(client_id, client_secret)
    liked_songs_df = build_liked_song_df(sp)
    # playlists_df = build_playlists_df(sp)
    # liked_songs_df.to_pickle(Path("playlist-creator", "data", "liked_songs_df.pkl"))
    # playlists_df.to_pickle(Path("playlist-creator", "data", "playlist_df.pkl"))
