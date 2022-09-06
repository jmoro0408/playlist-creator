import pandas as pd
from spotipy_interaction import SpotipyClient, import_config


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
    track_names = [x[0] for x in parsed_liked_tracks_json]
    artist_names = [x[1] for x in parsed_liked_tracks_json]
    track_ids = [
        x[2] for x in parsed_liked_tracks_json
    ]  # get the ID part of the parsed json tracks only
    audio_features = sp.get_track_features(track_ids)
    liked_song_df = pd.DataFrame(audio_features)
    liked_song_df["artist_names"] = artist_names
    liked_song_df["track_names"] = track_names
    return liked_song_df


if __name__ == "__main__":
    client_id, client_secret = import_config()
    sp = SpotipyClient(client_id, client_secret)
    liked_songs_df = build_liked_song_df(sp, 20)
    print(liked_songs_df.head())
