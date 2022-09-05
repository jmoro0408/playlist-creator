from spotipy_interaction import import_config, SpotipyClient
import pandas as pd

def build_liked_song_df(sp:SpotipyClient) -> pd.DataFrame:
    """Return a dataframe with the users liked tracks,
    with artist, duration, and audio features for each track

    Args:
        sp (SpotipyClient): authorised spotipy client object

    Returns:
        pd.DataFrame: dataframe with track details
    """
    liked_tracks_json =  sp.get_users_liked_tracks()
    return sp.parse_users_liked_tracks(liked_tracks_json)


if __name__ == "__main__":
    client_id, client_secret = import_config()
    sp = SpotipyClient(client_id, client_secret)

    liked_songs_test = build_liked_song_df(sp)
    print(liked_songs_test)
