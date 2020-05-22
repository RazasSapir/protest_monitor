import os
from collections import namedtuple
from instaloader import Instaloader
from instaloader.structures import TopSearchResults

USERNAME = ""
PASSWORD = ""
LOCATION_SCEME = ["name", "id", "lat", "lng"]
Location = namedtuple("Location", LOCATION_SCEME)


def get_loader(username, password, quiet=False):
    loader = Instaloader(
        quiet=quiet,
        download_videos=False,
        download_comments=False,
        download_geotags=False,
        download_video_thumbnails=False
    )
    loader.login(username, password)
    return loader


def download_from_geo(loader, geo_loc, target_dir):
    img_gen = loader.get_location_posts(geo_loc)
    for i, img in enumerate(img_gen):
        img_name = str(img.date).replace(':', '-') + " " + img.owner_username
        success = loader.download_pic(os.path.join(target_dir, img_name), img.url, img.date)
        print(str(i+1) + ". " + img_name + ", saved: " + str(success))


def get_search_results(loader, search_query, max_results=5):
    search_results = []
    search_gen = TopSearchResults(loader.context, search_query).get_locations()
    try:
        for i, search in enumerate(search_gen):
            if i >= max_results:
                break
            search_results.append(Location(search.name, search.id, search.lat, search.lng))
    except (GeneratorExit, KeyError) as e:
        pass
    return search_results


def main():
    pass


if __name__ == "__main__":
    main()
