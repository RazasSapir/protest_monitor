import csv
import queue
from retrieving_images import retrieve_images
from shutil import move
from time import sleep
from threading import Thread
from PIL import Image

from protest_classification import classify_protest
from sign_detection import signs_detector
from counting_people import people_counter
from search_protests.constants import *


downloaded_images_queue = queue.Queue()  # First queue - posts
protest_post_queue = queue.Queue()  # Second queue - analyzed images
protests_queue = queue.Queue()  # Third queue - batched protests

running = {}


def retrieve_from_geo(loader, location_id, target_dir, cycles):
    print("Start Retrieve")
    image_gen = loader.get_location_posts(location_id)
    num_downloaded = 1
    while running["retrieve_thread"]:
        if num_downloaded >= cycles * CYCLE_SIZE:
            running["retrieve_thread"] = False
            print("Stopping Retrieve")
        try:
            img = next(image_gen)
        except (StopIteration, GeneratorExit) as e:
            running["retrieve_thread"] = False
            print("No more images: " + str(e))
        img_path_no_type = os.path.join(target_dir, img.shortcode)
        if loader.download_pic(img_path_no_type, img.url, img.date):
            print("Downloaded: " + img.shortcode + ".jpg.")
        else:
            print(img.shortcode + ".jpg was already downloaded.")
        img_path = img_path_no_type + ".jpg"
        curr_post = Post(img_path, img.shortcode, img.date, img.owner_username)
        downloaded_images_queue.put(curr_post)
        num_downloaded += 1


def eval_protests(csv_path, protest_model):
    print("Start Eval")
    while running["eval_thread"]:
        if downloaded_images_queue.empty():
            if not running["retrieve_thread"]:
                running["eval_thread"] = False
                print("Stopping Eval")
            sleep(SLEEP_TIME)
        else:
            curr_post = downloaded_images_queue.get()
            print("Analyzing: " + str(curr_post.shortcode))
            protest_analysis = classify_protest.eval_image(protest_model, curr_post.path)
            curr_protest = Protest._make([curr_post] + list(protest_analysis.values()))
            protest_post_queue.put(curr_protest)
            print(str(curr_protest.post.shortcode) + ": " + str(curr_protest.protest))
            with open(csv_path, 'a') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter="\t")
                log_post = list(curr_protest.post) + list(curr_protest)[1:]
                csv_writer.writerow(log_post)


def identify_protests():
    curr_protest = []
    last_process_post = None
    print("Start Identify")
    while running["find_protests_thread"]:
        if protest_post_queue.empty():
            if not running["eval_thread"]:
                if not len(curr_protest) == 0:
                    protests_queue.put(curr_protest)
                running["find_protests_thread"] = False
                print("Stopping Identify")
            sleep(SLEEP_TIME)
        else:
            curr_protest_post = protest_post_queue.get()
            if last_process_post is None:
                last_process_post = curr_protest_post
            if curr_protest_post.post.datetime.date() == last_process_post.post.datetime.date():
                if curr_protest_post.protest > PROTEST_THRESHOLD:
                    curr_protest.append(curr_protest_post)
                else:
                    print("Deleting: " + str(curr_protest_post.post.shortcode))
                    os.remove(curr_protest_post.post.path)
            else:
                if not len(curr_protest) == 0:
                    protests_queue.put(curr_protest)
                curr_protest = []
                if curr_protest_post.protest > PROTEST_THRESHOLD:
                    curr_protest.append(curr_protest_post)
                else:
                    print("Deleting: " + str(curr_protest_post.post.shortcode))
                    os.remove(curr_protest_post.post.path)
            last_process_post = curr_protest_post


def handle_protests(target_dir, yolo_model, crowd_model):
    print("Start Saving")
    while running["save_protest_thread"]:
        if protests_queue.empty():
            if not running["find_protests_thread"]:
                running["save_protest_thread"] = False
                print("Stopping Save")
            sleep(SLEEP_TIME)
        else:
            curr_protest = protests_queue.get()
            curr_protest_dir = os.path.join(target_dir, str(curr_protest[0].post.datetime.date()).replace(':', '-'))
            os.mkdir(curr_protest_dir)
            sum_stats = []
            max_people = 0
            for protest_post in curr_protest:
                print("Looking for signs in: " + protest_post.post.shortcode)
                new_location = os.path.join(curr_protest_dir, os.path.basename(protest_post.post.path))
                os.rename(protest_post.post.path, new_location)
                signs_in_post = signs_detector.find_signs(new_location)
                print("Saved new protest with " + str(len(signs_in_post)) + " signs in it.")
                if not len(signs_in_post) == 0:
                    sign_dir = os.path.join(curr_protest_dir, protest_post.post.shortcode)
                    os.mkdir(sign_dir)
                    for i, sign in enumerate(signs_in_post):
                        sign_image = Image.fromarray(sign)
                        sign_path = os.path.join(sign_dir, "sign" + str(i) + ".jpg")
                        sign_image.save(sign_path)
                max_people = max(max_people, people_counter.count_people(new_location, yolo_model, crowd_model))
                if len(sum_stats) == 0:
                    sum_stats = list(protest_post)[1:]
                else:
                    sum_stats = [i + j for i, j in zip(sum_stats, list(protest_post)[1:])]
            avg_stats = [str(stat * 1.0 / len(curr_protest)) for stat in sum_stats] + [max_people]
            columns = list(PROTEST_SCHEME)[1:] + ["max_people"]
            save_stats = [key + ":" + str(round(float(value), 2)) + "\n" for key, value in zip(columns, avg_stats)]
            with open(os.path.join(curr_protest_dir, "stats.txt"), 'a') as stat_file:
                stat_file.writelines(save_stats)


def seek_protests(loader, requested_location, cycles):
    global running
    running = {}
    target_dir = os.path.join(BASE_DOWNLOAD_DIR, str(requested_location.id))
    protest_pred_model = classify_protest.get_model(PROTEST_MODEL_PATH)
    yolo_model, crowd_model = people_counter.get_models(YOLO_MODEL_PATH, CROWD_MODEL_PATH)
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
        csv_path = os.path.join(target_dir, str(requested_location.id) + "_analysis.txt")
        with open(csv_path, "a") as csv_file:  # create file
            csv_writer = csv.writer(csv_file, delimiter='\t')
            csv_writer.writerow(POST_SCHEME + PROTEST_SCHEME[1:])
        threads = {
            "retrieve_thread": Thread(target=retrieve_from_geo,
                                      args=(loader, requested_location.id, target_dir, cycles)),
            "eval_thread": Thread(target=eval_protests,
                                  args=(csv_path, protest_pred_model)),
            "find_protests_thread": Thread(target=identify_protests),
            "save_protest_thread": Thread(target=handle_protests,
                                          args=(target_dir, yolo_model, crowd_model))}
        for thread_key in threads:
            running[thread_key] = True
            threads[thread_key].start()
            sleep(10)
        for thread_key in threads:
            threads[thread_key].join()
        places = get_places()
        places.append(requested_location)
        move(target_dir, FINAL_PROTEST_DIR)
        set_places(places)
        print("Finished Research")


def set_places(new_places):
    with open(PLACES_PATH, 'w', encoding="utf8") as new_places_file:
        writer = csv.writer(new_places_file, delimiter="\t")
        for p in new_places:
            writer.writerow(list(p))


def get_places():
    places = []
    with open(PLACES_PATH, 'r', encoding="utf8") as places_file:
        reader = csv.reader(places_file, delimiter='\t')
        for row in reader:
            if not len(row) == 0:
                places.append(retrieve_images.Location._make(row))
    return places


def main():
    pass


if __name__ == '__main__':
    main()
