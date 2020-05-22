import os
import json
from threading import Thread
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename

import search_protests
from retrieving_images import retrieve_images
from protest_classification import classify_protest
from search_protests import protest_seeker

app = Flask(__name__)
app.config['BASE_DIR'] = r""
app.config['STATIC_SOURCE'] = os.path.join(app.config['BASE_DIR'], r"server\static")
app.config['UPLOAD_FOLDER'] = os.path.join(app.config['BASE_DIR'], r"server\upload")
app.config['PROTEST_SOURCE'] = os.path.join(app.config['BASE_DIR'], r"data\protests_researched")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

IMAGE_TYPES = ["png", "jpg", "jpeg"]
WANTED_STATS = ["violence", "fire", "police"]
MAX_SEARCH_RESULTS = 5
WITH_RESEARCH = True


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in IMAGE_TYPES


@app.route('/', methods=['POST', 'GET'])
def main_page():
    app.config['PLACES'] = protest_seeker.get_places()
    if request.method == 'POST':
        if len(request.form) == 1:
            loc_key = request.form['location']
            return redirect(url_for('location_summary', requested_loc=loc_key))
        elif len(request.form) == 3:
            requested_location = retrieve_images.Location._make(request.form['requested_location'].split(':'))
            cycles = int(request.form['num_cycles'])
            location_thread = Thread(target=protest_seeker.seek_protests, args=(app.config['LOADER'], requested_location, cycles))
            location_thread.start()
            flash('Your research has started')
        else:
            flash('Please fill the form')
    return render_template('choose_location.html', keys=[l.name for l in app.config['PLACES']])


@app.route('/location_summary/<requested_loc>')
def location_summary(requested_loc):
    loc = get_location_by_name(requested_loc)
    if loc is None:
        flash("The requested location was not researched.")
        return redirect(url_for('/'))
    protest_summary = get_protest_from_path(str(loc.id))
    labels = ["/".join(str(prot_date).split('-')) for prot_date in protest_summary.keys()]
    data = [len(images[0]) for images in protest_summary.values()]
    protest_summary = reversed(sorted(protest_summary.items()))
    return render_template('location_summary.html', location=loc,
                           protest=protest_summary, num_images=data, labels=labels)


@app.route('/request_protest_image/<protest_image>')
def request_protest_image(protest_image):
    img_source = os.path.join(app.config["PROTEST_SOURCE"], protest_image)
    splited_path = img_source.split('\\')
    return send_from_directory('\\'.join(splited_path[:-1]), splited_path[-1])


@app.route('/query_location/<q_string>')
def query_location(q_string):
    locations = retrieve_images.get_search_results(app.config['LOADER'], q_string, MAX_SEARCH_RESULTS)
    data = [(l.name, ":".join([str(i) for i in l])) for l in locations]
    return json.dumps(data, ensure_ascii=False)


@app.route('/favicon.ico')
def get_icon():
    return send_from_directory(app.config['STATIC_SOURCE'], "favicon.ico")


def get_stat_protest(absolute_dir_path):
    stats = {}
    max_people = -1
    if os.path.isdir(absolute_dir_path):
        stats_file_path = os.path.join(absolute_dir_path, "stats.txt")
        with open(stats_file_path, 'r') as stats_file:
            for line in stats_file.readlines():
                key, value = line.split(":")
                if key == "max_people":
                    max_people = int(float(value[:-1]))
                elif key in WANTED_STATS:
                    stats[key] = float(value)
    return stats, max_people


def get_location_by_name(loc_name):
    places = protest_seeker.get_places()
    for p in places:
        if p.name == loc_name:
            return p
    return None


def get_protest_from_path(protest_code):
    protest_path = os.path.join(app.config['PROTEST_SOURCE'], protest_code)
    protest = {}
    for date in os.listdir(protest_path):
        relative_dir_path = os.path.join(protest_code, date)
        absolute_dir_path = os.path.join(protest_path, date)
        if os.path.isdir(absolute_dir_path):
            imgs = {}
            for img in os.listdir(absolute_dir_path):
                relative_img_path = os.path.join(relative_dir_path, img)
                absolute_img_path = os.path.join(absolute_dir_path, img)
                if os.path.isdir(absolute_img_path):
                    continue
                if img.split('.')[1] in IMAGE_TYPES:
                    absolute_sign_dir = absolute_img_path.split('.')[0]
                    relative_sign_dir = relative_img_path.split('.')[0]
                    sign_paths = []
                    if os.path.isdir(absolute_sign_dir):
                        for sign in os.listdir(absolute_sign_dir):
                            sign_path = os.path.join(relative_sign_dir, sign)
                            sign_paths.append(sign_path)
                    imgs[relative_img_path] = sign_paths
            stats, max_people = get_stat_protest(absolute_dir_path)
            protest[date] = (imgs, stats, max_people)
    return protest


if __name__ == '__main__':
    app.secret_key = ''
    if WITH_RESEARCH:
        app.config['MODEL'] = classify_protest.get_model(search_protests.protest_seeker.PROTEST_MODEL_PATH)
        app.config['LOADER'] = retrieve_images.get_loader(retrieve_images.USERNAME, retrieve_images.PASSWORD, quiet=True)
    app.run(host='0.0.0.0', port=80)
