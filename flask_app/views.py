from flask_app import app
from flask import render_template, request, jsonify, make_response
import os
import json
import model

@app.route("/")
@app.route("/index.html")
@app.route("/index")
def front_page():
    """ Render front page template """
    return render_template('index.html', title="Watch_and_learn")

@app.route("/about")
def about():
    """ Render slides page """
    return render_template("about.html", title="Watch_and_learn")

@app.route('/similarity_search', methods=['POST'])
def similarity_search():
    """ Accept user-provided image upload and classify """
    image = request.files['file']
    if image and valid_filename(image.filename):
        try:
           image_urls = model.predict_similar_images(image)
           #print "images urls are: ", image_urls
           return make_response(json.dumps(image_urls))
        except IOError:
           return json_error("Invalid image file or bad upload")
    else:
        return json_error("Invalid image file")

def valid_filename(filename):
    """ Return if a file is valid based on extension """
    valid_ext = [".jpg", ".png", ".gif", ".jpeg"]
    return os.path.splitext(filename)[-1] in valid_ext

def json_error(message):
    response = jsonify(message=message)
    response.status_code = 500
    return response
