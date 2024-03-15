import flask
import openai
import json

app = flask.Flask(__name__)
global open_ai_key


def verify_config_file():
    try:
        with open("config.json", "r") as file:
            try:
                open_ai_key = json.load(file)["openai_key"]
            except KeyError:
                raise "No openai_key found in config file"
    except FileNotFoundError:
        raise "No config file found"


@app.route('/')
def home_page():
    return flask.render_template("homepage.html")


if __name__ == '__main__':
    verify_config_file()
    app.run()
