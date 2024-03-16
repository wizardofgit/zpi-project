import flask
import openai
import json

app = flask.Flask(__name__)


def verify_config_file():
    try:
        with open("config.json", "r") as file:
            try:
                json.load(file)["openai_api_credentials"]["api_key"]
            except KeyError:
                raise "No openai_key found in config file"
    except FileNotFoundError:
        raise "No config file found"


@app.route('/', methods=['GET', 'POST'])
def home_page():
    return flask.render_template("homepage.html")


@app.route('/generic_prompt', methods=['GET'])
def generic_prompt():
    prompt = "Generate 10-record patient database containing relevant patient information and their tests' results"

    open_ai_key = json.load(open("config.json", "r"))["openai_api_credentials"]["api_key"]
    client = openai.Client(api_key=open_ai_key)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {'role': 'system', 'content': prompt}
        ]
    )

    return response.choices[0].message.content


if __name__ == '__main__':
    verify_config_file()
    app.run()
