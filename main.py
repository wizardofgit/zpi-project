import flask
import openai
import json
import io
import csv

app = flask.Flask(__name__)

API_KEY = json.load(open("config.json", "r"))["openai_api_credentials"]["api_key"]

def verify_config_file():
    """Verify if the config file exists and contains the openai_api_credentials"""
    try:
        with open("config.json", "r") as file:
            try:
                json.load(file)["openai_api_credentials"]["api_key"]
            except KeyError:
                raise "No openai_key found in config file"
    except FileNotFoundError:
        raise "No config file found"

@app.route('/generate', methods=['GET', 'POST'])
def generate_prompt():
    if flask.request.method == 'GET':
        return flask.render_template("prompt_ask.html")
    else:
        prompt = (f"Generate a synthetic pataient database with {int(flask.request.form['number_of_records'])}"
                  f" records containing the following columns: {', '.join(csv_header)}")
        print(f"Prompt: {prompt}")

        client = openai.Client(api_key=API_KEY)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {'role': 'system', 'content': 'You are a data scientist working at a hospital. You need to generate a synthetic patient database '
                                              'that contain relevant information about the patients and their test results. The database should contain '
                                              'columns that are listed in the prompt. Make sure that the data is realistic and can be used for testing purposes.'
                                              'It must contain realistic relationships between the columns.'},
                {'role': 'user', 'content': prompt}
            ]
        )

        client.close()

        return response.choices[0].message.content


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Upload a csv file with patient data and redirect to the page with the results of the GPT-3 API
    after reading the csv file and extracting the header."""

    if flask.request.method == 'GET':
        return flask.render_template("upload.html")
    else:
        file = flask.request.files['csv']

        if file and file.filename.endswith('.csv'):
            # Read the CSV file directly from the uploaded file object
            csv_data = io.StringIO(file.stream.read().decode("UTF8"), newline=None)

            csv_reader = csv.reader(csv_data)
            global csv_header
            csv_header = next(csv_reader)

            del csv_reader
            del csv_data

            return flask.redirect("/generate")
        else:
            return 'Please upload a CSV file'


@app.route('/', methods=['GET'])
def home_page():
    """Homepage of the website. Contains buttons:
    - Create a generic prompt (for testing purposes)
    - Upload csv file with patient data"""

    return flask.render_template("homepage.html")


@app.route('/generic_prompt', methods=['GET'])
def generic_prompt():
    """Testing site for the GPT-3 API that allows to create a generic prompt"""

    prompt = "Generate 10-record patient database containing relevant patient information and their tests' results"

    client = openai.Client(api_key=API_KEY)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {'role': 'system', 'content': prompt}
        ]
    )

    client.close()

    return response.choices[0].message.content


if __name__ == '__main__':
    verify_config_file()
    app.run()
