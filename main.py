import flask
import openai
import json
import io
import csv
import tempfile

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


def calculate_summary(data_list):
    male_heights = []
    male_weights = []
    female_heights = []
    female_weights = []

    for record in data_list:
        try:
            weight_str, height_str, gender = record

            weight = int(weight_str.split()[0])
            height = int(height_str.split()[0])
            gender = gender.strip().lower()

            if gender == 'male':
                male_heights.append(height)
                male_weights.append(weight)
            elif gender == 'female':
                female_heights.append(height)
                female_weights.append(weight)
        except (ValueError, IndexError) as e:
            continue

    num_male_records = len(male_heights)
    num_female_records = len(female_heights)

    print(f"Number of male records: {num_male_records}, Male heights: {male_heights}, Male weights: {male_weights}")
    print(
        f"Number of female records: {num_female_records}, Female heights: {female_heights}, Female weights: {female_weights}")

    average_male_height = round(sum(male_heights) / num_male_records, 2) if num_male_records > 0 else 0
    average_male_weight = round(sum(male_weights) / num_male_records, 2) if num_male_records > 0 else 0

    average_female_height = round(sum(female_heights) / num_female_records, 2) if num_female_records > 0 else 0
    average_female_weight = round(sum(female_weights) / num_female_records, 2) if num_female_records > 0 else 0

    return num_male_records, average_male_height, average_male_weight, num_female_records, average_female_height, average_female_weight


@app.route('/generate', methods=['GET', 'POST'])
def generate_prompt():
    if flask.request.method == 'GET':
        return flask.render_template("prompt_ask.html")
    else:
        prompt = (f"Generate a synthetic patient database with {int(flask.request.form['number_of_records'])}"
                  f" records containing the following columns: {', '.join(csv_header)}, please use ',' as separator.")
        print(f"Prompt: {prompt}")

        client = openai.Client(api_key=API_KEY)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {'role': 'system',
                 'content': 'You are a data scientist working at a hospital. You need to generate a synthetic patient database '
                            'that contain relevant information about the patients and their test results. The database should contain '
                            'columns that are listed in the prompt. Make sure that the data is realistic and can be used for testing purposes.'
                            'It must contain realistic relationships between the columns.'},
                {'role': 'user', 'content': prompt}
            ]
        )

        client.close()

        headers = ', '.join(csv_header).split(';')
        data_str = response.choices[0].message.content
        data_list = data_str.split("\n")
        for i in range(len(data_list)):
            data_list[i] = data_list[i].split(",")
        data_list.pop(0)

        num_male_records, average_male_height, average_male_weight, num_female_records, average_female_height, average_female_weight = calculate_summary(
            data_list)
        total_records = len(data_list)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
            csv_writer = csv.writer(temp_file)
            csv_writer.writerow(headers)
            csv_writer.writerows(data_list)
            temp_file_path = temp_file.name
        print(headers, data_list)

        # Pass data to HTML template
        return flask.render_template("table.html", headers=headers, data=data_list, csv_file=temp_file_path,
                                     total_records=total_records,
                                     num_male_records=num_male_records, average_male_weight=average_male_weight,
                                     average_male_height=average_male_height, num_female_records=num_female_records,
                                     average_female_weight=average_female_weight,
                                     average_female_height=average_female_height)


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


@app.route('/download_csv')
def download_csv():
    csv_file = flask.request.args.get('csv_file')
    if csv_file:
        return flask.send_file(csv_file, mimetype='text/csv', as_attachment=True, download_name='generated_data.csv')
    else:
        return "Error: CSV file not found"


if __name__ == '__main__':
    verify_config_file()
    app.run()
