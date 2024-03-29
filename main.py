import flask
import openai
import json
import io
import csv
import tempfile
from fpdf import FPDF
from matplotlib import pyplot as plt

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


def calculate_summary(data_list, headers):
    male_heights = []
    male_weights = []
    female_heights = []
    female_weights = []

    weight_index = None
    height_index = None
    gender_index = None

    for i, column_name in enumerate(headers):
        if 'weight' in column_name.lower():
            weight_index = i
        elif 'height' in column_name.lower():
            height_index = i
        elif 'gender' in column_name.lower() or 'sex' in column_name.lower():
            gender_index = i

    for record in data_list:
        try:
            if weight_index is not None:
                weight_str = record[weight_index]
                weight = int(weight_str.split()[0])
                male_weights.append(weight) if record[gender_index].strip().lower() in ['male',
                                                                                        'm'] else female_weights.append(
                    weight)

            if height_index is not None:
                height_str = record[height_index]
                height = int(height_str.split()[0])
                male_heights.append(height) if record[gender_index].strip().lower() in ['male',
                                                                                        'm'] else female_heights.append(
                    height)
        except (ValueError, IndexError):
            continue

    num_male_records = max(len(male_heights), len(male_weights))
    num_female_records = max(len(female_heights), len(female_weights))

    average_male_weight = round(sum(male_weights) / len(male_weights), 2) if male_weights else 0
    average_male_height = round(sum(male_heights) / len(male_heights), 2) if male_heights else 0

    average_female_weight = round(sum(female_weights) / len(female_weights), 2) if female_weights else 0
    average_female_height = round(sum(female_heights) / len(female_heights), 2) if female_heights else 0

    return num_male_records, average_male_height, average_male_weight, num_female_records, average_female_height, average_female_weight


def generate_pdf_from_data(headers, data_list, summary_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for header in headers:
        pdf.set_font("Arial", size=12, style='B')
        pdf.cell(40, 10, header, border=1)
    pdf.ln()

    for row in data_list:
        for item in row:
            pdf.set_font("Arial", size=12)
            pdf.cell(40, 10, str(item), border=1)
        pdf.ln()

    pdf.ln()
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(0, 10, "Summary", ln=True)
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, f"Total number of records: {summary_data['total_records']}", ln=True)

    def replace_zero_with_dash(value):
        return "--" if value == 0 else str(value) + ' kg'

    def format_height(height):
        return f"{height} cm" if height >= 3 else f"{height} m"

    if summary_data['num_male_records'] > 0:
        pdf.cell(10)
        pdf.cell(0, 10, "a) Male:", ln=True)
        pdf.cell(20)
        pdf.cell(0, 10, f"- Number of records: {summary_data['num_male_records']}", ln=True)
        pdf.cell(20)
        pdf.cell(0, 10, f"- Average weight: {replace_zero_with_dash(summary_data['average_male_weight'])}", ln=True)
        pdf.cell(20)
        pdf.cell(0, 10, f"- Average height: {format_height(summary_data['average_male_height'])}", ln=True)

    if summary_data['num_female_records'] > 0:
        pdf.cell(10)
        pdf.cell(0, 10, "b) Female:", ln=True)
        pdf.cell(20)
        pdf.cell(0, 10, f"- Number of records: {summary_data['num_female_records']}", ln=True)
        pdf.cell(20)
        pdf.cell(0, 10, f"- Average weight: {replace_zero_with_dash(summary_data['average_female_weight'])}", ln=True)
        pdf.cell(20)
        pdf.cell(0, 10, f"- Average height: {format_height(summary_data['average_female_height'])}", ln=True)

    plt.figure(figsize=(6, 4))
    labels = []
    sizes = []

    if summary_data['num_male_records'] > 0:
        labels.append('Male')
        sizes.append(summary_data['num_male_records'])

    if summary_data['num_female_records'] > 0:
        labels.append('Female')
        sizes.append(summary_data['num_female_records'])

    colors = ['skyblue', 'pink']
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.axis('equal')
    plt.title('Gender Distribution', fontweight='bold')

    for text in plt.gca().texts:
        if '%' in text.get_text():
            text.set_fontsize(12)
            text.set_weight('bold')

    temp_image_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_image_file.name)
    plt.close()

    if pdf.get_y() > 200:
        pdf.add_page()

    pdf.image(temp_image_file.name, x=10, y=pdf.get_y(), w=140)
    pdf.ln()

    with tempfile.NamedTemporaryFile(mode='w+b', delete=False, suffix='.pdf') as temp_file:
        pdf.output(temp_file.name)

    return temp_file.name


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

        # szybko naprawilem zeby wczytywac obydwa csv
        headers = ', '.join(csv_header).split(';')
        if len(headers) == 1:
            headers = headers[0].split(', ')
        data_str = response.choices[0].message.content
        data_list = data_str.split("\n")
        for i in range(len(data_list)):
            data_list[i] = data_list[i].split(",")
        data_list.pop(0)

        num_male_records, average_male_height, average_male_weight, num_female_records, average_female_height, average_female_weight = calculate_summary(
            data_list, headers)
        total_records = len(data_list)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
            csv_writer = csv.writer(temp_file)
            csv_writer.writerow(headers)
            csv_writer.writerows(data_list)
            temp_file_path = temp_file.name

            # Wywołaj funkcję pomocniczą do generowania pliku PDF
            # Wywołaj funkcję pomocniczą do generowania pliku PDF
            pdf_file_path = generate_pdf_from_data(headers, data_list, {
                'total_records': total_records,
                'num_male_records': num_male_records,
                'average_male_weight': average_male_weight,
                'average_male_height': average_male_height,
                'num_female_records': num_female_records,
                'average_female_weight': average_female_weight,
                'average_female_height': average_female_height
            })

        # Pass data to HTML template
        return flask.render_template("table.html", headers=headers, data=data_list, csv_file=temp_file_path,
                                     pdf_file=pdf_file_path,
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


@app.route('/download_pdf')
def download_pdf():
    pdf_file = flask.request.args.get('pdf_file')
    if pdf_file:
        return flask.send_file(pdf_file, mimetype='text/pdf', as_attachment=True, download_name='generated_data.pdf')
    else:
        return "Error: PDF file not found"


if __name__ == '__main__':
    verify_config_file()
    app.run()
