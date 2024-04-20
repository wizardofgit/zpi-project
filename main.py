import flask
import openai
import json
import io
import csv
import tempfile
from fpdf import FPDF
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re

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


def generate_pdf_from_data(headers, data_list):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=8)
    cell_width = 20  # Nowa szerokość komórki
    for header in headers:
        pdf.set_font("Arial", size=8, style='B')
        pdf.cell(cell_width, 10, header, border=1)
    pdf.ln()

    unique_data_set = set()

    # dlatego trzeba znalezc unikatowe, bo zauwazylem, ze kazdy wiersz sie zdublowal
    print('Ile mamy wierszy danych: ', len(data_list))
    for row in data_list:
        row_tuple = tuple(row)
        if row_tuple not in unique_data_set:
            for item in row:
                pdf.set_font("Arial", size=8)
                pdf.cell(cell_width, 10, str(item), border=1)
            pdf.ln()
            unique_data_set.add(row_tuple)

    pdf.ln()

    df = pd.DataFrame(unique_data_set, columns=headers)

    df.drop_duplicates(inplace=True)

    df = convert_to_numeric(df)

    df.info()

    for column in df.columns:
        if df[column].dtype in ['int64', 'float64', 'int32']:
            mean_val = df[column].mean()
            median_val = df[column].median()
            mode_val = df[column].mode()[0]
            std_val = df[column].std()

            pdf.set_font("Arial", size=12, style='B')
            pdf.cell(0, 10, f"Statystyki kolumny {column}:", ln=True)
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, f"srednia: {mean_val}", ln=True)
            pdf.cell(0, 10, f"mediana: {median_val}", ln=True)
            pdf.cell(0, 10, f"moda: {mode_val}", ln=True)
            pdf.cell(0, 10, f"odchylenie standardowe: {std_val}", ln=True)

            unique_values = df[column].nunique()
            if unique_values <= 2:
                # Tworzymy wykres kołowy dla zmiennych numerycznych
                plt.figure(figsize=(6, 4))
                df[column].value_counts().plot(kind='pie', autopct='%1.1f%%')
                plt.title(f"Wykres kołowy dla {column}")
                plt.legend(labels=df[column].unique(), loc="best")
                plt.axis('equal')

                for text in plt.gca().texts:
                    if '%' in text.get_text():
                        text.set_fontsize(12)
                        text.set_weight('bold')

                temp_image_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                plt.savefig(temp_image_file.name)
                plt.close()

                if pdf.get_y() > 200:
                    pdf.add_page()

                pdf.image(temp_image_file.name, x=pdf.l_margin, y=pdf.get_y(), w=140)
                pdf.ln(100)
            else:
                plt.figure(figsize=(6, 4))
                sns.boxplot(data=df, x=column)
                plt.title(f"Wykres pudełkowy dla {column}")
                plt.xlabel(column)
                plt.ylabel("Wartość")
                temp_image_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                plt.savefig(temp_image_file.name)
                plt.close()

                if pdf.get_y() > 200:
                    pdf.add_page()

                pdf.image(temp_image_file.name, x=pdf.l_margin, y=pdf.get_y(), w=140)
                pdf.ln(100)
        else:
            plt.figure(figsize=(6, 4))
            sns.histplot(data=df, x=column, discrete=True, stat='count')
            plt.title(f"Histogram dla {column}")
            plt.xlabel(column)
            plt.ylabel("Liczba wystąpień")

            values, counts = np.unique(df[column], return_counts=True)

            x_labels = range(len(values))

            plt.xticks(x_labels, [f'{i + 1}' for i in x_labels])
            pdf.set_font("Arial", size=12, style='B')
            pdf.cell(200, 10, txt=f"Legenda dla histogramu {column}", ln=True)
            pdf.set_font("Arial", size=9)
            for i, value in enumerate(values, start=1):
                pdf.cell(200, 10, txt=f"{i}: {value}", ln=True)

            temp_image_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            plt.savefig(temp_image_file.name)
            plt.close()

            if pdf.get_y() > 200:
                pdf.add_page()

            pdf.image(temp_image_file.name, x=pdf.l_margin, y=pdf.get_y(), w=140)

            pdf.ln(100)

    with tempfile.NamedTemporaryFile(mode='w+b', delete=False, suffix='.pdf') as temp_file:
        pdf.output(temp_file.name)

    return temp_file.name


def extract_numeric_value(text):
    match = re.search(r'(\d+\.?\d*)', text)
    if match:
        return float(match.group())
    else:
        return text


def convert_to_numeric(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: extract_numeric_value(x))
    return df


@app.route('/generate', methods=['GET', 'POST'])
def generate_prompt():
    global previous_data
    global headers

    if 'headers' in flask.request.args:
        headers = list(json.loads(flask.request.args['headers']))

    if flask.request.method == 'GET':
        return flask.render_template("prompt_ask.html")
    else:
        records_to_generate = int(flask.request.form['number_of_records'])
        prompt = flask.request.form.get("prompt")
        new_column_name = flask.request.form.get("new_column_name")
        new_entries = flask.request.form.get("new_entries")

        if new_column_name:
            headers.append(new_column_name)
            # headers.append(new_column_name)
            prompt = (
                f"Update this data set {previous_data} with new column {new_column_name}, please use ';' as separator."
                f"Make sure that none of the previous data is lost adn entries stay teh same except for addition of new column"
                f"Make sure that if there's a listing inside of a cell it's separated by a & eg: drug a & drug b."
                f"Remember to make sure that entries make sense and have realistic realtionships with each other")

        elif new_entries:
            records_to_generate += int(new_entries)
            prompt = (f"Generate {new_entries} with following headers: {headers}, please use ';' as separator."
                      f"Make sure not to add headers."
                      f"Make sure that new entries stay consistent with this data {previous_data} formatting wise"
                      # f"Make sure that if there's a listing inside of a cell it's separated by a & eg: drug a & drug b."
                      f"Remember to make sure that entries make sense and have realistic realtionships with each other")
        if not prompt:
            prompt = (f"Generate a synthetic patient database with {records_to_generate}"
                      f" records containing the following columns: {''.join(csv_header)}, please use ';' as separator."
                      f"Make sure that if there's a listing inside of a cell it's separated by a & eg: drug a & drug b"
                      f"Remember to make sure that entries make sense and have realistic realtionships with each other")
        # print(f"Prompt: {prompt}")

        client = openai.Client(api_key=API_KEY)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {'role': 'system',
                 'content': 'You are a data scientist working at a hospital. You need to generate a synthetic patient database '
                            'that contain relevant information about the patients and their test results. The database should contain '
                            'columns that are listed in the prompt. Make sure that the data is realistic and can be used for testing purposes.'
                            'It must contain realistic relationships between the columns. Do not use any type of numeration in the data.'},
                {'role': 'user', 'content': prompt}
            ]
        )

        print(headers)
        data_str = response.choices[0].message.content
        if new_entries:
            previous_data += "\n" + data_str + "\n"
        else:
            previous_data = data_str
        # print(previous_data)
        data_list = previous_data.split("\n")
        print(data_list)
        for i in range(len(data_list)):
            data_list[i] = data_list[i].split(";")
        data_list.pop(0)

        # making sure that chat returns the exact number of records requested
        # if not - ask for more data
        while len(data_list) - 1 < records_to_generate:
            if records_to_generate - records_to_generate >= 25:
                to_generate = 25
            else:
                to_generate = records_to_generate - len(data_list)

            prompt = (f"Generate {to_generate} with following headers: {headers}, please use ';' as separator."
                      f"Make sure not to add headers."
                      f"Make sure that new entries stay consistent with this data {previous_data} formatting wise"
                      # f"Make sure that if there's a listing inside of a cell it's separated by a & eg: drug a & drug b."
                      f"Remember to make sure that entries make sense and have realistic realtionships with each other")

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {'role': 'system',
                     'content': 'You are a data scientist working at a hospital. You need to generate a synthetic patient database '
                                'that contain relevant information about the patients and their test results. The database should contain '
                                'columns that are listed in the prompt. Make sure that the data is realistic and can be used for testing purposes.'
                                'It must contain realistic relationships between the columns. Do not use any type of numeration in the data.'},
                    {'role': 'user', 'content': prompt}
                ]
            )

            new_data_str = response.choices[0].message.content
            previous_data += new_data_str
            new_data_list = new_data_str.split("\n")
            for i in range(len(new_data_list)):
                data_list.append(new_data_list[i].split(";"))

            print(new_data_list)

        client.close()

        # remove empty records
        indices_to_remove = []
        for i in range(len(data_list)):
            if data_list[i] == '' or data_list[i] == '\n' or data_list[i] == ['']:
                indices_to_remove.append(i)
        for index in sorted(indices_to_remove, reverse=True):
            del data_list[index]
        # remove empty headers
        indices_to_remove = []
        for i in range(len(headers)):
            if headers[i] == '' or headers[i] == '\n' or headers[i] == ['']:
                indices_to_remove.append(i)
        for index in sorted(indices_to_remove, reverse=True):
            del headers[index]

        print("len of datalist:", len(data_list), " ", data_list)
        # remove spaces in records
        # Czy jest to potrzebne? Jeżeli tak to wypadałoby to zrobić tak by nie niszczyć tekstów ze spacjami. MS
        # for i in range(len(data_list)):
        #     for j in range(len(data_list[i])):
        #         data_list[i][j] = data_list[i][j].replace(" ", "")

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
            csv_writer = csv.writer(temp_file)
            csv_writer.writerow(headers)
            csv_writer.writerows(data_list)
            temp_file_path = temp_file.name

        # Wywołaj funkcję pomocniczą do generowania pliku PDF
        pdf_file_path = generate_pdf_from_data(headers, data_list)

        # Pass data to HTML template
        return flask.render_template("table.html", headers=headers, data=data_list[:records_to_generate],
                                     csv_file=temp_file_path,
                                     pdf_file=pdf_file_path,
                                     total_records=records_to_generate)


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
            print(csv_header)
            if ';' in csv_header[0]:
                csv_header = csv_header[0].split(';')
            del csv_reader
            del csv_data

            print(json.dumps(csv_header))

            return flask.redirect(f"/generate?headers={json.dumps(csv_header)}")
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
