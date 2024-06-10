import matplotlib
import flask
import openai
import json
import csv
import re
import os
import io
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A2
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import seaborn as sns
import pandas as pd
import tempfile
import numpy as np

app = flask.Flask(__name__)

API_KEY = json.load(open(os.path.join(os.path.dirname(__file__), "config.json"), 'r'))["openai_api_credentials"][
    "api_key"]


def verify_config_file():
    """Verify if the config file exists and contains the openai_api_credentials"""
    try:
        with open(os.path.join(os.path.dirname(__file__), "config.json"), "r") as file:
            try:
                json.load(file)["openai_api_credentials"]["api_key"]
            except KeyError:
                raise "No openai_key found in config file"
    except FileNotFoundError:
        raise "No config file found"


def generate_pdf_from_data(headers, data_list, records_to_generate):
    matplotlib.use('Agg')
    print(len(data_list))
    # czasem jest za duzo wartosci w jednym wierszu, a czasem za malo
    # jesli za duzo to usuwam ostatnie wartosci tak zeby zgadzala sie ich ilosc z iloscia atrybutow
    # jesli jest ich za malo, to po prostu kopiuje wartosci z pierwszej kolumny tyle razy, ile trzeba
    data_list = [
        x[:len(headers)] + [x[0]] * (len(headers) - len(x)) if len(x) < len(headers) else x[:len(headers)] for x
        in data_list]
    df = pd.DataFrame(data_list, columns=headers)
    df = convert_to_numeric(df)
    chart_streams = []

    for column in df.columns:
        create_box_plot_pdf(chart_streams, df, column)

    for column in df.columns:
        create_pie_chart_pdf(chart_streams, df, column)

    for column in df.columns:
        create_hist_plot_pdf(chart_streams, df, column)

    numerical_columns = df.select_dtypes(include=['float64', 'int64', 'int32']).columns
    non_numerical_columns = df.select_dtypes(exclude=['float64', 'int64', 'int32']).columns
    if len(non_numerical_columns) > 2:
        non_numerical_columns = np.random.choice(non_numerical_columns, size=2, replace=False)

    # pairplots
    if numerical_columns.empty:
        print("Brak zmiennych numerycznych do wygenerowania pairplota.")
    else:
        pairplot = sns.pairplot(df)
        pairplot_stream = io.BytesIO()
        pairplot.savefig(pairplot_stream, format='png')
        pairplot_stream.seek(0)
        chart_streams.append(('Seaborn Pairplot', pairplot_stream))

        for col in non_numerical_columns:
            pairplot_hue = sns.pairplot(df, hue=col)
            pairplot_hue_stream = io.BytesIO()
            pairplot_hue.savefig(pairplot_hue_stream, format='png')
            pairplot_hue_stream.seek(0)
            chart_streams.append((f'Seaborn Pairplot_{col}', pairplot_hue_stream))

    # Tworzenie PDF ze spisem treści
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    doc = SimpleDocTemplate(temp_file.name, pagesize=A2)
    page_width, page_height = A2
    styles = getSampleStyleSheet()
    flowables = []

    toc = [('Tabela danych', 'tabela')] + chart_streams

    flowables.append(Paragraph('<font size="70">Spis tresci</font>', styles['Title']))  # Zwiększona czcionka
    flowables.append(Spacer(1, 120))

    for idx, (title, _) in enumerate(toc, start=1):
        link_style = styles['Normal']
        link_style.fontSize = 15
        link_style.alignment = 1
        flowables.append(Paragraph(f'<a href="#{idx}">{idx}. {title}</a>', link_style))
        flowables.append(Spacer(1, 16))

    flowables.append(PageBreak())
    flowables.append(Paragraph(f'<a name="1"></a>', styles['Normal']))
    flowables.append(Paragraph('1. Tabela danych', styles['Title']))
    flowables.append(Spacer(1, 12))
    flowables.append(add_table(df))

    for idx, (title, chart_stream) in enumerate(toc[1:], start=2):
        flowables.append(PageBreak())
        flowables.append(Paragraph(f'<a name="{idx}"></a>', styles['Normal']))
        flowables.append(Paragraph(f'{idx}. {title}', styles['Title']))
        flowables.append(Spacer(1, 12))

        image = Image(chart_stream)
        image_width, image_height = image.drawWidth, image.drawHeight
        scale_x = page_width / image_width
        scale_y = page_height / image_height
        scale_factor = min(scale_x, scale_y)
        new_width = image_width * scale_factor * 0.95
        new_height = image_height * scale_factor * 0.95

        image.drawWidth = new_width
        image.drawHeight = new_height
        flowables.append(image)
        flowables.append(Spacer(1, 12))

    # Generowanie PDF
    doc.build(flowables)
    return temp_file.name


def add_table(data):
    header = list(data.keys())
    rows = [header]
    for i in range(len(data[header[0]])):
        row = [truncate_and_split_text(str(data[key][i])) for key in header]
        rows.append(row)
    table = Table(rows)
    max_page_width = A2[0] - 2 * inch
    num_columns = len(header)
    col_width = max_page_width / num_columns

    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 2),
        ('RIGHTPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ])

    table.setStyle(style)
    table._argW = [col_width] * num_columns
    return table


def create_box_plot_pdf(chart_streams, df, column):
    if df[column].dtype in ['int64', 'float64', 'int32']:
        mean_val = df[column].mean()
        median_val = df[column].median()
        mode_val = df[column].mode()[0]
        std_val = df[column].std()
        fig = plt.figure(figsize=(8, 10))

        ax1 = fig.add_subplot(3, 1, 1)
        sns.boxplot(data=df, x=column, ax=ax1)
        ax1.set_title(f"Wykres pudełkowy dla {column}")
        ax1.set_xlabel(column)
        ax1.set_ylabel("Wartość")

        ax2 = fig.add_subplot(3, 1, 2)
        info_text = f"Średnia: {round(mean_val, 2)}\nMediana: {round(median_val, 2)}\nModa: {round(mode_val, 2)}\nOdchylenie standardowe: {round(std_val, 2)}"
        ax2.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=12)
        ax2.axis('off')
        histogram_stream = io.BytesIO()
        plt.savefig(histogram_stream, format='png')
        histogram_stream.seek(0)
        chart_streams.append(
            (f'BoxPlot_{column}', histogram_stream))


def create_pie_chart_pdf(chart_streams, df, column):
    if df[column].dtype not in ['int64', 'float64', 'int32']:
        unique_values = df[column].nunique()
        if unique_values <= 2:
            fig = plt.figure(figsize=(8, 10))
            ax1 = fig.add_subplot(1, 1, 1)
            df[column].value_counts().plot(kind='pie', autopct='%1.1f%%')
            ax1.set_title(f"Wykres kołowy dla {column}")
            ax1.legend(labels=df[column].unique(), loc="best")
            plt.axis('equal')
            histogram_stream = io.BytesIO()
            plt.savefig(histogram_stream, format='png')
            histogram_stream.seek(0)
            chart_streams.append(
                (f'PieChart_{column}', histogram_stream))


def create_hist_plot_pdf(chart_streams, df, column):
    if df[column].dtype not in ['int64', 'float64', 'int32']:
        unique_values = df[column].nunique()
        if unique_values > 2:
            fig = plt.figure(figsize=(8, 10))
            ax1 = fig.add_subplot(1, 1, 1)
            sns.histplot(data=df, x=column, discrete=True, stat='count')
            ax1.set_title(f"Histogram dla {column}")
            ax1.set_xlabel(column)
            ax1.set_ylabel("Liczba wystąpień")

            max_label_length = 13

            unique_values = df[column].unique()
            xtick_labels = [str(val) for val in unique_values]

            truncated_xtick_labels = [label[:max_label_length] + '...' if len(label) > max_label_length else label
                                      for label in xtick_labels]

            ax1.set_xticks(unique_values)
            ax1.set_xticklabels(truncated_xtick_labels)
            ax1.tick_params(axis='x', labelsize=10, rotation=30)

            histogram_stream = io.BytesIO()
            plt.savefig(histogram_stream, format='png')
            histogram_stream.seek(0)
            chart_streams.append(
                (f'HistogramChart_{column}', histogram_stream))


def truncate_and_split_text(text):
    max_length = 13
    truncated_text = text[:max_length] + '...' if len(text) > max_length else text
    return truncated_text


def extract_numeric_value(text):
    try:
        if len(text) > 12:
            return text

        match = re.match(r'^\d+\.?\d*', text)
        if match:
            numeric_value = float(match.group(0))
            return numeric_value
        else:
            return text
    except (AttributeError, ValueError):
        return text


def all_values_are_numeric(column):
    return all(isinstance(val, (int, float)) for val in column)


def convert_to_numeric(df):
    for col in df.columns:
        if all_values_are_numeric(df[col].apply(lambda x: extract_numeric_value(x))):
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
        print('tyle jest', len(data_list))
        print('tyle ma byc', records_to_generate)

        if len(data_list) > records_to_generate:
            data_list = data_list[:records_to_generate]
        print('po usunieciu nadmiarowych', len(data_list))

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
        pdf_file_path = generate_pdf_from_data(headers, data_list, records_to_generate)
        print(pdf_file_path)
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
