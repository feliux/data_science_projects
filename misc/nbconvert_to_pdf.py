# jupyter nbconvert --execute --config nbconvert_pdf.py

c = get_config()

c.ExecutePreprocessor.force_raise_errors = True

c.TemplateExporter.exclude_output_prompt = True
c.TemplateExporter.exclude_input_prompt = True

c.TemplateExporter.exclude_input = True
c.TemplateExporter.exclude_output = False

c.NbConvertApp.notebooks = ["Influxdb.ipynb", "Nbconvert.ipynb"]
c.NbConvertApp.export_format = "pdf"
