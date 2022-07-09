c = get_config()

c.NbConvertApp.notebooks = ["Influxdb.ipynb"]
c.NbConvertApp.export_format = "latex"
c.NbConvertApp.postprocessor_class = "PDF"

c.Exporter.template_file = "nbconvert_latex_notebook_style.tplx"
