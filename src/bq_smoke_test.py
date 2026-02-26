from google.cloud import bigquery
client = bigquery.Client()
print("Project:", client.project)
datasets = list(client.list_datasets())
print("Datasets:", [d.dataset_id for d in datasets])
