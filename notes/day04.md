	1.	Projection pushdown: partial reading, DuckDB supports projection pushdown into the Parquet file itself. That is to say, when querying a Parquet file, only the columns required for the query are read. This allows you to read only the part of the Parquet file that you are interested in. SELECT doc_id will only scan part of the file rather than import the file by SELECT *.
	2.	Filter pushdown: DuckDB also supports filter pushdown into the Parquet reader. When you apply a filter to a column that is scanned from a Parquet file, the filter will be pushed down into the scan, and can even be used to skip parts of the file using the built-in zonemaps. Note that this will depend on whether or not your Parquet file contains zonemaps.
	3.	Why Parquet for chunk tables: Parquet files use a columnar storage format and contain basic statistics such as zonemaps. Thanks to these features, DuckDB can leverage optimizations such as projection and filter pushdown on Parquet files. Therefore, workloads that combine projection, filtering, and aggregation tend to perform quite well when run on Parquet files.
	4.	note from duckdb_parquet_demo.py: 	
	•	DuckDB lets you run SQL directly on Parquet files via read_parquet()—no “import to database” step required.
	•	.df() converts the result to a pandas DataFrame; great for quick debugging.
	•	Later, you’ll use the same pattern for:
	•	embeddings tables
	•	evaluation results
	•	log tables (local fallback)
output:
================================================================================
Query 1: Select only doc_id (projection pushdown)
- SQL:

    SELECT doc_id
    FROM read_parquet('data/processed/chunks_short.parquet')
    LIMIT 5;
    
- Result preview:
             doc_id
0  squad_v2_train_0
1  squad_v2_train_1
2  squad_v2_train_1
3  squad_v2_train_2
4  squad_v2_train_3

================================================================================
Query 2: Filter by source (filter pushdown)
- SQL:

    SELECT COUNT(*) AS n_chunks
    FROM read_parquet('data/processed/chunks_short.parquet')
    WHERE source = 'squad_v2';
    
- Result preview:
   n_chunks
0     26896

================================================================================
Query 3: Aggregate chunks per doc (GROUP BY)
- SQL:

    SELECT doc_id, COUNT(*) AS n_chunks
    FROM read_parquet('data/processed/chunks_short.parquet')
    GROUP BY doc_id
    ORDER BY n_chunks DESC
    LIMIT 10;
    
- Result preview:
                doc_id  n_chunks
0  squad_v2_train_2875         5
1  squad_v2_train_2882         5
2  squad_v2_train_2881         5
3  squad_v2_train_2879         5
4  squad_v2_train_2876         5
5  squad_v2_train_2877         5
6  squad_v2_train_8385         5
7  squad_v2_train_2878         5
8  squad_v2_train_2880         5
9  squad_v2_train_1721         4

Done.
	5.	DuckDB can query Parquet without loading into pandas.
