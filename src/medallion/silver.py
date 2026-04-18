import re
import io
import pypdf
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, explode, current_timestamp
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType

class SilverLayer:
    def __init__(self, catalog: str, schema: str, bronze_table: str, silver_table: str, chunk_size: int = 1000):
        self.spark = SparkSession.getActiveInstance()
        self.bronze_fqn = f"{catalog}.{schema}.{bronze_table}"
        self.silver_fqn = f"{catalog}.{schema}.{silver_table}"
        self.chunk_size = chunk_size

    @staticmethod
    def parse_pdf_hierarchical(content: bytes, filename: str):
        """
        Hierarchical parser for legal PDFs. 
        Maintains heading stack to provide full context for each chunk.
        """
        pdf_file = io.BytesIO(content)
        reader = pypdf.PdfReader(pdf_file)
        
        chunks = []
        heading_stack = [] # Stack to keep track of Part > Chapter > Section
        
        # Regex for common legal structures in India
        patterns = {
            'PART': re.compile(r'^PART\s+[IVXLCDM]+', re.IGNORECASE),
            'CHAPTER': re.compile(r'^CHAPTER\s+[IVXLCDM\d]+', re.IGNORECASE),
            'SECTION': re.compile(r'^(\d+)\.\s+'),
            'ARTICLE': re.compile(r'^Article\s+(\d+)', re.IGNORECASE)
        }

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            lines = text.split('\n')
            
            page_buffer = ""
            for line in lines:
                line = line.strip()
                if not line: continue
                
                # Check for headings to update stack
                matched_heading = False
                for level, pattern in patterns.items():
                    if pattern.match(line):
                        # Update stack: find if level already exists and pop above it
                        # Simple implementation: just append for now to show context
                        heading_stack = [h for h in heading_stack if h['level'] != level]
                        heading_stack.append({'level': level, 'text': line})
                        matched_heading = True
                        break
                
                page_buffer += line + " "
                
                if len(page_buffer) >= 1000: # Threshold for chunking
                    context_str = " > ".join([h['text'] for h in heading_stack])
                    chunks.append({
                        'source_file': filename,
                        'page': page_num + 1,
                        'heading_chain': context_str if context_str else "N/A",
                        'raw_text': page_buffer.strip(),
                        'section_id': heading_stack[-1]['text'] if heading_stack else "root",
                        'xrefs': [] # Logic for extracting cross-references would go here
                    })
                    page_buffer = ""
            
            # Remaining text on page
            if page_buffer:
                context_str = " > ".join([h['text'] for h in heading_stack])
                chunks.append({
                    'source_file': filename,
                    'page': page_num + 1,
                    'heading_chain': context_str if context_str else "N/A",
                    'raw_text': page_buffer.strip(),
                    'section_id': heading_stack[-1]['text'] if heading_stack else "root",
                    'xrefs': []
                })

        return chunks

    def process_silver(self):
        """
        Read from Bronze, parse, and write to Silver Delta.
        """
        print(f"[*] Processing Bronze {self.bronze_fqn} -> Silver {self.silver_fqn}...")
        
        # Define UDF for hierarchical parsing
        schema = ArrayType(StructType([
            StructField("source_file", StringType(), True),
            StructField("page", IntegerType(), True),
            StructField("heading_chain", StringType(), True),
            StructField("raw_text", StringType(), True),
            StructField("section_id", StringType(), True),
            StructField("xrefs", ArrayType(StringType()), True)
        ]))

        parse_udf = udf(lambda content, filename: SilverLayer.parse_pdf_hierarchical(content, filename), schema)

        # Read Bronze
        bronze_df = self.spark.table(self.bronze_fqn)

        # Transform
        silver_df = (bronze_df.withColumn("parsed", parse_udf(col("content"), col("path")))
                     .withColumn("chunk", explode(col("parsed")))
                     .select(
                         col("chunk.source_file"),
                         col("chunk.page"),
                         col("chunk.heading_chain"),
                         col("chunk.raw_text"),
                         col("chunk.section_id"),
                         col("chunk.xrefs"),
                         current_timestamp().alias("processed_at")
                     ))

        # Write to Silver
        (silver_df.write.format("delta")
         .mode("overwrite")
         .option("overwriteSchema", "true")
         .saveAsTable(self.silver_fqn))

        # Enable CDF
        self.spark.sql(f"ALTER TABLE {self.full_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
        
        print(f"[+] Silver processing complete. Table: {self.silver_fqn}")

if __name__ == "__main__":
    import yaml
    with open("configs/settings.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    silver = SilverLayer(
        catalog=cfg['tables']['catalog'],
        schema=cfg['tables']['schema'],
        bronze_table=cfg['tables']['bronze_table'],
        silver_table=cfg['tables']['silver_table']
    )
    silver.process_silver()
