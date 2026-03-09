import psycopg2
import os

url_5432 = "postgresql://postgres:Bhavy%4020084444@db.zxsgcmvlmzgwoaiezfzz.supabase.co:5432/postgres"
url_6543 = "postgresql://postgres:Bhavy%4020084444@db.zxsgcmvlmzgwoaiezfzz.supabase.co:6543/postgres"
url_5432_ssl = url_5432 + "?sslmode=require"
url_6543_ssl = url_6543 + "?sslmode=require"
url_pooler = "postgresql://postgres.zxsgcmvlmzgwoaiezfzz:Bhavy%4020084444@aws-0-ap-south-1.pooler.supabase.com:6543/postgres?sslmode=require"

for name, url in [("5432_ssl", url_5432_ssl), ("6543_ssl", url_6543_ssl), ("pooler", url_pooler)]:
    print(f"Testing {name}...")
    try:
        conn = psycopg2.connect(url, connect_timeout=3)
        print(f"SUCCESS: {name}")
        conn.close()
        break
    except Exception as e:
        print(f"FAILED: {name} - {e}")
