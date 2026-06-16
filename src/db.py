
import psycopg2, os
import boto3
from contextlib import contextmanager
from dotenv import load_dotenv


load_dotenv()

# PostgreSQL — persistent storage
PG_DSN = os.getenv('DATABASE_URL')  

@contextmanager
def get_db():
    conn = psycopg2.connect(PG_DSN)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# AWS DynamoDB — async job tracking (used from Phase 9 onward)
# Lazy-initialised so importing db.py doesn't require AWS keys during Phase 1
_dynamo = None
_jobs_table = None

def _get_jobs_table():
    global _dynamo, _jobs_table
    if _jobs_table is None:
        _dynamo = boto3.resource(
            'dynamodb',
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        _jobs_table = _dynamo.Table('constructiq_jobs')
    return _jobs_table

def update_job_status(job_id: str, status: str, progress: str = ''):
    _get_jobs_table().update_item(
        Key={'job_id': job_id},
        UpdateExpression='SET #s = :s, progress = :p',
        ExpressionAttributeNames={'#s': 'status'},
        ExpressionAttributeValues={':s': status, ':p': progress}
    )


def test_connection():
    """Quick sanity check — run this at the end of Phase 1 setup."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
            tables = [r[0] for r in cur.fetchall()]
    print("Connected. Tables found:", tables)
    return tables


if __name__ == "__main__":
    test_connection()
