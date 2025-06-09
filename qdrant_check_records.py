import sys
import argparse
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, Match

def get_collection_count(client, collection_name):
    try:
        info = client.get_collection(collection_name)
        return info.points_count
    except Exception as e:
        print(f"Error fetching collection info: {e}")
        return None

def search_by_field(client, collection_name, field, value, limit=10):
    try:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key=field,
                    match=Match(value=value)
                )
            ]
        )
        results = client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=limit
        )
        return results[0]
    except Exception as e:
        print(f"Error searching by field: {e}")
        return []

def export_all_ids(client, collection_name, out_path="qdrant_ids.txt"):
    all_ids = set()
    next_page_offset = None
    while True:
        result, next_page_offset = client.scroll(
            collection_name=collection_name,
            limit=10000,
            offset=next_page_offset
        )
        for point in result:
            all_ids.add(str(point.id))
        if next_page_offset is None:
            break
    with open(out_path, "w", encoding="utf-8") as f:
        for id_ in all_ids:
            f.write(id_ + "\n")
    print(f"Exported {len(all_ids)} ids from Qdrant to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Qdrant tender_vectors collection checker")
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--count", action="store_true", help="Show record count")
    parser.add_argument("--field", type=str, help="Field name to search")
    parser.add_argument("--value", type=str, help="Field value to search")
    parser.add_argument("--limit", type=int, default=10, help="Max results to show")
    parser.add_argument("--export-ids", action="store_true", help="Export all IDs to a file")
    parser.add_argument("--out-path", type=str, default="qdrant_ids.txt", help="Output path for exported IDs")
    args = parser.parse_args()

    client = QdrantClient(host=args.host, port=args.port)
    collection_name = "tender_vectors"

    if args.count:
        count = get_collection_count(client, collection_name)
        if count is not None:
            print(f"Total records in '{collection_name}': {count}")
        return

    if args.field and args.value:
        results = search_by_field(client, collection_name, args.field, args.value, args.limit)
        print(f"Found {len(results)} records matching {args.field}={args.value}:")
        for idx, record in enumerate(results, 1):
            print(f"{idx}. ID: {record.id}, Payload: {record.payload}")
        return

    if args.export_ids:
        export_all_ids(client, collection_name, args.out_path)
        return

    parser.print_help()

if __name__ == "__main__":
    main()