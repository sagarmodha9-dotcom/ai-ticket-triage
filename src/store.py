"""Where tickets live.

DynamoTicketStore  reads and updates tickets in DynamoDB (used in AWS).
MemoryTable / MemoryTicketStore  keep tickets in memory (used by tests, the local
server and the Docker container, so everything runs without an AWS account).
"""
import copy


class DynamoTicketStore:
    def __init__(self, table):
        self.table = table

    def get(self, ticket_id: str) -> dict | None:
        return self.table.get_item(Key={"ticket_id": ticket_id}).get("Item")

    def recent(self, since_ts: int) -> list[dict]:
        """Tickets created since a time. A Scan is fine for a small table; at scale,
        add a GSI on created_at and Query it instead."""
        from boto3.dynamodb.conditions import Attr
        items, kwargs = [], {"FilterExpression": Attr("created_at").gte(since_ts)}
        while True:
            page = self.table.scan(**kwargs)
            items.extend(page.get("Items", []))
            if "LastEvaluatedKey" not in page:
                return items
            kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]

    def update(self, ticket_id: str, fields: dict, note: dict | None = None) -> None:
        sets, names, values = [], {}, {}
        for i, (key, value) in enumerate(fields.items()):
            sets.append(f"#f{i} = :v{i}")
            names[f"#f{i}"] = key
            values[f":v{i}"] = value
        if note is not None:
            sets.append("#notes = list_append(if_not_exists(#notes, :empty), :note)")
            names["#notes"] = "agent_notes"
            values[":empty"] = []
            values[":note"] = [note]
        self.table.update_item(
            Key={"ticket_id": ticket_id},
            UpdateExpression="SET " + ", ".join(sets),
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
            ConditionExpression="attribute_exists(ticket_id)",
        )


class MemoryTable:
    """Just enough of the DynamoDB Table API (put_item / get_item) for app.py."""

    def __init__(self):
        self.items = {}

    def put_item(self, Item):
        self.items[Item["ticket_id"]] = copy.deepcopy(Item)

    def get_item(self, Key):
        item = self.items.get(Key["ticket_id"])
        return {"Item": copy.deepcopy(item)} if item else {}


class MemoryTicketStore:
    def __init__(self, table: MemoryTable):
        self.table = table

    def get(self, ticket_id: str) -> dict | None:
        item = self.table.items.get(ticket_id)
        return copy.deepcopy(item) if item else None

    def recent(self, since_ts: int) -> list[dict]:
        return [copy.deepcopy(t) for t in self.table.items.values() if t.get("created_at", 0) >= since_ts]

    def update(self, ticket_id: str, fields: dict, note: dict | None = None) -> None:
        item = self.table.items[ticket_id]
        item.update(copy.deepcopy(fields))
        if note is not None:
            item.setdefault("agent_notes", []).append(note)
