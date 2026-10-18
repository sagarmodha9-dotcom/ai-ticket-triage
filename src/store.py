"""Where tickets live.

DynamoTicketStore  reads and updates tickets in DynamoDB (used in AWS).
MemoryTable / MemoryTicketStore  keep tickets in memory (used by tests, the local
server and the Docker container, so everything runs without an AWS account).

update() changes only the fields it is given, appends to lists without reading them first
(so two requests at once can't wipe out each other's notes), and can require a field to still
have the value we last saw (so a stale read can never overwrite a newer change).
"""
import copy


class ConflictError(Exception):
    """The ticket changed (or vanished) since we read it. Read it again and retry."""


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

    def update(self, ticket_id: str, fields: dict | None = None, append: dict | None = None,
               expect: dict | None = None) -> None:
        """fields: values to SET. append: {list_field: [items]} added to the end of a list.
        expect: {field: value} that must still be true, or ConflictError is raised."""
        sets, names, values = [], {}, {}
        for i, (key, value) in enumerate((fields or {}).items()):
            sets.append(f"#f{i} = :f{i}")
            names[f"#f{i}"], values[f":f{i}"] = key, value
        for i, (key, items) in enumerate((append or {}).items()):
            sets.append(f"#a{i} = list_append(if_not_exists(#a{i}, :empty), :a{i})")
            names[f"#a{i}"], values[f":a{i}"], values[":empty"] = key, list(items), []
        conditions = ["attribute_exists(ticket_id)"]
        for i, (key, value) in enumerate((expect or {}).items()):
            conditions.append(f"#e{i} = :e{i}")
            names[f"#e{i}"], values[f":e{i}"] = key, value
        try:
            self.table.update_item(
                Key={"ticket_id": ticket_id},
                UpdateExpression="SET " + ", ".join(sets),
                ExpressionAttributeNames=names,
                ExpressionAttributeValues=values,
                ConditionExpression=" AND ".join(conditions),
            )
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if code == "ConditionalCheckFailedException":
                raise ConflictError("ticket changed since it was read") from None
            raise


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

    def update(self, ticket_id: str, fields: dict | None = None, append: dict | None = None,
               expect: dict | None = None) -> None:
        item = self.table.items.get(ticket_id)
        if item is None or any(item.get(k) != v for k, v in (expect or {}).items()):
            raise ConflictError("ticket changed since it was read")
        item.update(copy.deepcopy(fields or {}))
        for key, items in (append or {}).items():
            item.setdefault(key, []).extend(copy.deepcopy(list(items)))
