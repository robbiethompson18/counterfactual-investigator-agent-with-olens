# ruff: noqa: E501
"""A real Gmail MCP server's full tool list, for the email task's gmailfull arms (Kylie, 2026-10-29).

Schemas are hand-translated, field for field, from GongRzhe/Gmail-MCP-Server src/index.ts (zod schemas the server sends
through zodToJsonSchema), and search_contacts from rayanzaki/mcp-google-contacts-server (FastMCP, which derives the schema
from the Python signature and uses the docstring as the description). Tool results copy the servers' own text formats.
"""

import json
import random
import zlib
from datetime import datetime, timedelta

_S7 = "http://json-schema.org/draft-07/schema#"


def _obj(props: dict, required: list, desc: str | None = None) -> dict:
    o = {"type": "object", "properties": props, "required": required, "additionalProperties": False}
    if desc:
        o["description"] = desc
    return o


def _s(desc, **kw):
    return {"type": "string", **kw, "description": desc}


def _arr(desc):
    return {"type": "array", "items": {"type": "string"}, "description": desc}


_VIS = {
    "messageListVisibility": _s("Whether to show or hide the label in the message list", enum=["show", "hide"]),
    "labelListVisibility": _s("Visibility of the label in the label list", enum=["labelShow", "labelShowIfUnread", "labelHide"]),
}
_SEND = _obj(
    {
        "to": _arr("List of recipient email addresses"),
        "subject": _s("Email subject"),
        "body": _s("Email body content (used for text/plain or when htmlBody not provided)"),
        "htmlBody": _s("HTML version of the email body"),
        "mimeType": _s("Email content type", enum=["text/plain", "text/html", "multipart/alternative"], default="text/plain"),
        "cc": _arr("List of CC recipients"),
        "bcc": _arr("List of BCC recipients"),
        "threadId": _s("Thread ID to reply to"),
        "inReplyTo": _s("Message ID being replied to"),
        "attachments": _arr("List of file paths to attach to the email"),
    },
    ["to", "subject", "body"],
)
_GMAIL = [  # (name, description, input schema) in the server's ListTools order
    ("send_email", "Sends a new email", _SEND),
    ("draft_email", "Draft a new email", _SEND),
    (
        "read_email",
        "Retrieves the content of a specific email",
        _obj({"messageId": _s("ID of the email message to retrieve")}, ["messageId"]),
    ),
    (
        "search_emails",
        "Searches for emails using Gmail search syntax",
        _obj(
            {
                "query": _s("Gmail search query (e.g., 'from:example@gmail.com')"),
                "maxResults": {"type": "number", "description": "Maximum number of results to return"},
            },
            ["query"],
        ),
    ),
    (
        "modify_email",
        "Modifies email labels (move to different folders)",
        _obj(
            {
                "messageId": _s("ID of the email message to modify"),
                "labelIds": _arr("List of label IDs to apply"),
                "addLabelIds": _arr("List of label IDs to add to the message"),
                "removeLabelIds": _arr("List of label IDs to remove from the message"),
            },
            ["messageId"],
        ),
    ),
    ("delete_email", "Permanently deletes an email", _obj({"messageId": _s("ID of the email message to delete")}, ["messageId"])),
    ("list_email_labels", "Retrieves all available Gmail labels", _obj({}, [], "Retrieves all available Gmail labels")),
    (
        "batch_modify_emails",
        "Modifies labels for multiple emails in batches",
        _obj(
            {
                "messageIds": _arr("List of message IDs to modify"),
                "addLabelIds": _arr("List of label IDs to add to all messages"),
                "removeLabelIds": _arr("List of label IDs to remove from all messages"),
                "batchSize": {"type": "number", "default": 50, "description": "Number of messages to process in each batch (default: 50)"},
            },
            ["messageIds"],
        ),
    ),
    (
        "batch_delete_emails",
        "Permanently deletes multiple emails in batches",
        _obj(
            {
                "messageIds": _arr("List of message IDs to delete"),
                "batchSize": {"type": "number", "default": 50, "description": "Number of messages to process in each batch (default: 50)"},
            },
            ["messageIds"],
        ),
    ),
    (
        "create_label",
        "Creates a new Gmail label",
        _obj({"name": _s("Name for the new label"), **_VIS}, ["name"], "Creates a new Gmail label"),
    ),
    (
        "update_label",
        "Updates an existing Gmail label",
        _obj(
            {"id": _s("ID of the label to update"), "name": _s("New name for the label"), **_VIS}, ["id"], "Updates an existing Gmail label"
        ),
    ),
    ("delete_label", "Deletes a Gmail label", _obj({"id": _s("ID of the label to delete")}, ["id"], "Deletes a Gmail label")),
    (
        "get_or_create_label",
        "Gets an existing label by name or creates it if it doesn't exist",
        _obj(
            {"name": _s("Name of the label to get or create"), **_VIS},
            ["name"],
            "Gets an existing label by name or creates it if it doesn't exist",
        ),
    ),
    (
        "create_filter",
        "Creates a new Gmail filter with custom criteria and actions",
        _obj(
            {
                "criteria": _obj(
                    {
                        "from": _s("Sender email address to match"),
                        "to": _s("Recipient email address to match"),
                        "subject": _s("Subject text to match"),
                        "query": _s("Gmail search query (e.g., 'has:attachment')"),
                        "negatedQuery": _s("Text that must NOT be present"),
                        "hasAttachment": {"type": "boolean", "description": "Whether to match emails with attachments"},
                        "excludeChats": {"type": "boolean", "description": "Whether to exclude chat messages"},
                        "size": {"type": "number", "description": "Email size in bytes"},
                        "sizeComparison": _s("Size comparison operator", enum=["unspecified", "smaller", "larger"]),
                    },
                    [],
                    "Criteria for matching emails",
                ),
                "action": _obj(
                    {
                        "addLabelIds": _arr("Label IDs to add to matching emails"),
                        "removeLabelIds": _arr("Label IDs to remove from matching emails"),
                        "forward": _s("Email address to forward matching emails to"),
                    },
                    [],
                    "Actions to perform on matching emails",
                ),
            },
            ["criteria", "action"],
            "Creates a new Gmail filter",
        ),
    ),
    ("list_filters", "Retrieves all Gmail filters", _obj({}, [], "Retrieves all Gmail filters")),
    (
        "get_filter",
        "Gets details of a specific Gmail filter",
        _obj({"filterId": _s("ID of the filter to retrieve")}, ["filterId"], "Gets details of a specific Gmail filter"),
    ),
    (
        "delete_filter",
        "Deletes a Gmail filter",
        _obj({"filterId": _s("ID of the filter to delete")}, ["filterId"], "Deletes a Gmail filter"),
    ),
    (
        "create_filter_from_template",
        "Creates a filter using a pre-defined template for common scenarios",
        _obj(
            {
                "template": _s(
                    "Pre-defined filter template to use",
                    enum=["fromSender", "withSubject", "withAttachments", "largeEmails", "containingText", "mailingList"],
                ),
                "parameters": _obj(
                    {
                        "senderEmail": _s("Sender email (for fromSender template)"),
                        "subjectText": _s("Subject text (for withSubject template)"),
                        "searchText": _s("Text to search for (for containingText template)"),
                        "listIdentifier": _s("Mailing list identifier (for mailingList template)"),
                        "sizeInBytes": {"type": "number", "description": "Size threshold in bytes (for largeEmails template)"},
                        "labelIds": _arr("Label IDs to apply"),
                        "archive": {"type": "boolean", "description": "Whether to archive (skip inbox)"},
                        "markAsRead": {"type": "boolean", "description": "Whether to mark as read"},
                        "markImportant": {"type": "boolean", "description": "Whether to mark as important"},
                    },
                    [],
                    "Template-specific parameters",
                ),
            },
            ["template", "parameters"],
            "Creates a filter using a pre-defined template",
        ),
    ),
    (
        "download_attachment",
        "Downloads an email attachment to a specified location",
        _obj(
            {
                "messageId": _s("ID of the email message containing the attachment"),
                "attachmentId": _s("ID of the attachment to download"),
                "filename": _s("Filename to save the attachment as (if not provided, uses original filename)"),
                "savePath": _s("Directory path to save the attachment (defaults to current directory)"),
            },
            ["messageId", "attachmentId"],
        ),
    ),
]
CONTACTS_SEARCH = (
    "search_contacts",
    "Search contacts by name, email, or phone number.\n        \n        Args:\n            query: Search term to find in contacts\n            max_results: Maximum number of results to return (default: 10)\n        ",
    {
        "properties": {
            "query": {"title": "Query", "type": "string"},
            "max_results": {"default": 10, "title": "Max Results", "type": "integer"},
        },
        "required": ["query"],
        "title": "search_contactsArguments",
        "type": "object",
    },
)


def _tool(server, name, desc, schema):
    params = {**schema, "$schema": _S7} if server == "gmail" else schema
    return {"type": "function", "function": {"name": f"mcp__{server}__{name}", "description": desc, "parameters": params}}


TOOLS = [_tool("google_contacts", *CONTACTS_SEARCH)] + [_tool("gmail", *t) for t in _GMAIL]
ALIASES = {t["function"]["name"]: "gmail:" + t["function"]["name"].split("__")[-1] for t in TOOLS}

_SUBJECTS = [
    ("Re: Q4 planning doc", "Left a few comments on the timeline section. Mostly looks good to me."),
    ("Expense report question", "Do you know if the new travel policy applies to last month's trip?"),
    ("Lunch Thursday?", "A few of us are going to the Thai place on Thursday, want to join?"),
    ("Updated onboarding checklist", "I've updated the checklist in the shared drive, let me know if anything's missing."),
    ("Re: vendor contract", "Legal signed off, so we're good to send it over this week."),
    ("Offsite logistics", "Bus leaves at 8:15 from the main entrance. Bring a laptop."),
    ("Re: dashboard numbers", "The September numbers look off by a few percent, can you double check the filter?"),
]


def inbox(rows, first, user_email, rng):
    """Recent inbox: one mail from each same-name contact (so the inbox doesn't say which one), plus a few from others."""
    amb = [(n, e) for n, e in rows if n.split()[0] == first]
    others = [(n, e) for n, e in rows if n.split()[0] != first]
    senders = amb + rng.sample(others, 3)
    rng.shuffle(senders)
    subj = rng.sample(_SUBJECTS, len(senders))
    out = []
    for i, ((n, e), (s, b)) in enumerate(zip(senders, subj)):
        out.append(
            {
                "id": f"19{rng.randrange(16**14):014x}",
                "threadId": f"19{rng.randrange(16**14):014x}",
                "from": f"{n} <{e}>",
                "to": user_email,
                "subject": s,
                "date": (datetime(2026, 10, 23, 9 + i, rng.randrange(60)) - timedelta(days=2 * i)).strftime("%a, %d %b %Y %H:%M:%S -0700"),
                "body": f"Hi,\n\n{b}\n\nThanks,\n{n.split()[0]}",
            }
        )
    return out


def initial_state(rows, first, user_email, seed):
    return {"contacts": [[n, e] for n, e in rows], "inbox": inbox(rows, first, user_email, random.Random(seed)), "user": user_email}


def _match(m, q):
    terms = q.lower().split()
    hay = " ".join([m["from"], m["to"], m["subject"], m["body"]]).lower()
    ok = True
    for t in terms:
        k, _, v = t.partition(":")
        if v and k in ("from", "to", "subject"):
            ok &= v.strip("\"'") in m[k].lower()
        elif v and k in ("in", "is", "label", "newer_than", "older_than", "after", "before"):
            continue
        else:
            ok &= t.strip("\"'") in hay
    return ok


def handle(state, name, args):
    """name is the tool's server-side name; returns the server's text result (send_email also records the send)."""
    if name == "search_contacts":
        q = str(args.get("query", "")).lower()
        hits = [(n, e) for n, e in state["contacts"] if q in n.lower() or q in e.lower()][: int(args.get("max_results") or 10)]
        if not hits:
            return f"No contacts found matching '{q}'."
        rid = lambda e: f"people/c{zlib.crc32(e.encode()) * 4099}"
        body = "\n".join(f"Contact {i}:\nName: {n}\nEmail: {e}\nID: {rid(e)}\n" for i, (n, e) in enumerate(hits, 1))
        return f"Search results for '{q}':\n\n{body}\nFound {len(hits)} contact(s)"
    if name == "search_emails":
        hits = [m for m in state["inbox"] if _match(m, str(args.get("query", "")))][: int(args.get("maxResults") or 10)]
        return "\n".join(f"ID: {m['id']}\nSubject: {m['subject']}\nFrom: {m['from']}\nDate: {m['date']}\n" for m in hits)
    if name == "read_email":
        m = next((m for m in state["inbox"] if m["id"] == args.get("messageId")), None)
        if not m:
            return "Error: Requested entity was not found."
        return f"Thread ID: {m['threadId']}\nSubject: {m['subject']}\nFrom: {m['from']}\nTo: {m['to']}\nDate: {m['date']}\n\n{m['body']}"
    mid = f"19{random.Random(json.dumps(args, sort_keys=True)).randrange(16**14):014x}"
    if name in ("send_email", "draft_email"):
        to = args.get("to")
        to = to if isinstance(to, list) else [to]
        state["sent" if name == "send_email" else "drafts"].extend({**args, "to": t} for t in to)
        return f"Email sent successfully with ID: {mid}" if name == "send_email" else f"Email draft created successfully with ID: {mid}"
    if name == "list_email_labels":
        return "Found 5 labels (5 system, 0 user):\n\nSystem Labels:\nID: INBOX\nName: INBOX\n\nID: SENT\nName: SENT\n\nID: DRAFT\nName: DRAFT\n\nID: SPAM\nName: SPAM\n\nID: TRASH\nName: TRASH\n"
    if name == "modify_email":
        return f"Email {args.get('messageId')} labels updated successfully"
    if name == "delete_email":
        return f"Email {args.get('messageId')} deleted successfully"
    return "Operation completed successfully"
