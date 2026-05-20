"""AI database wrapper for MCP-connected databases.

This module provides a generic adapter for database connections that are
managed through an MCP-style connection object, plus an LLM-assisted wrapper
that converts plain-language requests into safe SQL execution.
"""

from __future__ import annotations
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class DatabaseConnector(ABC):
    """Abstract interface for a database connector."""

    @abstractmethod
    def execute(self, sql: str, parameters: Optional[Sequence[Any]] = None) -> Any:
        """Execute a SQL statement and return the raw database result."""

    @abstractmethod
    def fetch_all(self, sql: str, parameters: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
        """Execute a SQL query and return all rows as dictionaries."""

    @abstractmethod
    def introspect_schema(self) -> Dict[str, List[str]]:
        """Return a schema overview for tables and column names."""

    @abstractmethod
    def close(self) -> None:
        """Close the underlying database connection."""


class MCPDatabaseConnector(DatabaseConnector):
    """Adapter for an MCP-managed database connection object."""

    def __init__(self, mcp_connection: Any, connection_id: str):
        self._mcp_connection = mcp_connection
        self.connection_id = connection_id

    def execute(self, sql: str, parameters: Optional[Sequence[Any]] = None) -> Any:
        payload = {
            "connection_id": self.connection_id,
            "sql": sql,
            "parameters": list(parameters or []),
        }

        if hasattr(self._mcp_connection, "execute"):
            return self._mcp_connection.execute(payload)

        if hasattr(self._mcp_connection, "run"):
            return self._mcp_connection.run(payload)

        raise NotImplementedError(
            "MCP connection object must implement execute(payload) or run(payload)."
        )

    def fetch_all(self, sql: str, parameters: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
        result = self.execute(sql, parameters)
        if isinstance(result, dict) and "rows" in result:
            return result["rows"]
        if isinstance(result, list):
            return result
        return []

    def introspect_schema(self) -> Dict[str, List[str]]:
        if hasattr(self._mcp_connection, "introspect_schema"):
            return self._mcp_connection.introspect_schema(self.connection_id)

        # Fallback generic introspection for SQL-compatible MCP providers.
        schema: Dict[str, List[str]] = {}
        try:
            tables = self.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            if isinstance(tables, dict) and "rows" in tables:
                table_names = [row[0] for row in tables["rows"]]
            elif isinstance(tables, list):
                table_names = [row[0] for row in tables]
            else:
                table_names = []

            for table in table_names:
                columns = self.execute(f"PRAGMA table_info({table});")
                if isinstance(columns, dict) and "rows" in columns:
                    schema[table] = [row[1] for row in columns["rows"]]
                elif isinstance(columns, list):
                    schema[table] = [row[1] for row in columns]
        except Exception:
            pass

        return schema

    def close(self) -> None:
        if hasattr(self._mcp_connection, "close"):
            self._mcp_connection.close()


class AIDatabaseWrapper:
    """AI-driven wrapper for database queries and commands."""

    ALLOWED_STATEMENTS = {
        "SELECT",
        "INSERT",
        "UPDATE",
        "DELETE",
        "CREATE",
        "ALTER",
        "DROP",
    }

    def __init__(self, connector: DatabaseConnector, llm: Optional[ChatOpenAI] = None):
        self.connector = connector
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
        self._schema: Optional[Dict[str, List[str]]] = None

    def _load_schema(self) -> Dict[str, List[str]]:
        if self._schema is None:
            self._schema = self.connector.introspect_schema()
        return self._schema

    def _format_schema(self, schema: Dict[str, List[str]]) -> str:
        if not schema:
            return "No schema information is available."
        return json.dumps(schema, indent=2)

    def _generate_sql(self, natural_language: str) -> Tuple[str, List[Any], str]:
        schema = self._load_schema()
        system_prompt = f"""You are an AI assistant that converts natural-language database requests into SQL statements.
Only return valid JSON with the fields: sql, parameters, explanation.
The database schema is:
{self._format_schema(schema)}
Allowed statements: {sorted(self.ALLOWED_STATEMENTS)}.
If the user request is read-only, produce a SELECT statement.
Do not wrap your response in markdown code fences.
"""

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User request: {natural_language}")
        ])

        try:
            parsed = json.loads(response.content)
            sql = parsed.get("sql", "").strip()
            parameters = parsed.get("parameters", [])
            explanation = parsed.get("explanation", "Generated SQL from user request.")
        except json.JSONDecodeError:
            raise ValueError(
                "Unable to parse SQL generation response from the LLM."
            )

        self._ensure_safe_statement(sql)

        return sql, list(parameters), explanation

    def _ensure_safe_statement(self, sql: str) -> None:
        normalized = sql.strip().lstrip("(\n").upper()
        if not normalized:
            raise ValueError("Empty SQL statement is not allowed.")

        first_word = normalized.split(None, 1)[0]
        if first_word not in self.ALLOWED_STATEMENTS:
            raise ValueError(
                f"SQL statement '{first_word}' is not permitted by the AI wrapper."
            )

    def execute_natural_language(self, natural_language: str) -> Dict[str, Any]:
        sql, params, explanation = self._generate_sql(natural_language)
        raw = self.connector.execute(sql, params)
        return {
            "query": natural_language,
            "sql": sql,
            "parameters": params,
            "explanation": explanation,
            "result": raw,
        }

    def query_natural_language(self, natural_language: str) -> Dict[str, Any]:
        sql, params, explanation = self._generate_sql(natural_language)
        rows = self.connector.fetch_all(sql, params)
        return {
            "query": natural_language,
            "sql": sql,
            "parameters": params,
            "explanation": explanation,
            "rows": rows,
        }

    def schema(self) -> Dict[str, List[str]]:
        return self._load_schema()


# Example usage with an MCP-style client.
if __name__ == "__main__":
    class DummyMCPClient:
        """Example stub that simulates an MCP database connector."""

        def execute(self, payload: Dict[str, Any]) -> Any:
            sql = payload["sql"].strip().upper()
            if sql.startswith("SELECT"):
                return [{"id": 1, "name": "Sample"}]
            return {"status": "ok", "sql": payload["sql"]}

        def close(self) -> None:
            pass

    mcp_client = DummyMCPClient()
    connector = MCPDatabaseConnector(mcp_client, connection_id="demo")
    ai_db = AIDatabaseWrapper(connector)

    print("Schema:", ai_db.schema())
    print(ai_db.query_natural_language("List all customers with their id and name."))
