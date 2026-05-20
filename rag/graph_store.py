from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase

from rag.config import (
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
)


class TaxLawGraphStore:
    def __init__(
        self,
        uri: str = NEO4J_URI,
        username: str = NEO4J_USERNAME,
        password: str = NEO4J_PASSWORD,
        database: str = NEO4J_DATABASE,
    ) -> None:
        if not uri or not username or not password:
            raise RuntimeError(
                "Neo4j credentials are missing. Set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD."
            )
        self._database = database
        self._driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "TaxLawGraphStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

# Graph scehma


    def ensure_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT page_id IF NOT EXISTS FOR (p:Page) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT keyword_name IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE",
            "CREATE CONSTRAINT section_name IF NOT EXISTS FOR (s:Section) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT statute_reference_name IF NOT EXISTS FOR (r:StatuteReference) REQUIRE r.name IS UNIQUE",
            "CREATE FULLTEXT INDEX chunk_text_index IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
            "CREATE FULLTEXT INDEX section_title_index IF NOT EXISTS FOR (s:Section) ON EACH [s.name]",
            "CREATE FULLTEXT INDEX statute_reference_index IF NOT EXISTS FOR (r:StatuteReference) ON EACH [r.name]",
        ]
        with self._driver.session(database=self._database) as session:
            for statement in statements:
                session.run(statement).consume()

    def clear_graph(self) -> None:
        cypher = """
        MATCH (n)
        WHERE any(label IN labels(n) WHERE label IN [
            "Document",
            "Page",
            "Chunk",
            "Keyword",
            "Section",
            "StatuteReference"
        ])
        DETACH DELETE n
        """
        with self._driver.session(database=self._database) as session:
            session.run(cypher).consume()

    def ingest_documents(self, documents: list[dict[str, Any]]) -> None:

        document_query = """
        MERGE (d:Document {id: $document_id})
        SET d.name = $name,
            d.path = $path,
            d.document_type = $document_type,
            d.document_year = $document_year
        """
        page_query = """
        MATCH (d:Document {id: $document_id})
        MERGE (p:Page {id: $page_id})
        SET p.number = $page_number,
            p.preview = $preview
        MERGE (d)-[:HAS_PAGE]->(p)
        """
        chunk_query = """
        MATCH (p:Page {id: $page_id})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text,
            c.source_file = $source_file,
            c.source_path = $source_path,
            c.document_id = $document_id,
            c.document_type = $document_type,
            c.document_year = $document_year,
            c.page_number = $page_number,
            c.total_pages = $total_pages,
            c.chunk_index = $chunk_index,
            c.section_title = $section_title,
            c.statute_reference = $statute_reference,
            c.keywords = $keywords
        MERGE (p)-[:HAS_CHUNK]->(c)
        WITH c
        OPTIONAL MATCH (c)-[old_rel:HAS_KEYWORD|IN_SECTION|REFERS_TO]->()
        DELETE old_rel
        WITH c
        OPTIONAL MATCH (c)-[old_next:NEXT]->()
        DELETE old_next
        WITH c
        UNWIND $keywords AS keyword_name
        MERGE (k:Keyword {name: keyword_name})
        MERGE (c)-[:HAS_KEYWORD]->(k)
        """
        section_query = """
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (s:Section {name: $section_title})
        MERGE (c)-[:IN_SECTION]->(s)
        """
        reference_query = """
        MATCH (c:Chunk {id: $chunk_id})
        UNWIND $reference_names AS reference_name
        MERGE (r:StatuteReference {name: reference_name})
        MERGE (c)-[:REFERS_TO]->(r)
        """
        next_query = """
        MATCH (left:Chunk {id: $left_chunk_id})
        MATCH (right:Chunk {id: $right_chunk_id})
        MERGE (left)-[:NEXT]->(right)
        """

        with self._driver.session(database=self._database) as session:
            for document in documents:
                session.run(
                    document_query,
                    document_id=document["document_id"],
                    name=document["name"],
                    path=document["path"],
                    document_type=document["document_type"],
                    document_year=document["document_year"],
                ).consume()

                for page in document["pages"]:
                    session.run(
                        page_query,
                        document_id=document["document_id"],
                        page_id=page["page_id"],
                        page_number=page["page_number"],
                        preview=page["preview"],
                    ).consume()

                    previous_chunk_id: str | None = None
                    for chunk in page["chunks"]:
                        metadata = chunk["metadata"]
                        session.run(
                            chunk_query,
                            page_id=page["page_id"],
                            chunk_id=chunk["chunk_id"],
                            text=chunk["text"],
                            source_file=metadata.get("source_file", "Unknown"),
                            source_path=metadata.get("source_path", "Unknown"),
                            document_id=metadata.get("document_id", document["document_id"]),
                            document_type=metadata.get("document_type", document["document_type"]),
                            document_year=metadata.get("document_year", document["document_year"]),
                            page_number=metadata.get("page_number", "Unknown"),
                            total_pages=metadata.get("total_pages", "Unknown"),
                            chunk_index=metadata.get("chunk_index", 0),
                            section_title=metadata.get("section_title", "Unknown"),
                            statute_reference=metadata.get("statute_reference", "Unknown"),
                            keywords=chunk["keywords"],
                        ).consume()

                        section_title = metadata.get("section_title", "Unknown")
                        if section_title and section_title != "Unknown":
                            session.run(
                                section_query,
                                chunk_id=chunk["chunk_id"],
                                section_title=section_title,
                            ).consume()

                        reference_names = [
                            name.strip()
                            for name in str(metadata.get("statute_reference", "")).split("|")
                            if name.strip() and name.strip() != "Unknown"
                        ]
                        if reference_names:
                            session.run(
                                reference_query,
                                chunk_id=chunk["chunk_id"],
                                reference_names=reference_names,
                            ).consume()

                        if previous_chunk_id is not None:
                            session.run(
                                next_query,
                                left_chunk_id=previous_chunk_id,
                                right_chunk_id=chunk["chunk_id"],
                            ).consume()
                        previous_chunk_id = chunk["chunk_id"]

    def search_chunk_text(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        cypher = """
        CALL db.index.fulltext.queryNodes("chunk_text_index", $query_text) YIELD node, score
        RETURN node.id AS chunk_id,
               node.text AS text,
               node.source_file AS source_file,
               node.source_path AS source_path,
               node.document_id AS document_id,
               node.document_type AS document_type,
               node.document_year AS document_year,
               node.page_number AS page_number,
               node.total_pages AS total_pages,
               node.chunk_index AS chunk_index,
               node.section_title AS section_title,
               node.statute_reference AS statute_reference,
               node.keywords AS keywords,
               score AS retrieval_score
        ORDER BY score DESC, node.chunk_index ASC
        LIMIT $limit
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, query_text=query_text, limit=limit)
            return [record.data() for record in result]

    def search_sections(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        cypher = """
        CALL db.index.fulltext.queryNodes("section_title_index", $query_text) YIELD node, score
        MATCH (c:Chunk)-[:IN_SECTION]->(node)
        RETURN c.id AS chunk_id,
               c.text AS text,
               c.source_file AS source_file,
               c.source_path AS source_path,
               c.document_id AS document_id,
               c.document_type AS document_type,
               c.document_year AS document_year,
               c.page_number AS page_number,
               c.total_pages AS total_pages,
               c.chunk_index AS chunk_index,
               c.section_title AS section_title,
               c.statute_reference AS statute_reference,
               c.keywords AS keywords,
               score AS retrieval_score
        ORDER BY score DESC, c.chunk_index ASC
        LIMIT $limit
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, query_text=query_text, limit=limit)
            return [record.data() for record in result]

    def search_references(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        cypher = """
        CALL db.index.fulltext.queryNodes("statute_reference_index", $query_text) YIELD node, score
        MATCH (c:Chunk)-[:REFERS_TO]->(node)
        RETURN c.id AS chunk_id,
               c.text AS text,
               c.source_file AS source_file,
               c.source_path AS source_path,
               c.document_id AS document_id,
               c.document_type AS document_type,
               c.document_year AS document_year,
               c.page_number AS page_number,
               c.total_pages AS total_pages,
               c.chunk_index AS chunk_index,
               c.section_title AS section_title,
               c.statute_reference AS statute_reference,
               c.keywords AS keywords,
               score AS retrieval_score
        ORDER BY score DESC, c.chunk_index ASC
        LIMIT $limit
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, query_text=query_text, limit=limit)
            return [record.data() for record in result]

    def fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []

        cypher = """
        UNWIND $chunk_ids AS chunk_id
        MATCH (c:Chunk {id: chunk_id})
        RETURN c.id AS chunk_id,
               c.text AS text,
               c.source_file AS source_file,
               c.source_path AS source_path,
               c.document_id AS document_id,
               c.document_type AS document_type,
               c.document_year AS document_year,
               c.page_number AS page_number,
               c.total_pages AS total_pages,
               c.chunk_index AS chunk_index,
               c.section_title AS section_title,
               c.statute_reference AS statute_reference,
               c.keywords AS keywords,
               0.0 AS retrieval_score
        ORDER BY c.page_number ASC, c.chunk_index ASC
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, chunk_ids=chunk_ids)
            return [record.data() for record in result]



    def fetch_neighbor_chunk_ids(
        self,
        chunk_ids: list[str],
        per_seed_limit: int,
    ) -> list[str]:
        if not chunk_ids:
            return []

        cypher = """
        UNWIND $chunk_ids AS seed_id
        MATCH (seed:Chunk {id: seed_id})
        OPTIONAL MATCH (previous:Chunk)-[:NEXT]->(seed)
        WITH seed_id, collect(previous.id)[0..$per_seed_limit] AS previous_ids
        MATCH (seed:Chunk {id: seed_id})
        OPTIONAL MATCH (seed)-[:NEXT]->(next:Chunk)
        WITH previous_ids + collect(next.id)[0..$per_seed_limit] AS combined_ids
        UNWIND combined_ids AS chunk_id
        WITH DISTINCT chunk_id
        WHERE chunk_id IS NOT NULL
        RETURN chunk_id
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(
                cypher,
                chunk_ids=chunk_ids,
                per_seed_limit=per_seed_limit,
            )
            return [record["chunk_id"] for record in result]




    def fetch_shared_keyword_chunk_ids(
        self,
        chunk_ids: list[str],
        keywords: list[str],
        per_seed_limit: int,
    ) -> list[str]:
        if not chunk_ids or not keywords:
            return []

        cypher = """
        UNWIND $chunk_ids AS seed_id
        MATCH (seed:Chunk {id: seed_id})-[:HAS_KEYWORD]->(k:Keyword)<-[:HAS_KEYWORD]-(related:Chunk)
        WHERE related.id <> seed_id
          AND k.name IN $keywords
        WITH seed_id, related, count(DISTINCT k) AS shared_keyword_count
        ORDER BY shared_keyword_count DESC, related.chunk_index ASC
        WITH seed_id, collect(related.id)[0..$per_seed_limit] AS related_ids
        UNWIND related_ids AS chunk_id
        WITH DISTINCT chunk_id
        WHERE chunk_id IS NOT NULL
        RETURN chunk_id
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(
                cypher,
                chunk_ids=chunk_ids,
                keywords=keywords,
                per_seed_limit=per_seed_limit,
            )
            return [record["chunk_id"] for record in result]

