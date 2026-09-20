"""Neo4j Knowledge Graph store with parameterized Cypher queries."""

import logging
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase, Driver
from app.config import settings

logger = logging.getLogger(__name__)


class Neo4jGraphStore:
    """Knowledge graph storage and retrieval powered by Neo4j."""

    def __init__(self) -> None:
        self.driver: Optional[Driver] = None
        self._connected: bool = False
        self._init_driver()

    def _init_driver(self) -> None:
        """Initialize Neo4j driver if credentials are provided."""
        if not settings.NEO4J_URI or not settings.NEO4J_PASSWORD:
            logger.info("Neo4j URI or password not configured. Running in offline/mock mode.")
            self._connected = False
            return

        try:
            auth = (settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD) if settings.NEO4J_USERNAME else None
            self.driver = GraphDatabase.driver(settings.NEO4J_URI, auth=auth)
            # Verify connectivity
            self.driver.verify_connectivity()
            self._connected = True
            logger.info("Successfully connected to Neo4j graph database.")
            self._setup_schema()
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j database: {e}. Running without active graph DB.")
            self._connected = False
            self.driver = None

    def _setup_schema(self) -> None:
        """Create basic indexes or constraints safely."""
        if not self._connected or not self.driver:
            return
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        ]
        try:
            with self.driver.session(database=settings.NEO4J_DATABASE) as session:
                for q in queries:
                    try:
                        session.run(q)
                    except Exception as e:
                        logger.debug(f"Schema constraint notice: {e}")
        except Exception as e:
            logger.warning(f"Schema initialization warning: {e}")

    def is_available(self) -> bool:
        """Check if Neo4j is connected and reachable."""
        if not self.driver:
            return False
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def add_document_and_chunks(
        self, doc_id: str, doc_name: str, chunks: List[Dict[str, Any]]
    ) -> bool:
        """Store Document node, Chunk nodes, and HAS_CHUNK relationships."""
        if not self.is_available():
            logger.debug("Neo4j unavailable; skipping graph insertion for document.")
            return False

        try:
            with self.driver.session(database=settings.NEO4J_DATABASE) as session:
                # Merge Document node
                session.run(
                    "MERGE (d:Document {id: $doc_id}) ON CREATE SET d.name = $doc_name",
                    doc_id=doc_id,
                    doc_name=doc_name,
                )
                # Merge Chunks and connect to Document
                for c in chunks:
                    session.run(
                        """
                        MATCH (d:Document {id: $doc_id})
                        MERGE (c:Chunk {id: $chunk_id})
                        ON CREATE SET c.text = $text, c.page = $page, c.doc_id = $doc_id, c.doc_name = $doc_name
                        MERGE (d)-[:HAS_CHUNK]->(c)
                        """,
                        doc_id=doc_id,
                        chunk_id=c["chunk_id"],
                        text=c["text"][:500],  # Store snippet for graph context
                        page=c["page"],
                        doc_name=doc_name,
                    )
            return True
        except Exception as e:
            logger.error(f"Error adding document and chunks to Neo4j: {e}")
            return False

    def add_entities_and_relations(
        self, chunk_id: str, entities: List[str], relationships: List[Dict[str, str]]
    ) -> bool:
        """Store extracted entities, chunk MENTIONS, and RELATED_TO links."""
        if not self.is_available():
            return False

        try:
            with self.driver.session(database=settings.NEO4J_DATABASE) as session:
                # Add entities and link to Chunk
                for ent in entities:
                    ent_clean = ent.strip()
                    if not ent_clean:
                        continue
                    session.run(
                        """
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (e:Entity {name: $entity_name})
                        MERGE (c)-[:MENTIONS]->(e)
                        """,
                        chunk_id=chunk_id,
                        entity_name=ent_clean,
                    )

                # Add relationships between entities
                for rel in relationships:
                    source = rel.get("source", "").strip()
                    target = rel.get("target", "").strip()
                    rel_type = rel.get("relationship", "RELATED_TO").strip().upper()
                    # Sanitize relationship type string to valid Cypher identifier
                    clean_rel_type = "".join(ch if ch.isalnum() else "_" for ch in rel_type) or "RELATED_TO"

                    if source and target and source.lower() != target.lower():
                        session.run(
                            """
                            MERGE (source:Entity {name: $source_name})
                            MERGE (target:Entity {name: $target_name})
                            MERGE (source)-[r:RELATED_TO {type: $rel_type}]->(target)
                            """,
                            source_name=source,
                            target_name=target,
                            rel_type=clean_rel_type,
                        )
            return True
        except Exception as e:
            logger.error(f"Error adding entities and relations to Neo4j: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get database node counts and connection status."""
        if not self.is_available():
            return {
                "status": "unavailable",
                "message": "Neo4j is not configured or reachable.",
            }

        try:
            with self.driver.session(database=settings.NEO4J_DATABASE) as session:
                doc_cnt = session.run("MATCH (d:Document) RETURN count(d) AS count").single()["count"]
                chunk_cnt = session.run("MATCH (c:Chunk) RETURN count(c) AS count").single()["count"]
                ent_cnt = session.run("MATCH (e:Entity) RETURN count(e) AS count").single()["count"]

                return {
                    "status": "connected",
                    "document_count": doc_cnt,
                    "chunk_count": chunk_cnt,
                    "entity_count": ent_cnt,
                    "message": "Neo4j graph database is active and operational.",
                }
        except Exception as e:
            return {
                "status": "unavailable",
                "message": f"Failed to retrieve graph status: {str(e)}",
            }

    def get_entity_and_related(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Fetch an entity and its directly connected neighbors."""
        if not self.is_available():
            return None

        try:
            with self.driver.session(database=settings.NEO4J_DATABASE) as session:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE toLower(e.name) = toLower($entity_name)
                    OPTIONAL MATCH (e)-[r:RELATED_TO]-(related:Entity)
                    RETURN e.name AS entity, collect(DISTINCT related.name) AS related_entities
                    """,
                    entity_name=entity_name,
                ).single()

                if result and result["entity"]:
                    return {
                        "entity": result["entity"],
                        "related_entities": [r for r in result["related_entities"] if r],
                    }
                return None
        except Exception as e:
            logger.error(f"Error fetching entity '{entity_name}': {e}")
            return None

    def search_graph_for_entities(self, entity_names: List[str]) -> List[Dict[str, Any]]:
        """Search graph for query entities, connected entities, and mentioning chunks."""
        if not self.is_available() or not entity_names:
            return []

        try:
            with self.driver.session(database=settings.NEO4J_DATABASE) as session:
                query = """
                UNWIND $entity_names AS q_name
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower(q_name) OR toLower(q_name) CONTAINS toLower(e.name)
                OPTIONAL MATCH (e)-[r:RELATED_TO]-(related:Entity)
                OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e)
                RETURN e.name AS entity,
                       collect(DISTINCT related.name)[..5] AS related_entities,
                       collect(DISTINCT {
                           chunk_id: c.id,
                           doc_name: c.doc_name,
                           page: c.page,
                           text: c.text
                       })[..2] AS chunks
                LIMIT 8
                """
                results = session.run(query, entity_names=entity_names)
                graph_data = []
                for record in results:
                    graph_data.append({
                        "entity": record["entity"],
                        "related_entities": [r for r in record["related_entities"] if r],
                        "chunks": [c for c in record["chunks"] if c.get("chunk_id")],
                    })
                return graph_data
        except Exception as e:
            logger.error(f"Error during graph retrieval: {e}")
            return []

    def close(self) -> None:
        """Close database driver connection."""
        if self.driver:
            self.driver.close()
            self._connected = False


# Global singleton instance
graph_store = Neo4jGraphStore()
