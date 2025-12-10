"""
Text builder for creating embeddable text from graph nodes.
"""

from __future__ import annotations

from typing import Any


class NodeTextBuilder:
    """
    Builds text representations of graph nodes for embedding.
    
    Combines multiple fields into a single searchable text that captures
    the semantic meaning of each node type.
    """
    
    @staticmethod
    def build_table_text(node: dict[str, Any]) -> str:
        """
        Build embeddable text for a Table node.
        
        Format: {table_name} | {business_name} | {description} | {grain} | tags: {tags}
        """
        parts = [
            node.get("table_name", ""),
            node.get("business_name", ""),
            node.get("description", ""),
            node.get("grain", ""),
        ]
        
        tags = node.get("tags", [])
        if tags:
            parts.append(f"tags: {' '.join(tags)}")
        
        return " | ".join(filter(None, parts))
    
    @staticmethod
    def build_column_text(node: dict[str, Any]) -> str:
        """
        Build embeddable text for a Column node.
        
        Format: {table_name}.{column_name} | {business_name} | {description} | {semantics}
        """
        table_name = node.get("table_name", "")
        column_name = node.get("column_name", "")
        full_name = f"{table_name}.{column_name}" if table_name else column_name
        
        parts = [
            full_name,
            node.get("business_name", ""),
            node.get("description", ""),
        ]
        
        semantics = node.get("semantics", [])
        if semantics:
            parts.append(f"semantics: {' '.join(semantics)}")
        
        unit = node.get("unit", "")
        if unit:
            parts.append(f"unit: {unit}")
        
        return " | ".join(filter(None, parts))
    
    @staticmethod
    def build_concept_text(node: dict[str, Any]) -> str:
        """
        Build embeddable text for a Concept node.
        
        Format: {name} | synonyms: {synonyms}
        """
        parts = [node.get("name", "")]
        
        synonyms = node.get("synonyms", [])
        if synonyms:
            parts.append(f"synonyms: {' '.join(synonyms)}")
        
        return " | ".join(filter(None, parts))
    
    @staticmethod
    def build_metric_text(node: dict[str, Any]) -> str:
        """
        Build embeddable text for a Metric node.
        
        Format: {name} | {business_name} | {description} | {expression} | {grain}
        """
        parts = [
            node.get("name", ""),
            node.get("business_name", ""),
            node.get("description", ""),
            f"expression: {node.get('expression', '')}",
            f"base_table: {node.get('base_table', '')}",
        ]
        
        tags = node.get("tags", [])
        if tags:
            parts.append(f"tags: {' '.join(tags)}")
        
        return " | ".join(filter(None, parts))
    
    @classmethod
    def build_text(cls, node: dict[str, Any], label: str) -> str:
        """
        Build embeddable text based on node label.
        
        Args:
            node: Node properties dictionary
            label: Node label (Table, Column, Concept, Metric)
            
        Returns:
            Embeddable text string
        """
        builders = {
            "Table": cls.build_table_text,
            "Column": cls.build_column_text,
            "Concept": cls.build_concept_text,
            "Metric": cls.build_metric_text,
        }
        
        builder = builders.get(label)
        if builder:
            return builder(node)
        
        # Fallback: concatenate all string values
        return " | ".join(
            str(v) for v in node.values() 
            if isinstance(v, str) and v
        )
